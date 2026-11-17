"""Extract tracked face crops (and matching full frames) from the videos of a dataset manifest.

For every video in the manifest (default manifests/ffpp_c23.csv; --dataset selects another) this samples N frames at a fixed
stride from the middle of the video, tracks one face across them, and saves

    <DATA_ROOT>/ffpp_c23/faces/<method>/<video>/<frame_idx:05d>.jpg      224x224 face crop
    <DATA_ROOT>/ffpp_c23/fullframe/<method>/<video>/<frame_idx:05d>.jpg  224x224 whole frame (direct resize)
    <DATA_ROOT>/ffpp_c23/faces/<method>/<video>/meta.json                frame indices, boxes, detection rate
    <DATA_ROOT>/ffpp_c23/frame_index.csv                                 one row per video (written at the end)

Face tracking: YuNet (OpenCV) runs on every sampled frame; the track starts at
the largest face and follows the detection closest to the previous box.
Missing detections are interpolated, centres/sizes are smoothed over time, and
a square crop of `--margin` x the face size is taken. A video is marked
`failed` (and has no crops) if fewer than `--min-det-rate` of its frames have
a detection.

The run is resumable: videos with a meta.json are skipped unless --overwrite.

usage:  python scripts/extract_faces.py [--dataset ffpp_c23|celebdf_v1|celebdf_v2] [--workers 6] [--limit 20] [--frames 32]
"""
import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import paths  # noqa: E402

OUT_SIZE = 224
JPEG_QUALITY = 95
DET_MAX_SIDE = 640  # frames are downscaled to this for detection only

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(paths.FACE_DETECTOR_MODEL), "", (320, 320), 0.6, 0.3, 50)
    return _detector


def choose_frame_indices(n_frames, n_out, max_stride):
    """n_out indices at a constant stride, centred in the video."""
    if n_frames < n_out:
        return list(range(n_frames)), 1
    stride = max(1, min(max_stride, n_frames // n_out))
    span = (n_out - 1) * stride + 1
    start = (n_frames - span) // 2
    return [start + i * stride for i in range(n_out)], stride


def read_frames(path, indices):
    """Sequential decode (reliable for all codecs); returns {index: frame}."""
    cap = cv2.VideoCapture(str(path))
    wanted, last, out, pos = set(indices), max(indices), {}, 0
    while pos <= last:
        if not cap.grab():
            break
        if pos in wanted:
            ok, frame = cap.retrieve()
            if ok:
                out[pos] = frame
        pos += 1
    cap.release()
    return out


def detect(frame):
    """All faces in a frame as rows [x, y, w, h, score] in full-resolution pixels."""
    h, w = frame.shape[:2]
    scale = min(1.0, DET_MAX_SIDE / max(h, w))
    small = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame
    det = get_detector()
    det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = det.detect(small)
    if faces is None:
        return np.zeros((0, 5), np.float32)
    out = faces[:, [0, 1, 2, 3, 14]].astype(np.float32)
    out[:, :4] /= scale
    return out


def track(dets_per_frame):
    """Pick one face per frame -> array [T, 3] of (cx, cy, size), NaN where missing."""
    T = len(dets_per_frame)
    trk = np.full((T, 3), np.nan, np.float32)
    prev = None
    for t, dets in enumerate(dets_per_frame):
        if len(dets) == 0:
            continue
        cx, cy = dets[:, 0] + dets[:, 2] / 2, dets[:, 1] + dets[:, 3] / 2
        size = np.maximum(dets[:, 2], dets[:, 3])
        if prev is None:
            k = int(np.argmax(dets[:, 2] * dets[:, 3]))  # start on the largest face
        else:
            dist = np.hypot(cx - prev[0], cy - prev[1])
            k = int(np.argmin(dist))
            if dist[k] > 1.5 * prev[2]:  # a different face, not our track
                continue
        trk[t] = prev = (cx[k], cy[k], size[k])
    return trk


def fill_and_smooth(trk, window=5):
    t = np.arange(len(trk))
    ok = ~np.isnan(trk[:, 0])
    out = np.empty_like(trk)
    for c in range(3):
        col = np.interp(t, t[ok], trk[ok, c])
        pad = window // 2
        padded = np.pad(col, pad, mode="edge")
        out[:, c] = np.convolve(padded, np.ones(window) / window, mode="valid")
    return out


def crop_square(frame, cx, cy, side):
    h, w = frame.shape[:2]
    half = side / 2
    x0, y0, x1, y1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    crop = frame[y0:y1, x0:x1]
    interp = cv2.INTER_AREA if crop.shape[0] > OUT_SIZE else cv2.INTER_CUBIC
    return cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=interp)


def process(job):
    row, cfg = job
    video_id = row["video_id"]
    face_dir = cfg["faces_dir"] / video_id
    full_dir = cfg["full_dir"] / video_id
    meta_path = face_dir / "meta.json"
    if meta_path.exists() and not cfg["overwrite"]:
        return json.loads(meta_path.read_text())["summary"]

    t0 = time.time()
    summary = {"video_id": video_id, "status": "ok", "n_saved": 0, "stride": 0, "det_rate": 0.0,
               "median_face_px": 0, "seconds": 0.0, "error": ""}
    try:
        n_frames = int(row["n_frames"])
        indices, stride = choose_frame_indices(n_frames, cfg["frames"], cfg["max_stride"])
        frames = read_frames(cfg["ffpp_root"] / row["rel_path"], indices)
        indices = [i for i in indices if i in frames]
        if len(indices) < cfg["min_frames"]:
            raise RuntimeError(f"only {len(indices)} frames decoded")

        trk = track([detect(frames[i]) for i in indices])
        det_rate = float((~np.isnan(trk[:, 0])).mean())
        summary.update(stride=stride, det_rate=round(det_rate, 3))
        if det_rate < cfg["min_det_rate"]:
            raise RuntimeError(f"face found in only {det_rate:.0%} of frames")

        smooth = fill_and_smooth(trk)
        face_dir.mkdir(parents=True, exist_ok=True)
        full_dir.mkdir(parents=True, exist_ok=True)
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        boxes = []
        for i, (cx, cy, size) in zip(indices, smooth):
            side = float(size) * cfg["margin"]
            cv2.imwrite(str(face_dir / f"{i:05d}.jpg"), crop_square(frames[i], cx, cy, side), params)
            full = cv2.resize(frames[i], (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(full_dir / f"{i:05d}.jpg"), full, params)
            boxes.append([round(float(cx), 1), round(float(cy), 1), round(side, 1)])
        summary.update(n_saved=len(indices), median_face_px=int(np.median(smooth[:, 2])))
        meta = {"frame_indices": indices, "crop_boxes_cx_cy_side": boxes,
                "detected": (~np.isnan(trk[:, 0])).tolist()}
    except Exception as exc:  # keep going; the failure is recorded
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        meta = {}
        face_dir.mkdir(parents=True, exist_ok=True)
    summary["seconds"] = round(time.time() - t0, 2)
    meta_path.write_text(json.dumps({"summary": summary, "config": cfg["public"], **meta}))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(paths.DATASETS), default="ffpp_c23")
    ap.add_argument("--frames", type=int, default=32, help="frames per video")
    ap.add_argument("--max-stride", type=int, default=8, help="sample every k-th frame (reduced for short videos)")
    ap.add_argument("--margin", type=float, default=1.3, help="crop side = margin x face size")
    ap.add_argument("--min-det-rate", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos per method (for testing)")
    ap.add_argument("--out-root", type=Path, default=None, help="default <DATA_ROOT>/<dataset>")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ds = paths.DATASETS[args.dataset]
    if not ds["manifest"].is_file():
        sys.exit(f"Manifest not found: {ds['manifest']}. Build it first (scripts/build_manifest*.py).")
    if not paths.FACE_DETECTOR_MODEL.is_file():
        sys.exit(f"Face detector model not found: {paths.FACE_DETECTOR_MODEL}")
    rows = list(csv.DictReader(open(ds["manifest"], encoding="utf-8")))
    if args.limit:
        seen, kept = {}, []
        for r in rows:
            seen[r["method"]] = seen.get(r["method"], 0) + 1
            if seen[r["method"]] <= args.limit:
                kept.append(r)
        rows = kept

    out_root = args.out_root or ds["data_dir"]
    public = {"frames": args.frames, "max_stride": args.max_stride, "margin": args.margin,
              "min_det_rate": args.min_det_rate, "out_size": OUT_SIZE, "jpeg_quality": JPEG_QUALITY,
              "detector": paths.FACE_DETECTOR_MODEL.name}
    cfg = {**public, "public": public, "min_frames": min(8, args.frames), "overwrite": args.overwrite,
           "ffpp_root": ds["root"], "faces_dir": out_root / "faces", "full_dir": out_root / "fullframe"}
    print("Reading videos from :", ds["root"])
    print("Face crops ->       :", cfg["faces_dir"])
    print("Full frames ->      :", cfg["full_dir"])
    print(f"Videos: {len(rows)} | frames/video: {args.frames} | workers: {args.workers}", flush=True)

    t0, results = time.time(), []
    with Pool(args.workers) as pool:
        for n, summary in enumerate(pool.imap_unordered(process, [(r, cfg) for r in rows], chunksize=4), 1):
            results.append(summary)
            if n % 100 == 0 or n == len(rows):
                failed = sum(s["status"] != "ok" for s in results)
                rate = n / (time.time() - t0)
                print(f"[{n}/{len(rows)}] failed={failed} {rate:.1f} videos/s "
                      f"eta={(len(rows) - n) / rate / 60:.0f} min", flush=True)

    results.sort(key=lambda s: s["video_id"])
    index_path = out_root / "frame_index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(results)
    failed = [s for s in results if s["status"] != "ok"]
    print("Saved ->", index_path)
    print(f"Done in {(time.time() - t0) / 60:.1f} min: {len(results) - len(failed)} ok, {len(failed)} failed")
    for s in failed[:20]:
        print("  FAILED", s["video_id"], "-", s["error"])


if __name__ == "__main__":
    main()
