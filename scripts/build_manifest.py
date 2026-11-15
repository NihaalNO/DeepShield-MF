"""Build the FaceForensics++ (c23) manifest with source families and the official split.

Reads   <FFPP_ROOT>/original_sequences/youtube/c23/videos/NNN.mp4
        <FFPP_ROOT>/manipulated_sequences/<Method>/c23/videos/TTT_SSS.mp4
        manifests/ffpp_official_splits/{train,val,test}.json  (official source-pair split)
Writes  manifests/ffpp_c23.csv            one row per video
        manifests/ffpp_c23_summary.json   counts per split/method + integrity checks

A fake `TTT_SSS` uses original TTT as target (background/pose) and SSS as source
(identity/expression). Both originals and every fake made from them form one
"source family"; a family is never divided between train/val/test.

usage:  python scripts/build_manifest.py [--ffpp-root PATH] [--no-probe]
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import paths  # noqa: E402

METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
COMPRESSION = "c23"
ORIG_RE = re.compile(r"^(\d{3})\.mp4$")
FAKE_RE = re.compile(r"^(\d{3})_(\d{3})\.mp4$")
FIELDS = ["video_id", "rel_path", "label", "label_idx", "method", "target_id", "source_id",
          "family_id", "split", "n_frames", "fps", "width", "height", "size_bytes"]


def load_official_families():
    """id -> (family_id, split) from the official pair lists."""
    fam = {}
    for split in ("train", "val", "test"):
        pairs = json.loads((paths.FFPP_OFFICIAL_SPLITS / f"{split}.json").read_text())
        for a, b in pairs:
            family_id = "_".join(sorted((a, b)))
            for vid in (a, b):
                if vid in fam:
                    raise ValueError(f"original {vid} appears in two official pairs")
                fam[vid] = (family_id, split)
    return fam


def probe(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    info = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), round(cap.get(cv2.CAP_PROP_FPS), 3),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ffpp-root", type=Path, default=paths.FFPP_ROOT)
    ap.add_argument("--no-probe", action="store_true", help="skip reading frame count/fps/size from each video")
    args = ap.parse_args()
    root = args.ffpp_root
    if not root.is_dir():
        sys.exit(f"FaceForensics++ not found at {root}. Set DEEPSHIELD_FFPP_DIR or pass --ffpp-root.")
    print("FF++ root :", root)

    fam = load_official_families()
    rows, skipped, problems = [], [], []

    def add(rel, label, method, target, source):
        f = root / rel
        for vid in (target, source):
            if vid not in fam:
                problems.append(f"{rel}: original {vid} not in official splits")
                return
        (family_t, split_t), (family_s, _) = fam[target], fam[source]
        if family_t != family_s:
            problems.append(f"{rel}: target and source are in different official families")
            return
        meta = ("", "", "", "") if args.no_probe else probe(f)
        if meta is None:
            problems.append(f"{rel}: cannot be opened by OpenCV")
            return
        rows.append(dict(zip(FIELDS, [
            f"{method}/{f.stem}", rel.as_posix(), label, paths.CLASS_TO_IDX[label], method, target, source,
            family_t, split_t, *meta, f.stat().st_size])))

    sources = [("original", Path("original_sequences/youtube") / COMPRESSION / "videos")]
    sources += [(m, Path("manipulated_sequences") / m / COMPRESSION / "videos") for m in METHODS]
    for method, rel_dir in sources:
        d = root / rel_dir
        if not d.is_dir():
            problems.append(f"missing folder: {rel_dir.as_posix()}")
            continue
        for f in sorted(d.iterdir()):
            m_o, m_f = ORIG_RE.match(f.name), FAKE_RE.match(f.name)
            if method == "original" and m_o:
                add(rel_dir / f.name, "real", method, m_o.group(1), m_o.group(1))
            elif method != "original" and m_f:
                add(rel_dir / f.name, "fake", method, m_f.group(1), m_f.group(2))
            else:
                skipped.append((rel_dir / f.name).as_posix())
        print(f"  {method:15s} {sum(r['method'] == method for r in rows):5d} videos", flush=True)

    # ---- integrity: no family may span splits --------------------------------
    fam_splits = {}
    for r in rows:
        fam_splits.setdefault(r["family_id"], set()).add(r["split"])
    leaking = [k for k, v in fam_splits.items() if len(v) > 1]

    paths.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.FFPP_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "dataset": "FaceForensics++", "compression": COMPRESSION,
        "split_source": "official FaceForensics++ source-pair split (manifests/ffpp_official_splits)",
        "class_names": paths.CLASS_NAMES,
        "n_videos": len(rows),
        "n_families": len(fam_splits),
        "families_spanning_splits": len(leaking),
        "videos_per_split": dict(Counter(r["split"] for r in rows)),
        "families_per_split": dict(Counter(next(iter(v)) for v in fam_splits.values())),
        "videos_per_split_and_method": {s: dict(Counter(r["method"] for r in rows if r["split"] == s))
                                        for s in ("train", "val", "test")},
        "skipped_non_video_files": skipped,
        "problems": problems,
    }
    out = paths.MANIFEST_DIR / "ffpp_c23_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Saved ->", paths.FFPP_MANIFEST)
    print("Saved ->", out)
    print(json.dumps({k: summary[k] for k in ("n_videos", "n_families", "families_spanning_splits",
                                                "videos_per_split", "skipped_non_video_files")}, indent=2))
    if problems:
        print(f"WARNING: {len(problems)} problem(s), first: {problems[:5]}")
    if leaking or problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
