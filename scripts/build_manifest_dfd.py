"""Build the manifest of FaceForensics++'s DeepFakeDetection set (DFD): an unseen-dataset test set.

DFD (Google / Jigsaw) has paid, consenting actors filmed in scripted scenes, and deepfakes made
between pairs of them. None of its actors or videos appear in FF++'s YouTube videos or in Celeb-DF,
so no model in this project has seen it. Every video is listed with split "test".

    original_sequences/actors/c23/videos/AA__scene.mp4                 real  family = AA
    manipulated_sequences/DeepFakeDetection/c23/videos/AA_BB__scene__ID.mp4  fake  family = AA (target actor)

Non-video files (e.g. tmp* leftovers of an interrupted download) are skipped and listed in the summary.
Columns as manifests/ffpp_c23.csv. Writes manifests/dfd_c23.csv and manifests/dfd_c23_summary.json.

usage:  python scripts/build_manifest_dfd.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import paths  # noqa: E402

FIELDS = ["video_id", "rel_path", "label", "label_idx", "method", "target_id", "source_id",
          "family_id", "split", "n_frames", "fps", "width", "height", "size_bytes"]
FOLDERS = {"original_sequences/actors/c23/videos": ("real", "original"),
           "manipulated_sequences/DeepFakeDetection/c23/videos": ("fake", "DeepFakeDetection")}


def main():
    ds = paths.DATASETS["dfd_c23"]
    root = ds["root"]
    rows, skipped, problems = [], [], []
    for folder, (label, method) in FOLDERS.items():
        d = root / folder
        if not d.is_dir():
            sys.exit(f"Not found: {d}. Download DFD with the FaceForensics++ script (DeepFakeDetection, "
                     "DeepFakeDetection_original, c23) into DEEPSHIELD_FFPP_DIR.")
        for f in sorted(d.iterdir()):
            if f.suffix.lower() != ".mp4":
                skipped.append(f"{folder}/{f.name}")
                continue
            head = f.stem.split("__")[0]
            actors = head.split("_")
            target, source = (actors[0], actors[0]) if label == "real" else (actors[0], actors[1])
            cap = cv2.VideoCapture(str(f))
            n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), round(cap.get(cv2.CAP_PROP_FPS), 3)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if n <= 0:
                problems.append(f"{folder}/{f.name}: cannot be opened")
                continue
            vid = f"{'actors' if label == 'real' else 'DeepFakeDetection'}/{f.stem}"
            rows.append({"video_id": vid, "rel_path": f"{folder}/{f.name}", "label": label,
                         "label_idx": paths.CLASS_TO_IDX[label], "method": method, "target_id": f"dfd{target}",
                         "source_id": f"dfd{source}", "family_id": f"dfd{target}", "split": "test",
                         "n_frames": n, "fps": fps, "width": w, "height": h, "size_bytes": f.stat().st_size})
    with open(ds["manifest"], "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = {"dataset": "FaceForensics++ DeepFakeDetection (DFD), c23", "videos": len(rows),
               "by_label": dict(Counter(r["label"] for r in rows)), "actors": len({r["target_id"] for r in rows}),
               "families": len({r["family_id"] for r in rows}),
               "frames_min_median_max": [min(r["n_frames"] for r in rows), sorted(r["n_frames"] for r in rows)[len(rows) // 2],
                                         max(r["n_frames"] for r in rows)],
               "resolution": dict(Counter(f"{r['width']}x{r['height']}" for r in rows).most_common(4)),
               "skipped_non_video_files": skipped, "problems": problems,
               "use": "unseen-dataset test only; never used for training, model selection or thresholds"}
    summary_path = ds["manifest"].with_name(ds["manifest"].stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved ->", ds["manifest"])
    print(json.dumps(summary, indent=2))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
