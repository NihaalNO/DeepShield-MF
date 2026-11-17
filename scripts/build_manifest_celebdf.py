"""Build the manifest of a Celeb-DF official test list (cross-dataset evaluation only).

Only the videos named in List_of_testing_videos.txt are listed, all with split "test": the
project never trains or tunes on Celeb-DF. The columns are those of manifests/ffpp_c23.csv.

Labels come from the folder, not from the number in the list (the list uses 1 = real, 0 = fake,
the opposite of this project's class indices); the two are cross-checked.

    Celeb-real/idA_NNNN.mp4          real   family = idA
    YouTube-real/NNNNN.mp4           real   family = yt_NNNNN
    Celeb-synthesis/idA_idB_NNNN.mp4 fake   family = idA (target identity)

Writes  manifests/celebdf_<version>.csv, manifests/celebdf_<version>_summary.json

usage:  python scripts/build_manifest_celebdf.py --version v2
"""
import argparse
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
FOLDER_LABEL = {"Celeb-real": "real", "YouTube-real": "real", "Celeb-synthesis": "fake"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", choices=["v1", "v2"], required=True)
    args = ap.parse_args()
    ds = paths.DATASETS[f"celebdf_{args.version}"]
    root, list_file = ds["root"], ds["root"] / "List_of_testing_videos.txt"
    if not list_file.is_file():
        sys.exit(f"Not found: {list_file}. Set DEEPSHIELD_CELEBDF_{args.version.upper()}_DIR in .env.")

    rows, problems = [], []
    for line in list_file.read_text().split("\n"):
        if not line.strip():
            continue
        list_label, rel = line.split()
        folder, name = rel.split("/")
        stem, label = Path(name).stem, FOLDER_LABEL[folder]
        if (list_label == "1") != (label == "real"):
            problems.append(f"{rel}: list label {list_label} disagrees with folder {folder}")
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: file missing")
            continue
        parts = stem.split("_")
        if folder == "YouTube-real":
            target = source = f"yt_{stem}"
        elif folder == "Celeb-real":
            target = source = parts[0]
        else:
            target, source = parts[0], parts[1]
        cap = cv2.VideoCapture(str(path))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), round(cap.get(cv2.CAP_PROP_FPS), 3)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if n <= 0:
            problems.append(f"{rel}: cannot be opened")
        rows.append({"video_id": f"{folder}/{stem}", "rel_path": rel, "label": label,
                     "label_idx": paths.CLASS_TO_IDX[label], "method": "original" if label == "real" else "Celeb-DF",
                     "target_id": target, "source_id": source, "family_id": target, "split": "test",
                     "n_frames": n, "fps": fps, "width": w, "height": h, "size_bytes": path.stat().st_size})
    rows.sort(key=lambda r: r["video_id"])
    if len({r["video_id"] for r in rows}) != len(rows):
        problems.append("duplicate videos in the list")

    with open(ds["manifest"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = {"dataset": f"Celeb-DF {args.version}", "source_list": "List_of_testing_videos.txt",
               "videos": len(rows), "by_label": dict(Counter(r["label"] for r in rows)),
               "by_folder": dict(Counter(r["rel_path"].split("/")[0] for r in rows)),
               "families": len({r["family_id"] for r in rows}),
               "frames_min_median_max": [min(r["n_frames"] for r in rows),
                                         sorted(r["n_frames"] for r in rows)[len(rows) // 2],
                                         max(r["n_frames"] for r in rows)],
               "use": "cross-dataset test only; never used for training, model selection or thresholds",
               "problems": problems}
    summary_path = ds["manifest"].with_name(ds["manifest"].stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved ->", ds["manifest"])
    print("Saved ->", summary_path)
    print(json.dumps(summary, indent=2))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
