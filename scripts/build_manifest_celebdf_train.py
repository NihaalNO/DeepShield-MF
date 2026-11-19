"""Build a train/validation manifest from the Celeb-DF v2 videos outside the official test lists.

Used to fine-tune on a second domain (README Step 9). Never listed:
  - every video in Celeb-DF v2's List_of_testing_videos.txt (the held-out test set)
  - every video in Celeb-DF v1's test list (v1 is a byte-identical subset of v2, so v1's test
    videos would otherwise leak into training)

Split, decided before any training:
  - validation identities: --val-ids celebrities drawn with --seed. A Celeb-real video of such an
    identity, and every Celeb-synthesis video that involves it (as target or source), goes to
    validation; no training video shares an identity with validation.
  - YouTube-real videos (no identity label) go to validation with probability --val-frac, by a
    hash of the file name.
  - fakes are subsampled per split (--max-train-fakes, --max-val-fakes), spread evenly over the
    target identities, so face extraction and training stay affordable. Every real is kept.

Identities of the official test list overlap the training identities (54 of 59 celebrities appear
in the test list), so Celeb-DF test results after this fine-tuning are same-dataset results.

Columns as manifests/ffpp_c23.csv; method is "original" for reals and "Celeb-DF" for fakes.
Writes  manifests/celebdf_v2_train.csv and manifests/celebdf_v2_train_summary.json

usage:  python scripts/build_manifest_celebdf_train.py [--val-ids 6] [--max-train-fakes 1500] [--max-val-fakes 300]
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepshield import paths  # noqa: E402

FIELDS = ["video_id", "rel_path", "label", "label_idx", "method", "target_id", "source_id",
          "family_id", "split", "n_frames", "fps", "width", "height", "size_bytes"]
FOLDERS = {"Celeb-real": "real", "YouTube-real": "real", "Celeb-synthesis": "fake"}


def test_list(root):
    f = root / "List_of_testing_videos.txt"
    return {line.split()[1] for line in f.read_text().split("\n") if line.strip()}


def even_sample(items, k, key, seed):
    """Up to k items spread evenly over the groups given by key (round robin over shuffled groups)."""
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups)
    rng.shuffle(order)
    out = []
    while len(out) < k and any(groups.values()):
        for g in order:
            if groups[g] and len(out) < k:
                out.append(groups[g].pop())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-ids", type=int, default=6, help="celebrity identities reserved for validation")
    ap.add_argument("--val-frac", type=float, default=0.15, help="share of YouTube-real videos in validation")
    ap.add_argument("--max-train-fakes", type=int, default=1500)
    ap.add_argument("--max-val-fakes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    v2 = paths.DATASETS["celebdf_v2"]["root"]
    v1 = paths.DATASETS["celebdf_v1"]["root"]
    excluded = test_list(v2) | test_list(v1)
    videos = []
    for folder, label in FOLDERS.items():
        for f in sorted((v2 / folder).glob("*.mp4")):
            rel = f"{folder}/{f.name}"
            if rel in excluded:
                continue
            parts = f.stem.split("_")
            if folder == "YouTube-real":
                target = source = f"yt_{f.stem}"
            elif folder == "Celeb-real":
                target = source = parts[0]
            else:
                target, source = parts[0], parts[1]
            videos.append({"rel": rel, "folder": folder, "label": label, "target": target, "source": source})

    celebs = sorted({v["target"] for v in videos if v["folder"] != "YouTube-real"} |
                    {v["source"] for v in videos if v["folder"] == "Celeb-synthesis"})
    val_ids = set(random.Random(args.seed).sample(celebs, args.val_ids))

    def split_of(v):
        if v["folder"] == "YouTube-real":
            h = int(hashlib.sha1(v["rel"].encode()).hexdigest(), 16) % 10_000
            return "val" if h < args.val_frac * 10_000 else "train"
        return "val" if {v["target"], v["source"]} & val_ids else "train"

    for v in videos:
        v["split"] = split_of(v)
    chosen = []
    for split, cap in (("train", args.max_train_fakes), ("val", args.max_val_fakes)):
        chosen += [v for v in videos if v["split"] == split and v["label"] == "real"]
        fakes = [v for v in videos if v["split"] == split and v["label"] == "fake"]
        chosen += even_sample(fakes, cap, key=lambda v: v["target"], seed=args.seed)

    rows, problems = [], []
    for v in sorted(chosen, key=lambda v: v["rel"]):
        path = v2 / v["rel"]
        cap = cv2.VideoCapture(str(path))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), round(cap.get(cv2.CAP_PROP_FPS), 3)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if n <= 0:
            problems.append(f"{v['rel']}: cannot be opened")
            continue
        rows.append({"video_id": f"{v['folder']}/{Path(v['rel']).stem}", "rel_path": v["rel"], "label": v["label"],
                     "label_idx": paths.CLASS_TO_IDX[v["label"]],
                     "method": "original" if v["label"] == "real" else "Celeb-DF",
                     "target_id": v["target"], "source_id": v["source"], "family_id": v["target"], "split": v["split"],
                     "n_frames": n, "fps": fps, "width": w, "height": h, "size_bytes": path.stat().st_size})

    ds = paths.DATASETS["celebdf_v2_train"]
    with open(ds["manifest"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    by = Counter((r["split"], r["label"]) for r in rows)
    train_ids = {r["target_id"] for r in rows if r["split"] == "train"} | {r["source_id"] for r in rows if r["split"] == "train"}
    summary = {"dataset": "Celeb-DF v2, videos outside the v1 and v2 official test lists",
               "seed": args.seed, "val_identities": sorted(val_ids),
               "identity_overlap_train_val": sorted((train_ids & val_ids)),
               "videos": len(rows), "by_split_and_label": {f"{s}/{l}": n for (s, l), n in sorted(by.items())},
               "available_before_sampling": dict(Counter((v["split"] + "/" + v["label"]) for v in videos)),
               "excluded_test_videos": len(excluded), "problems": problems,
               "use": "fine-tuning (train) and model selection / threshold (val); never the official test lists"}
    summary_path = ds["manifest"].with_name(ds["manifest"].stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved ->", ds["manifest"])
    print(json.dumps(summary, indent=2))
    if problems or summary["identity_overlap_train_val"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
