"""Cached-feature dataset for FF++: manifest rows, splits and window batches."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import paths

SPLITS = ("train", "val", "test")


class FeatureStore:
    """Read-only view of <DATA_ROOT>/<dataset>/features/<variant>/ joined with the dataset manifest."""

    def __init__(self, variant="faces", root=None, in_memory=True, gray_in_memory=False, dataset="ffpp_c23",
                 cnn_backbone="resnet50"):
        self.variant, self.dataset = variant, dataset
        ds = paths.DATASETS[dataset]
        self.dir = Path(root or ds["data_dir"] / "features") / variant
        if not (self.dir / "meta.json").is_file():
            raise FileNotFoundError(f"No feature cache at {self.dir}. Run scripts/cache_features.py "
                                    f"--dataset {dataset} --variant {variant}.")
        self.meta = json.loads((self.dir / "meta.json").read_text())
        done = np.load(self.dir / "done.npy")
        if not done.all():
            raise RuntimeError(f"Feature cache at {self.dir} is incomplete ({int(done.sum())}/{len(done)} videos).")

        manifest = {r["video_id"]: r for r in csv.DictReader(open(ds["manifest"], encoding="utf-8"))}
        self.rows = []
        for r in csv.DictReader(open(self.dir / "index.csv", encoding="utf-8")):
            m = manifest[r["video_id"]]
            self.rows.append({"row": int(r["row"]), "video_id": r["video_id"], "label_idx": int(m["label_idx"]),
                              "method": m["method"], "family_id": m["family_id"], "split": m["split"]})
        self.n_frames = self.meta["frames_per_video"]

        def load(name):
            arr = np.load(self.dir / f"{name}.npy", mmap_mode="r")
            return np.ascontiguousarray(arr) if in_memory else arr

        if not (self.dir / f"{cnn_backbone}.npy").is_file():
            raise FileNotFoundError(f"No {cnn_backbone} features in {self.dir}. Run scripts/cache_extra_backbone.py "
                                    f"--dataset {dataset} --variant {variant} --backbone {cnn_backbone}.")
        self.cnn = load(cnn_backbone)  # features fed to the model's "cnn" branch
        self.vit = load("vit_base_patch16_224")
        # Grayscale frames (FFT branch input, ~9.6 GB): on disk by default, in RAM when requested.
        self.gray = None
        if self.meta.get("gray"):
            self.gray = np.load(self.dir / "gray.npy", mmap_mode=None if gray_in_memory else "r")
        self.gray_in_memory = gray_in_memory and self.gray is not None

    def split_rows(self, split, allow_test=False, limit_families=0, methods=None):
        """Manifest rows of one split. The test split needs an explicit allow_test=True."""
        if split not in SPLITS:
            raise ValueError(split)
        if split == "test" and not allow_test:
            raise PermissionError("The test split is reserved for the final evaluation (allow_test=True).")
        rows = [r for r in self.rows if r["split"] == split]
        if methods is not None:  # reals ("original") are always kept
            rows = [r for r in rows if r["method"] == "original" or r["method"] in methods]
        if limit_families:
            keep = set(sorted({r["family_id"] for r in rows})[:limit_families])
            rows = [r for r in rows if r["family_id"] in keep]
        return rows


def window_starts(n_frames, seq_len, stride=None):
    """Start indices of the windows cut from one video (default: 50 % overlap)."""
    stride = stride or max(1, seq_len // 2)
    return list(range(0, n_frames - seq_len + 1, stride))


class WindowBatches:
    """Iterates batches of windows: dict(cnn, vit, gray, label, video) as tensors.

    One sample is (video, start): `seq_len` consecutive cached frames of a video.
    `video` is the position of the video in `rows`, used to aggregate window
    scores into one score per video.
    """

    def __init__(self, store, rows, seq_len=8, stride=None, batch_size=64, shuffle=False, seed=0,
                 use_gray=False, shuffle_frames=False, device="cpu"):
        self.store, self.rows, self.seq_len = store, rows, seq_len
        self.batch_size, self.shuffle, self.use_gray = batch_size, shuffle, use_gray
        self.shuffle_frames, self.device = shuffle_frames, device
        self.rng = np.random.default_rng(seed)
        starts = window_starts(store.n_frames, seq_len, stride)
        self.samples = np.array([(v, s) for v in range(len(rows)) for s in starts], dtype=np.int64)
        self.windows_per_video = len(starts)
        self.labels = np.array([rows[v]["label_idx"] for v, _ in self.samples], dtype=np.int64)
        if use_gray and store.gray is None:
            raise RuntimeError(f"The '{store.variant}' feature cache has no grayscale frames (needed by the FFT branch).")

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __iter__(self):
        order = self.rng.permutation(len(self.samples)) if self.shuffle else np.arange(len(self.samples))
        offsets = np.arange(self.seq_len)
        for b in range(0, len(order), self.batch_size):
            idx = np.sort(order[b:b + self.batch_size])  # sorted -> sequential disk reads
            vid, start = self.samples[idx, 0], self.samples[idx, 1]
            rows = np.array([self.rows[v]["row"] for v in vid])
            frames = start[:, None] + offsets[None, :]  # [B, T]
            if self.shuffle_frames:  # diagnostic: destroy chronological order
                frames = self.rng.permuted(frames, axis=1)
            batch = {
                "cnn": torch.from_numpy(self.store.cnn[rows[:, None], frames].astype(np.float32)),
                "vit": torch.from_numpy(self.store.vit[rows[:, None], frames].astype(np.float32)),
                "label": torch.from_numpy(self.labels[idx]),
                "video": torch.from_numpy(vid),
            }
            if self.use_gray:
                if self.store.gray_in_memory:
                    g = self.store.gray[rows[:, None], frames]  # [B, T, H, W] uint8
                else:
                    g = np.stack([self.store.gray[r][f] for r, f in zip(rows, frames)])
                batch["gray"] = torch.from_numpy(g)
            yield {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
