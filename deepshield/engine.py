"""Training/evaluation loops and the checkpoint contract."""
import numpy as np
import torch
import torch.nn as nn

from . import metrics, paths
from .models import HybridHead

CHECKPOINT_FORMAT = "deepshield-hybrid-head/1"


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, batches, criterion, optimizer, grad_clip=1.0):
    model.train()
    total, n = 0.0, 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), batch["label"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(batch["label"])
        n += len(batch["label"])
    return total / max(n, 1)


@torch.no_grad()
def predict(model, batches, criterion=None, fake_index=0):
    """Window-level fake probabilities -> (p_fake_per_video, mean_loss)."""
    model.eval()
    probs, vids, total, n = [], [], 0.0, 0
    for batch in batches:
        logits = model(batch)
        if criterion is not None:
            total += criterion(logits, batch["label"]).item() * len(batch["label"])
            n += len(batch["label"])
        probs.append(torch.softmax(logits.float(), dim=1)[:, fake_index].cpu().numpy())
        vids.append(batch["video"].cpu().numpy())
    p_video = metrics.aggregate_videos(np.concatenate(probs), np.concatenate(vids), len(batches.rows))
    return p_video, (total / n if n else float("nan"))


def rows_to_arrays(rows, fake_index=0):
    y_fake = np.array([int(r["label_idx"] == fake_index) for r in rows])
    return y_fake, np.array([r["method"] for r in rows]), np.array([r["family_id"] for r in rows])


def save_checkpoint(path, model, model_config, data_config, class_names, threshold, extra=None):
    """Everything needed to rebuild the model and interpret its outputs."""
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,      # HybridHead kwargs
        "data_config": data_config,        # variant, seq_len, stride, backbones, normalisation
        "class_names": list(class_names),  # output index -> label
        "fake_index": list(class_names).index("fake"),
        "threshold": float(threshold),     # video is fake if mean window P(fake) >= threshold (chosen on validation)
        "extra": extra or {},
    }, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild the model; refuses checkpoints whose class contract differs from the project's."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: unknown checkpoint format {ckpt.get('format')!r}")
    if ckpt["class_names"] != paths.CLASS_NAMES:
        raise ValueError(f"{path}: class_names {ckpt['class_names']} != project contract {paths.CLASS_NAMES}")
    model = HybridHead(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt
