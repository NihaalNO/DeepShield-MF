"""Frozen backbones for analyses that need features of new pixels (robustness, occlusion, profiling).

Same models, preprocessing and precision as scripts/cache_features.py, so features computed here
match the cached ones.
"""
import numpy as np
import timm
import torch

BACKBONES = {"resnet50": 2048, "vit_base_patch16_224": 768}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load(device):
    return {name: timm.create_model(name, pretrained=True, num_classes=0).eval().to(device) for name in BACKBONES}


@torch.no_grad()
def embed(models, rgb, device, batch=128):
    """rgb uint8 [N, 224, 224, 3] -> (resnet50 [N, 2048], vit [N, 768]) as float16, like the cache."""
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    out = {name: [] for name in models}
    for b in range(0, len(rgb), batch):
        x = torch.from_numpy(rgb[b:b + batch]).to(device).permute(0, 3, 1, 2).float().div_(255)
        x = (x - mean) / std
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            for name, m in models.items():
                out[name].append(m(x).float().cpu().numpy().astype(np.float16))
    return np.concatenate(out["resnet50"]), np.concatenate(out["vit_base_patch16_224"])


def to_gray(rgb):
    """FFT-branch input, defined as in the cache: round(mean(R, G, B)) as uint8."""
    return np.rint(rgb.astype(np.float32).mean(axis=-1)).astype(np.uint8)
