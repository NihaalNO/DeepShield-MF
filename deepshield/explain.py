"""Per-frame heatmaps for the served fine-tuned model (Grad-CAM on both branches), with a faithfulness test.

Why gradients and not per-frame occlusion: a segment score averages 32 frames over 7 windows, so
hiding one 32 x 32 cell of one frame moves the evidence by about 0.01 log-odds (measured on FF++
test videos). Such maps are mostly noise. Grad-CAM gives each frame its own map from one backward
pass:

    target  = segment log-odds toward the verdict: mean over windows of (logit_fake - logit_real),
              negated when the verdict is "real"
    ResNet  = ReLU( sum_k  mean_xy(d target / d A_k) * A_k )     A: layer4 output, 7 x 7
    ViT     = ReLU( sum_d  (d target / d Z_d) * Z_d )             Z: last block's patch tokens, 14 x 14
    map     = mean of the two, each scaled by its maximum over the segment's frames

So a hot region pushed the model toward its verdict (fake or real), on a scale shared by all
frames of the segment: a cool frame contributed little. Gradient maps are an approximation;
`deletion_test` checks them: hiding each frame's K hottest cells (mean colour, 7 x 7 grid) should
weaken the verdict more than hiding K random cells.
"""
import base64

import cv2
import numpy as np
import torch

from deepshield.data import window_starts

GRID, PATCH, T = 7, 32, 8


def _prep(det, crops_bgr):
    rgb = np.ascontiguousarray(np.asarray(crops_bgr)[..., ::-1])
    x = torch.from_numpy(rgb).to(det.device).permute(0, 3, 1, 2).float().div(255)
    return (x - det.model.mean) / det.model.std


def _margin(det, cnn_seq, vit_seq):
    """Window sequences [B, T, D] -> logit_fake - logit_real [B]."""
    lg = det.model.head({"cnn": cnn_seq, "vit": vit_seq}).float()
    return lg[:, det.fake_index] - lg[:, 1 - det.fake_index]


def segment_logodds(det, crops_bgr):
    """Mean window log-odds (logit_fake - logit_real) of a segment."""
    with torch.no_grad(), torch.autocast(det.device.type, enabled=det.device.type == "cuda"):
        x = _prep(det, crops_bgr)
        c, v = det.model.cnn(x).float(), det.model.vit(x).float()
        return float(torch.stack([_margin(det, c[None, s:s + T], v[None, s:s + T])[0]
                                  for s in window_starts(len(crops_bgr), T)]).mean())


def gradcam_maps(det, crops_bgr, verdict_fake, chunk=8):
    """-> (cnn maps [N, 7, 7], vit maps [N, 14, 14], combined [N, 7, 7]); non-negative, shared scale."""
    crops = np.asarray(crops_bgr)
    n = len(crops)
    starts = window_starts(n, T)
    with torch.no_grad(), torch.autocast(det.device.type, enabled=det.device.type == "cuda"):
        x_all = _prep(det, crops)
        c_all, v_all = det.model.cnn(x_all).float(), det.model.vit(x_all).float()
    cnn_maps, vit_maps = np.zeros((n, 7, 7), np.float32), np.zeros((n, 14, 14), np.float32)
    sign = 1.0 if verdict_fake else -1.0
    store = {}
    h1 = det.model.cnn.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = det.model.vit.blocks[-1].register_forward_hook(lambda m, i, o: store.__setitem__("z", o))
    try:
        for f0 in range(0, n, chunk):
            idx = list(range(f0, min(n, f0 + chunk)))
            with torch.enable_grad(), torch.autocast(det.device.type, enabled=det.device.type == "cuda"):
                x = x_all[idx].detach().requires_grad_(True)
                c_chunk, v_chunk = det.model.cnn(x).float(), det.model.vit(x).float()
                a, z = store["a"], store["z"]
                a.retain_grad()
                z.retain_grad()
                c_seq, v_seq = c_all.clone(), v_all.clone()
                c_seq[idx], v_seq[idx] = c_chunk, v_chunk
                target = sign * torch.stack([_margin(det, c_seq[None, s:s + T], v_seq[None, s:s + T])[0] for s in starts]).mean()
            target.backward()
            ga, av = a.grad.float(), a.detach().float()                 # [B, 2048, 7, 7]
            cam = torch.relu((ga.mean(dim=(2, 3), keepdim=True) * av).sum(1))
            gz, zv = z.grad.float()[:, 1:], z.detach().float()[:, 1:]   # patch tokens, drop CLS
            tok = torch.relu((gz * zv).sum(-1)).reshape(len(idx), 14, 14)
            cnn_maps[idx], vit_maps[idx] = cam.cpu().numpy(), tok.cpu().numpy()
            det.model.zero_grad(set_to_none=True)
    finally:
        h1.remove()
        h2.remove()
    cn = cnn_maps / max(float(cnn_maps.max()), 1e-12)
    vn = vit_maps / max(float(vit_maps.max()), 1e-12)
    vn7 = vn.reshape(n, 7, 2, 7, 2).mean(axis=(2, 4))
    return cnn_maps, vit_maps, (cn + vn7) / 2


def _hide(frame, cells):
    out = frame.copy()
    fill = frame.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    for r, c in cells:
        out[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = fill
    return out


def deletion_test(det, crops_bgr, maps, verdict_fake, k=5, seed=0):
    """Log-odds toward the verdict: original, with each frame's k hottest cells hidden, with k random cells hidden."""
    rng = np.random.default_rng(seed)
    crops = np.asarray(crops_bgr)
    sign = 1.0 if verdict_fake else -1.0
    top = [[divmod(int(i), GRID) for i in np.argsort(-m.ravel())[:k]] for m in maps]
    rnd = [[divmod(int(i), GRID) for i in rng.choice(GRID * GRID, k, replace=False)] for _ in maps]
    base = sign * segment_logodds(det, crops)
    t = sign * segment_logodds(det, np.stack([_hide(c, cells) for c, cells in zip(crops, top)]))
    r = sign * segment_logodds(det, np.stack([_hide(c, cells) for c, cells in zip(crops, rnd)]))
    return base, t, r


def overlay(crop_bgr, heat, size=160, alpha=0.75):
    """Heat (0..1, shared scale) as a warm overlay; JPEG data URL."""
    # clip after resizing: cubic interpolation overshoots past 1, which would wrap to black in uint8
    h = np.clip(cv2.resize(np.clip(heat, 0, 1).astype(np.float32), (224, 224), interpolation=cv2.INTER_CUBIC), 0, 1)
    colour = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_INFERNO).astype(np.float32)
    w = (alpha * np.clip(h, 0, 1))[..., None]
    img = crop_bgr.astype(np.float32) * (1 - w) + colour * w
    img = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
