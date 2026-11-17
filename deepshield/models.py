"""Trainable part of the hybrid model, operating on cached backbone features.

`HybridHead` with cnn+vit+fft and temporal="bilstm" is the TemporalHybridModel
of notebooks/frac_df_cnnvit_fft_temporal.ipynb with its frozen ResNet50/ViT
replaced by their cached outputs: same FFT branch, LayerNorm, BiLSTM(192),
mean pooling and classifier. The switches exist only for the ablation matrix.
"""
import torch
import torch.nn as nn

CNN_DIM, VIT_DIM = 2048, 768

# Ablation matrix (Chapter 16.2). G and H reuse these with other evaluation/sequence settings.
EXPERIMENTS = {
    "A": dict(use_cnn=True, use_vit=False, use_fft=False, temporal="mean"),    # ResNet50, mean features
    "B": dict(use_cnn=False, use_vit=True, use_fft=False, temporal="mean"),    # ViT, mean features
    "C": dict(use_cnn=True, use_vit=True, use_fft=False, temporal="mean"),     # ResNet50+ViT, mean features
    "D": dict(use_cnn=True, use_vit=False, use_fft=False, temporal="bilstm"),  # ResNet50 + BiLSTM
    "E": dict(use_cnn=True, use_vit=True, use_fft=False, temporal="bilstm"),   # ResNet50+ViT + BiLSTM
    "F": dict(use_cnn=True, use_vit=True, use_fft=True, temporal="bilstm"),    # full hybrid (N5)
}

# Optional improvements (Chapter 16.4). Each changes one thing relative to its reference row.
EXPERIMENTS.update({
    # fusion, reference C (ResNet50+ViT, mean features)
    "I": dict(EXPERIMENTS["C"], fusion="proj"),    # independent branch projections, then concatenation
    "J": dict(EXPERIMENTS["C"], fusion="gate"),    # projections + learned softmax gate over branches
    "J0": dict(EXPERIMENTS["C"], fusion="avg"),    # gate ablation: projections averaged with fixed weights 1/K
    # causal recurrence, reference E (BiLSTM)
    "K": dict(EXPERIMENTS["E"], temporal="lstm"),  # one-directional LSTM, last hidden state
    # lightweight spatial encoder, references A (ResNet50) and C
    "L": dict(EXPERIMENTS["A"], cnn_backbone="efficientnet_b0", cnn_dim=1280),
    # frequency-band variants, references F (full spectrum) and E (no FFT branch)
    "M1": dict(EXPERIMENTS["F"], fft_mode="radial"),    # azimuthally averaged spectrum (112 radii) + MLP
    "M2": dict(EXPERIMENTS["F"], fft_mode="highpass"),  # spectrum with radii below 28 px removed
    "M3": dict(EXPERIMENTS["F"], fft_mode="bands"),     # low / mid / high radial bands as 3 input channels
})
FFT_MODES = ("full", "radial", "highpass", "bands")
BAND_EDGES = (28, 70)  # radii in px of the 224x224 centred spectrum (Nyquist = 112)


class FFTBranch(nn.Module):
    """Same layers and spectrum computation as the notebook; input is the grayscale frame in [0, 1]."""

    def __init__(self, out_dim=128, mode="full", size=224):
        super().__init__()
        if mode not in FFT_MODES:
            raise ValueError(mode)
        self.mode = mode
        if mode != "full":  # radius of every spectrum pixel from the centre (DC after fftshift)
            yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
            radius = torch.hypot((yy - size // 2).float(), (xx - size // 2).float())
            lo, hi = BAND_EDGES
            if mode == "highpass":
                self.register_buffer("mask", (radius >= lo).float(), persistent=False)
            elif mode == "bands":
                self.register_buffer("mask", torch.stack([radius < lo, (radius >= lo) & (radius < hi),
                                                          radius >= hi]).float(), persistent=False)
            else:  # radial: [size*size, n_bins] averaging matrix over integer radii below Nyquist
                n_bins = size // 2
                idx = radius.round().long().clamp(max=n_bins).flatten()
                onehot = torch.zeros(size * size, n_bins + 1).scatter_(1, idx[:, None], 1.0)[:, :n_bins]
                self.register_buffer("radial", onehot / onehot.sum(dim=0, keepdim=True), persistent=False)
                self.net = nn.Sequential(nn.Linear(n_bins, 128), nn.ReLU(inplace=True),
                                         nn.Linear(128, out_dim), nn.ReLU(inplace=True))
                return
        self.net = nn.Sequential(
            nn.Conv2d(3 if mode == "bands" else 1, 16, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(64, out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, gray):  # [B, T, H, W], uint8 or float in [0, 1]
        b, t, h, w = gray.shape
        gray = gray.float() / 255.0 if gray.dtype == torch.uint8 else gray.float()
        fft = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"), dim=(-2, -1))
        mag = torch.log1p(torch.abs(fft))
        mag = (mag - mag.mean(dim=(-2, -1), keepdim=True)) / (mag.std(dim=(-2, -1), keepdim=True) + 1e-6)
        # The spectrum is computed in float32; only the small CNN runs in mixed precision,
        # which halves its activation memory (B*T full-resolution maps) on a 6 GB GPU.
        if self.mode == "radial":
            return self.net(mag.reshape(b * t, h * w) @ self.radial).reshape(b, t, -1)
        mag = mag.reshape(b * t, 1, h, w)
        if self.mode != "full":
            mag = mag * self.mask  # highpass: [H, W]; bands: [3, H, W] -> three channels
        with torch.autocast(mag.device.type, enabled=mag.device.type == "cuda"):
            out = self.net(mag)
        return out.float().reshape(b, t, -1)


class HybridHead(nn.Module):
    """fusion:   "concat"  branches concatenated as they are (original model)
                 "proj"    each branch -> Linear(proj_dim) + LayerNorm + GELU, then concatenated
                 "gate"    projected branches combined per frame as sum_k g_k * p_k with
                           g = softmax(Linear([p_1 .. p_K])) over the K branches (g_k >= 0, sum_k g_k = 1)
                 "avg"     projected branches with fixed g_k = 1/K (ablation of the gate)
       temporal: "mean" | "bilstm" (mean over time of both directions) | "lstm" (causal, last state)"""

    def __init__(self, use_cnn=True, use_vit=True, use_fft=True, temporal="bilstm", num_classes=2,
                 fft_dim=128, lstm_hidden=192, dropout=0.35, fusion="concat", proj_dim=256,
                 fft_mode="full", cnn_dim=CNN_DIM, cnn_backbone="resnet50"):
        super().__init__()
        if not (use_cnn or use_vit or use_fft):
            raise ValueError("at least one branch is required")
        if temporal not in ("bilstm", "lstm", "mean"):
            raise ValueError(temporal)
        if fusion not in ("concat", "proj", "gate", "avg"):
            raise ValueError(fusion)
        self.use_cnn, self.use_vit, self.use_fft, self.temporal = use_cnn, use_vit, use_fft, temporal
        self.fusion, self.cnn_backbone = fusion, cnn_backbone  # cnn_backbone: which cached features feed "cnn"
        self.fft_branch = FFTBranch(fft_dim, fft_mode) if use_fft else None
        dims = [cnn_dim] * use_cnn + [VIT_DIM] * use_vit + [fft_dim] * use_fft
        self.last_gate = None  # mean gate weights of the last forward pass, for inspection
        if fusion == "concat":
            in_dim = sum(dims)
        else:
            self.proj = nn.ModuleList(nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
                                      for d in dims)
            in_dim = proj_dim * len(dims) if fusion == "proj" else proj_dim
            if fusion == "gate":
                self.gate = nn.Linear(proj_dim * len(dims), len(dims))
        self.temporal_norm = nn.LayerNorm(in_dim)
        if temporal in ("bilstm", "lstm"):
            bi = temporal == "bilstm"
            self.lstm = nn.LSTM(in_dim, lstm_hidden, num_layers=1, batch_first=True, bidirectional=bi)
            rep_dim = lstm_hidden * (2 if bi else 1)
        else:
            self.lstm, rep_dim = None, in_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(rep_dim), nn.Linear(rep_dim, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, batch):
        feats = []
        if self.use_cnn:
            feats.append(batch["cnn"])
        if self.use_vit:
            feats.append(batch["vit"])
        if self.use_fft:
            feats.append(self.fft_branch(batch["gray"]))
        if self.fusion != "concat":
            feats = [p(f) for p, f in zip(self.proj, feats)]
        if self.fusion in ("gate", "avg"):
            stack = torch.stack(feats, dim=-2)  # [B, T, K, proj_dim]
            if self.fusion == "gate":
                g = torch.softmax(self.gate(torch.cat(feats, dim=-1)), dim=-1)  # [B, T, K]
                self.last_gate = g.detach().mean(dim=(0, 1))
                fused = (g.unsqueeze(-1) * stack).sum(dim=-2)
            else:
                fused = stack.mean(dim=-2)
        else:
            fused = torch.cat(feats, dim=-1)
        seq = self.temporal_norm(fused)  # [B, T, D]
        if self.lstm is not None:
            seq, _ = self.lstm(seq)
        if self.temporal == "lstm":
            return self.classifier(seq[:, -1])  # causal: the state after the last frame
        return self.classifier(seq.mean(dim=1))  # temporal mean pooling


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
