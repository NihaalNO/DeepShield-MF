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


class FFTBranch(nn.Module):
    """Same layers and spectrum computation as the notebook; input is the grayscale frame in [0, 1]."""

    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
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
        with torch.autocast(mag.device.type, enabled=mag.device.type == "cuda"):
            out = self.net(mag.reshape(b * t, 1, h, w))
        return out.float().reshape(b, t, -1)


class HybridHead(nn.Module):
    def __init__(self, use_cnn=True, use_vit=True, use_fft=True, temporal="bilstm", num_classes=2,
                 fft_dim=128, lstm_hidden=192, dropout=0.35):
        super().__init__()
        if not (use_cnn or use_vit or use_fft):
            raise ValueError("at least one branch is required")
        if temporal not in ("bilstm", "mean"):
            raise ValueError(temporal)
        self.use_cnn, self.use_vit, self.use_fft, self.temporal = use_cnn, use_vit, use_fft, temporal
        self.fft_branch = FFTBranch(fft_dim) if use_fft else None
        in_dim = CNN_DIM * use_cnn + VIT_DIM * use_vit + fft_dim * use_fft
        self.temporal_norm = nn.LayerNorm(in_dim)
        if temporal == "bilstm":
            self.lstm = nn.LSTM(in_dim, lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
            rep_dim = lstm_hidden * 2
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
        seq = self.temporal_norm(torch.cat(feats, dim=-1))  # [B, T, D]
        if self.lstm is not None:
            seq, _ = self.lstm(seq)
        return self.classifier(seq.mean(dim=1))  # temporal mean pooling


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
