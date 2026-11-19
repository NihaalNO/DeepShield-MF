"""Serve the fine-tuned ResNet50 + ViT video model on new uploads (default: the second-domain model).

The pipeline is the one the model was trained and evaluated with, step for step:

    video -> 32 frames at a constant stride from the middle (scripts/extract_faces.py)
          -> YuNet face track, smoothed, square crop 1.3 x face size, 224 x 224
          -> JPEG quality 95 round trip (training read its crops from JPEG files)
          -> ResNet50 (layer4 fine-tuned) + ViT-B/16 (last two blocks fine-tuned)
          -> row-C head on 7 windows of 8 frames (50 % overlap)
          -> video score = mean window P(fake); fake if score >= the threshold chosen on validation

A still image has no frame sequence: its face crop is repeated to fill one 8-frame window.
The model was never evaluated on stills, so image results are flagged `evaluated: False`.

usage (library):
    det = FineTunedDetector(checkpoint_path, device)
    for event in det.analyze_video(path): ...   # progress events, then {"stage": "done", ...}
"""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import extract_faces as xf  # noqa: E402  (the training-time face pipeline)
from finetune_e2e import FRAMES, T, Net  # noqa: E402  (the trained model's definition)
from deepshield import paths  # noqa: E402
from deepshield.data import window_starts  # noqa: E402

# served model: trained on FF++ + Celeb-DF v2 train (docs/final_evaluation_protocol.md, "Second-domain model")
DEFAULT_CHECKPOINT = ROOT / "results" / "ffpp_c23" / "e2e_finetune_aug-frame_ffpp+celebdf" / "seed0" / "checkpoints" / "best_trainable.pth"
# extract_faces.py defaults used for every training, validation and test crop
MAX_STRIDE, MARGIN, MIN_DET_RATE, MIN_FRAMES, JPEG_QUALITY = 8, 1.3, 0.5, 8, 95
ARCH = "ResNet50 + ViT-B/16, fine-tuned, mean over 8-frame windows"


class NoFaceError(RuntimeError):
    pass


def _jpeg_round_trip(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _thumb(bgr, size=96):
    ok, buf = cv2.imencode(".jpg", cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA),
                           [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class FineTunedDetector:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if ckpt.get("format") != "deepshield-e2e-trainable/1":
            raise ValueError(f"{checkpoint}: not a fine-tuned DeepShield checkpoint ({ckpt.get('format')!r})")
        if ckpt["class_names"] != paths.CLASS_NAMES:
            raise ValueError(f"{checkpoint}: class_names {ckpt['class_names']} != {paths.CLASS_NAMES}")
        model = Net(ckpt["mode"])  # pretrained backbones from timm, then the fine-tuned layers on top
        missing = set(ckpt["trainable_state_dict"]) - set(model.state_dict())
        if missing:
            raise ValueError(f"{checkpoint}: unexpected keys {sorted(missing)[:5]}")
        model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        self.model = model.to(self.device).eval()
        self.threshold = float(ckpt["threshold"])
        self.fake_index = int(ckpt["fake_index"])
        self.checkpoint = str(checkpoint)
        if not paths.FACE_DETECTOR_MODEL.is_file():
            raise FileNotFoundError(f"Face detector model not found: {paths.FACE_DETECTOR_MODEL}")

    # ---- model -------------------------------------------------------------------------
    @torch.no_grad()
    def _window_probs(self, crops_bgr):
        """crops [N, 224, 224, 3] BGR uint8 -> P(fake) of each 8-frame window (50 % overlap)."""
        rgb = np.ascontiguousarray(np.stack(crops_bgr)[..., ::-1])
        x = torch.from_numpy(rgb).to(self.device).permute(0, 3, 1, 2).float().div(255)
        x = (x - self.model.mean) / self.model.std
        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            c, v = self.model.cnn(x).float(), self.model.vit(x).float()
        c, v = c.unsqueeze(0), v.unsqueeze(0)  # [1, N, D]
        probs = []
        for s in window_starts(len(crops_bgr), T):
            logits = self.model.head({"cnn": c[:, s:s + T], "vit": v[:, s:s + T]})
            probs.append(float(torch.softmax(logits.float(), dim=1)[0, self.fake_index]))
        return probs

    def _result(self, probs, extra):
        score = float(np.mean(probs))
        return {"stage": "done", "model_kind": "finetuned", "score": score, "threshold": self.threshold,
                "label": "fake" if score >= self.threshold else "real", "window_probs": probs,
                "windows_fake": int(sum(p >= self.threshold for p in probs)), **extra}

    # ---- video -------------------------------------------------------------------------
    def analyze_video(self, path):
        """Generator of progress events; the last one has stage 'done' (or raises NoFaceError)."""
        cap = cv2.VideoCapture(str(path))
        n_frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        if n_frames <= 0:
            raise RuntimeError("Unable to read the video's frame count")
        indices, stride = xf.choose_frame_indices(n_frames, FRAMES, MAX_STRIDE)
        yield {"stage": "decoding", "frames_total": n_frames, "frames_wanted": len(indices)}
        frames = xf.read_frames(path, indices)
        indices = [i for i in indices if i in frames]
        if len(indices) < MIN_FRAMES:
            raise RuntimeError(f"Only {len(indices)} frames could be decoded (need {MIN_FRAMES})")

        dets = []
        for k, i in enumerate(indices):
            dets.append(xf.detect(frames[i]))
            if k % 4 == 3 or k == len(indices) - 1:
                yield {"stage": "faces", "done": k + 1, "of": len(indices)}
        trk = xf.track(dets)
        det_rate = float((~np.isnan(trk[:, 0])).mean())
        if det_rate < MIN_DET_RATE:
            raise NoFaceError(f"A face was found in only {det_rate:.0%} of the sampled frames "
                              f"(the model needs at least {MIN_DET_RATE:.0%}).")
        smooth = xf.fill_and_smooth(trk)
        crops = [_jpeg_round_trip(xf.crop_square(frames[i], cx, cy, float(size) * MARGIN))
                 for i, (cx, cy, size) in zip(indices, smooth)]
        yield {"stage": "scoring", "frames": len(crops)}
        probs = self._window_probs(crops)
        yield self._result(probs, {
            "mode": "video", "evaluated": True, "frames_analyzed": len(crops), "face_rate": det_rate,
            "segment_seconds": [indices[0] / fps, indices[-1] / fps], "video_seconds": n_frames / fps,
            "stride": stride, "thumbs": [_thumb(crops[j]) for j in np.linspace(0, len(crops) - 1, 8).astype(int)]})

    # ---- image -------------------------------------------------------------------------
    def analyze_image(self, path):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError("Unable to read the image")
        dets = xf.detect(frame)
        if len(dets) == 0:
            raise NoFaceError("No face was found in the image.")
        k = int(np.argmax(dets[:, 2] * dets[:, 3]))
        x, y, w, h = dets[k, :4]
        crop = _jpeg_round_trip(xf.crop_square(frame, x + w / 2, y + h / 2, float(max(w, h)) * MARGIN))
        probs = self._window_probs([crop] * T)
        return self._result(probs, {"mode": "image", "evaluated": False, "frames_analyzed": 1, "face_rate": 1.0,
                                    "thumbs": [_thumb(crop)]})
