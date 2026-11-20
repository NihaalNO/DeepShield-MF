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
import json
import sys
import uuid
from collections import OrderedDict
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
from deepshield import paths, segments as sg  # noqa: E402
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
        # optional calibration next to the checkpoint's run directory (scripts/calibrate_served.py)
        self.calibration = None
        calib_path = Path(checkpoint).resolve().parent.parent / "calibration.json"
        if calib_path.is_file():
            c = json.loads(calib_path.read_text())
            if c.get("format") == "deepshield-calibration/1":
                self.calibration = c
                self.threshold = float(c["threshold_raw"])  # the decision rule chosen on validation
        # how segment scores become one video score (scripts/eval_segments.py, chosen on validation)
        self.aggregation = {"rule": "middle", "threshold": self.threshold}
        agg_path = Path(checkpoint).resolve().parent.parent / "aggregation.json"
        if agg_path.is_file():
            a = json.loads(agg_path.read_text())
            if a.get("format") == "deepshield-aggregation/1":
                self.aggregation = {"rule": a["rule"], "threshold": float(a["threshold"])}
        self.checkpoint = str(checkpoint)
        self._recent = OrderedDict()  # explain id -> crops of the segment behind a verdict (last 16 analyses)
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

    def calibrated(self, score):
        """P(fake) for equally likely real and fake videos (Platt scaling fitted on validation), or None."""
        if not self.calibration:
            return None
        z = float(np.log(max(score, 1e-6) / max(1 - score, 1e-6)))
        return float(1 / (1 + np.exp(-(self.calibration["a"] * z + self.calibration["b"]))))

    def _result(self, probs, extra):
        score = float(np.mean(probs))
        return {"stage": "done", "model_kind": "finetuned", "score": score, "threshold": self.threshold,
                "p_calibrated": self.calibrated(score),
                "label": "fake" if score >= self.threshold else "real", "window_probs": probs,
                "windows_fake": int(sum(p >= self.threshold for p in probs)), **extra}

    # ---- video: every segment ---------------------------------------------------------
    def analyze_video(self, path):
        """Generator of progress events over every segment (deepshield/segments.py); the last event has
        stage 'done'. Raises NoFaceError if no segment has a face in at least half of its frames."""
        cap = cv2.VideoCapture(str(path))
        n_frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        if n_frames <= 0:
            raise RuntimeError("Unable to read the video's frame count")
        plan, middle, stride = sg.plan(n_frames)
        yield {"stage": "plan", "segments": len(plan), "video_seconds": n_frames / fps}
        segs, best = [None] * len(plan), {}
        for done, (k, frames) in enumerate(sg.iter_segment_frames(path, plan), 1):
            crops, rate = sg.segment_crops(frames, plan[k])
            seg = {"start_s": plan[k][0] / fps, "end_s": plan[k][-1] / fps, "face_rate": rate, "middle": k == middle,
                   "score": None, "window_probs": None}
            if crops is not None:
                probs = self._window_probs(list(crops))
                seg.update(score=float(np.mean(probs)), window_probs=probs, p_calibrated=self.calibrated(float(np.mean(probs))))
                best[k] = crops
            segs[k] = seg
            yield {"stage": "segment", "done": done, "of": len(plan), "score": seg["score"]}
        scored = [k for k, x in enumerate(segs) if x and x["score"] is not None]
        if not scored:
            raise NoFaceError(f"No part of the video shows a face in at least {sg.MIN_DET_RATE:.0%} of its frames "
                              "(the model judges faces only).")
        rule, thr = self.aggregation["rule"], self.aggregation["threshold"]
        near_mid = min(scored, key=lambda k: abs(k - middle))
        if rule == "mean":
            score, decider = float(np.mean([segs[k]["score"] for k in scored])), max(scored, key=lambda k: segs[k]["score"])
        elif rule == "max":
            decider = max(scored, key=lambda k: segs[k]["score"])
            score = segs[decider]["score"]
        else:
            decider = near_mid
            score = segs[decider]["score"]
        crops = best[decider]
        explain_id = self._remember(crops)
        d = segs[decider]
        yield {"stage": "done", "model_kind": "finetuned", "mode": "video", "evaluated": rule == "middle",
               "rule": rule, "score": score, "threshold": thr, "p_calibrated": self.calibrated(score) if rule == "middle" else None,
               "label": "fake" if score >= thr else "real", "segments": segs, "decider": decider, "middle": middle,
               "segments_scored": len(scored), "window_probs": d["window_probs"],
               "windows_fake": int(sum(p >= self.threshold for p in d["window_probs"])), "frames_analyzed": len(crops),
               "face_rate": d["face_rate"], "segment_seconds": [d["start_s"], d["end_s"]], "video_seconds": n_frames / fps,
               "stride": stride, "thumbs": [_thumb(crops[j]) for j in np.linspace(0, len(crops) - 1, 8).astype(int)],
               "explain_id": explain_id}

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
                                    "explain_id": self._remember(np.stack([crop] * T)),
                                    "thumbs": [_thumb(crop)]})

    # ---- explanations ----------------------------------------------------------------
    def _remember(self, crops):
        key = uuid.uuid4().hex
        self._recent[key] = np.asarray(crops)
        while len(self._recent) > 16:
            self._recent.popitem(last=False)
        return key

    def explain(self, explain_id):
        """Grad-CAM heatmaps (deepshield/explain.py) for the segment behind a recent verdict."""
        from deepshield import explain as ex
        if explain_id not in self._recent:
            raise KeyError("Unknown or expired analysis; run the analysis again.")
        crops = self._recent[explain_id]
        score = float(np.mean(self._window_probs(list(crops))))
        verdict_fake = score >= self.threshold
        _, _, comb = ex.gradcam_maps(self, crops, verdict_fake)
        unique = 1 if all(np.array_equal(crops[0], c) for c in crops[1:]) else len(crops)
        heat = comb.reshape(len(comb), -1)
        scale = max(float(np.percentile(comb[:unique], 99)), 1e-6)  # one display scale for the whole segment
        return {"verdict": "fake" if verdict_fake else "real", "segment_score": score,
                "frames": [{"overlay": ex.overlay(crops[i], comb[i] / scale), "image": _thumb(crops[i], 160),
                            "heat_mean": float(heat[i].mean()), "heat_max": float(heat[i].max())} for i in range(unique)]}
