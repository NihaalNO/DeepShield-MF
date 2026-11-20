"""Whole-video reading: cut a video into the 32-frame segments the model was evaluated on.

The model was trained and tested on one segment per video: 32 frames at a constant stride (at most
8) from the middle (scripts/extract_faces.py). To read a whole video, the same segment is repeated
back to back before and after the middle one, so the middle segment is exactly the evaluated one:

    starts = middle_start + k * 32 * stride   for every k that keeps the segment inside the video,
    plus one segment flush with the first frame and one with the last, so the whole video is covered

Each segment gets its own face track and crops, exactly as in training (YuNet, smoothing, 1.3x
square crop, 224 x 224, JPEG quality 95 round trip). A segment where a face is found in fewer
than half of its frames is marked `no_face` and not scored.

Only CPU work lives here (decoding, detection, cropping), so evaluation scripts can run it in
worker processes; deepshield/serving.py scores the crops on the GPU.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import extract_faces as xf  # noqa: E402  (the training-time face pipeline)

FRAMES, MAX_STRIDE, MARGIN, MIN_DET_RATE, MIN_FRAMES, JPEG_QUALITY = 32, 8, 1.3, 0.5, 8, 95
MAX_SEGMENTS = 40  # a longer video is read at MAX_SEGMENTS evenly spaced segments (always with the middle one)


def plan(n_frames, max_segments=MAX_SEGMENTS):
    """-> (segments: list of frame-index lists, index of the middle (evaluated) segment, stride)."""
    mid, stride = xf.choose_frame_indices(n_frames, FRAMES, MAX_STRIDE)
    if len(mid) < FRAMES:  # a clip shorter than one segment: one (short) segment, as before
        return [mid], 0, stride
    step = FRAMES * stride
    span = (FRAMES - 1) * stride
    starts = [mid[0] - k * step for k in range(1, n_frames) if mid[0] - k * step >= 0][::-1]
    middle = len(starts)
    starts += [mid[0] + k * step for k in range(0, n_frames) if mid[0] + k * step + span <= n_frames - 1]
    # cover the ends too: one more segment flush with the first / last frame (it overlaps its neighbour)
    # (only when at least half a segment's worth of frames would otherwise go unread)
    if starts[0] >= step // 2:
        starts.insert(0, 0)
        middle += 1
    if n_frames - 1 - (starts[-1] + span) >= step // 2:
        starts.append(n_frames - 1 - span)
    if len(starts) > max_segments:
        keep = sorted(set(np.linspace(0, len(starts) - 1, max_segments - 1).round().astype(int).tolist()) | {middle})
        middle = keep.index(middle)
        starts = [starts[i] for i in keep]
    return [[s + i * stride for i in range(FRAMES)] for s in starts], middle, stride


def iter_segment_frames(path, segments):
    """Decode the video once, front to back; yield (segment index, {frame index: BGR frame}) as each
    segment completes, so only one or two segments of full frames are held in memory."""
    owner = {}
    for k, seg in enumerate(segments):
        for i in seg:
            owner.setdefault(i, []).append(k)
    last_of = {k: seg[-1] for k, seg in enumerate(segments)}
    pending = {k: {} for k in range(len(segments))}
    cap = cv2.VideoCapture(str(path))
    pos, last = 0, max(owner) if owner else -1
    try:
        while pos <= last:
            if not cap.grab():
                break
            if pos in owner:
                ok, frame = cap.retrieve()
                if ok:
                    for k in owner[pos]:
                        pending[k][pos] = frame
            for k in [k for k, li in last_of.items() if li == pos]:
                yield k, pending.pop(k)
                del last_of[k]
            pos += 1
    finally:
        cap.release()
    for k in list(last_of):  # the video ended early: yield what was decoded
        yield k, pending.pop(k)


def jpeg_round_trip(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def segment_crops(frames, indices):
    """-> (crops [N, 224, 224, 3] BGR uint8 or None, face detection rate)."""
    indices = [i for i in indices if i in frames]
    if len(indices) < MIN_FRAMES:
        return None, 0.0
    trk = xf.track([xf.detect(frames[i]) for i in indices])
    det_rate = float((~np.isnan(trk[:, 0])).mean())
    if det_rate < MIN_DET_RATE:
        return None, det_rate
    smooth = xf.fill_and_smooth(trk)
    crops = np.stack([jpeg_round_trip(xf.crop_square(frames[i], cx, cy, float(size) * MARGIN))
                      for i, (cx, cy, size) in zip(indices, smooth)])
    return crops, det_rate


def video_crops(path, n_frames=None):
    """All segments of a video with their crops (CPU only; used by evaluation workers).
    -> dict(segments=[{start, end, crops, face_rate}], middle, stride, n_frames, fps)"""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = n_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    segments, middle, stride = plan(n)
    out = [None] * len(segments)
    for k, frames in iter_segment_frames(path, segments):
        crops, rate = segment_crops(frames, segments[k])
        out[k] = {"start": segments[k][0], "end": segments[k][-1], "crops": crops, "face_rate": rate}
    return {"segments": out, "middle": middle, "stride": stride, "n_frames": n, "fps": fps}
