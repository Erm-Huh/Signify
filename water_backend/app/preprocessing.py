"""Video -> model input, following GestureSign's training/serving pipeline.

1. Decode frames with OpenCV (rotation metadata is applied by OpenCV's FFmpeg backend).
2. Run MediaPipe Holistic (legacy graph, the one that produced the asl-signs training
   data) on every frame and store the result in the Kaggle 543-point layout
   ``[T, 543, 3]`` (face 468, left hand 21, pose 33, right hand 21), NaN when missing.
3. Trim to the span of frames where a hand is visible (the browser demo segments
   signs by hand presence in the same spirit).
4. ``resample`` to 30 fps -> ``fit_length`` to <= 64 frames -> ``sequence_features``
   (460 floats per frame). Steps 4 are the vendored upstream functions, called in the
   same order as upstream ``fixtures.raw_to_model_input``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from app.config import Settings
from app.gesturesign import landmarks as lm
from app.gesturesign.features import FeatureSet, sequence_features
from app.gesturesign.resample import fit_length, resample

log = logging.getLogger(__name__)

# The browser demo closes a segment after 4 s; longer spans are probably several signs.
LONG_SIGN_SECONDS = 4.0
# Same bounds as the browser's ``shoulderInView``.
_IN_VIEW_MIN, _IN_VIEW_MAX = -0.02, 1.02


class VideoError(Exception):
    """A problem with the uploaded video that the client should hear about."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class ExtractedClip:
    landmarks: npt.NDArray[np.float32]  # [T, 543, 3], NaN where missing
    fps: float
    width: int
    height: int
    duration_s: float
    fps_source: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreparedInput:
    features: npt.NDArray[np.float32]  # [T, F]
    partial_view: bool
    stats: dict[str, float | int | str | bool]
    warnings: list[str]


# --------------------------------------------------------------------------- upload checks


def check_extension(filename: str | None, settings: Settings) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise VideoError(
            "unsupported_file_type",
            f"Unsupported file type '{suffix or '(none)'}'. Allowed: {allowed}.",
            status_code=415,
        )
    return suffix


# --------------------------------------------------------------------------- decoding


def _estimate_fps(timestamps_ms: list[float], container_fps: float) -> tuple[float, str]:
    """Prefer the frame timestamps (browser WebM often reports a bogus container fps)."""
    if len(timestamps_ms) >= 2:
        span = timestamps_ms[-1] - timestamps_ms[0]
        increasing = all(b > a for a, b in zip(timestamps_ms, timestamps_ms[1:]))
        if span > 0 and increasing:
            fps = (len(timestamps_ms) - 1) * 1000.0 / span
            if 5.0 <= fps <= 120.0:
                return fps, "timestamps"
    if math.isfinite(container_fps) and 5.0 <= container_fps <= 120.0:
        return container_fps, "container"
    return 30.0, "default"


def _holistic_to_frame(result: object) -> npt.NDArray[np.float32]:
    out = np.full((lm.N_POINTS, 3), np.nan, dtype=np.float32)
    parts = (
        ("face_landmarks", lm.FACE_START, lm.N_FACE),
        ("left_hand_landmarks", lm.LEFT_HAND_START, lm.N_HAND),
        ("pose_landmarks", lm.POSE_START, lm.N_POSE),
        ("right_hand_landmarks", lm.RIGHT_HAND_START, lm.N_HAND),
    )
    for attr, start, size in parts:
        landmarks = getattr(result, attr, None)
        if landmarks is None:
            continue
        points = landmarks.landmark[:size]
        out[start : start + len(points)] = [(p.x, p.y, p.z) for p in points]
    return out


def extract_landmarks(path: Path, settings: Settings) -> ExtractedClip:
    """Decode ``path`` and run MediaPipe Holistic on each frame."""
    import mediapipe as mp  # imported lazily: slow import, not needed for /health tests

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("unreadable_video", "Could not open the video. Is it a valid video file?")
    try:
        container_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if container_fps > 0 and frame_count > 0:
            duration = frame_count / container_fps
            if duration > settings.max_video_seconds * 1.5:
                raise VideoError(
                    "video_too_long",
                    f"Video is about {duration:.1f} s; the limit is "
                    f"{settings.max_video_seconds:.0f} s. Upload a short clip of one sign.",
                    status_code=413,
                )
        # Skip frames from high-frame-rate sources (e.g. 60 fps phones): the model runs at
        # 30 fps anyway, and the real fps is re-estimated from the kept frames' timestamps.
        stride = max(1, round(container_fps / 30.0)) if container_fps > 45 else 1
        max_frames = int(settings.max_video_seconds * 120)

        frames: list[npt.NDArray[np.float32]] = []
        timestamps: list[float] = []
        width = height = 0
        raw_index = 0
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=settings.holistic_model_complexity,
            smooth_landmarks=True,
            refine_face_landmarks=False,
        )
        try:
            while True:
                if not cap.grab():
                    break
                raw_index += 1
                if raw_index > max_frames:
                    raise VideoError(
                        "video_too_long",
                        f"Video has more than {max_frames} frames; upload a short clip.",
                        status_code=413,
                    )
                if (raw_index - 1) % stride:
                    continue
                ok, bgr = cap.retrieve()
                if not ok or bgr is None:
                    continue
                timestamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
                height, width = bgr.shape[:2]
                scale = settings.max_frame_side / max(height, width)
                if scale < 1.0:
                    bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                frames.append(_holistic_to_frame(holistic.process(rgb)))
        finally:
            holistic.close()
    finally:
        cap.release()

    if not frames:
        raise VideoError(
            "empty_or_corrupt_video",
            "No frames could be decoded. The file is empty, corrupt, or uses an unsupported codec.",
        )
    fps, fps_source = _estimate_fps(timestamps, container_fps / stride if container_fps else 0.0)
    warnings = []
    if fps_source == "default":
        warnings.append("Could not determine the video frame rate; assumed 30 fps.")
    landmarks = np.stack(frames)
    duration = len(frames) / fps
    if duration > settings.max_video_seconds:
        raise VideoError(
            "video_too_long",
            f"Video is {duration:.1f} s; the limit is {settings.max_video_seconds:.0f} s.",
            status_code=413,
        )
    return ExtractedClip(landmarks, fps, width, height, duration, fps_source, warnings)


# --------------------------------------------------------------------------- features


def _hand_present(landmarks: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
    left = np.isfinite(landmarks[:, lm.LEFT_HAND_START, 0])
    right = np.isfinite(landmarks[:, lm.RIGHT_HAND_START, 0])
    return left | right


def _shoulders_in_view(landmarks: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
    ok = np.ones(landmarks.shape[0], dtype=bool)
    for idx in (lm.POSE_LEFT_SHOULDER, lm.POSE_RIGHT_SHOULDER):
        xy = landmarks[:, lm.pose(idx), :2]
        with np.errstate(invalid="ignore"):
            inside = (xy >= _IN_VIEW_MIN) & (xy <= _IN_VIEW_MAX)
        ok &= inside.all(axis=1)
    return ok


def landmarks_to_features(
    landmarks: npt.NDArray[np.float32],
    fps: float,
    *,
    feature_set: FeatureSet,
    model_fps: float,
    min_frames: int,
    max_frames: int,
) -> PreparedInput:
    """[T, 543, 3] landmarks at ``fps`` -> model features, exactly as upstream does it."""
    hands = _hand_present(landmarks)
    if not hands.any():
        raise VideoError(
            "no_hands_detected",
            "No hands were detected in the video. Make sure the signer's hands and upper "
            "body are visible and well lit.",
        )
    idx = np.flatnonzero(hands)
    segment = landmarks[idx[0] : idx[-1] + 1]
    seg_seconds = segment.shape[0] / fps

    resampled = resample(segment, fps, model_fps)
    if resampled.shape[0] < min_frames:
        raise VideoError(
            "sign_too_short",
            f"Hands were visible for only {seg_seconds:.2f} s; at least "
            f"{min_frames / model_fps:.2f} s of signing is needed.",
        )
    fitted = fit_length(resampled, max_frames)
    features = sequence_features(fitted, feature_set)

    # Browser rule: "partial" when most hand frames lack a shoulder in view.
    hand_frames = segment[_hand_present(segment)]
    partial = int((~_shoulders_in_view(hand_frames)).sum()) > hand_frames.shape[0] / 2

    warnings = []
    if partial:
        warnings.append(
            "Signer is only partly in view (shoulders not visible in most frames); "
            "the model is unreliable without the upper body."
        )
    if seg_seconds > LONG_SIGN_SECONDS:
        warnings.append(
            f"Signing spans {seg_seconds:.1f} s. The model recognises ONE isolated sign; "
            "longer clips are compressed into 64 frames and accuracy drops."
        )
    hand_ratio = float(hands[idx[0] : idx[-1] + 1].mean())
    stats: dict[str, float | int | str | bool] = {
        "frames_processed": int(landmarks.shape[0]),
        "frames_with_hands": int(hands.sum()),
        "sign_segment_seconds": round(seg_seconds, 3),
        "hand_coverage_in_segment": round(hand_ratio, 3),
        "model_frames": int(features.shape[0]),
    }
    return PreparedInput(features, partial, stats, warnings)


def preprocess_video(
    path: Path,
    settings: Settings,
    *,
    feature_set: FeatureSet,
    model_fps: float,
    min_frames: int,
    max_frames: int,
) -> tuple[PreparedInput, float]:
    """Full video pipeline. Returns the model input and the elapsed preprocessing ms."""
    start = time.perf_counter()
    clip = extract_landmarks(path, settings)
    prepared = landmarks_to_features(
        clip.landmarks,
        clip.fps,
        feature_set=feature_set,
        model_fps=model_fps,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    prepared.warnings = clip.warnings + prepared.warnings
    prepared.stats.update(
        {
            "source_fps": round(clip.fps, 2),
            "fps_source": clip.fps_source,
            "video_seconds": round(clip.duration_s, 3),
            "width": clip.width,
            "height": clip.height,
        }
    )
    elapsed = (time.perf_counter() - start) * 1000
    log.info(
        "preprocessed %s: %d frames @ %.1f fps (%s), %d with hands, %d model frames, %.0f ms",
        path.name,
        clip.landmarks.shape[0],
        clip.fps,
        clip.fps_source,
        prepared.stats["frames_with_hands"],
        prepared.stats["model_frames"],
        elapsed,
    )
    return prepared, elapsed
