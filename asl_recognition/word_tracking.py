"""Normalized Holistic tracking for the standalone word demo; no camera access.

Frame layout follows GestureSign's assemble.ts/landmarks.ts (MIT, Copyright
2026 Shashank Yadav). See WORD_WEBCAM.md and WORD_RECOGNITION.md for attribution.
"""

from dataclasses import dataclass

import numpy as np

FACE_START, LEFT_HAND_START, POSE_START, RIGHT_HAND_START = 0, 468, 489, 522
N_POINTS = 543


def format_holistic(results):
    """Holistic results -> float32 [543,3]; retain normalized xyz and NaNs.

    Input must come from an UNMIRRORED image. Holistic already assigns hands
    anatomically; do not apply the browser HandLandmarker handedness fallback.
    Extra iris points, if present, are excluded from the 468-point face slot.
    """
    frame = np.full((N_POINTS, 3), np.nan, dtype=np.float32)
    for name, start, count in (("face_landmarks", FACE_START, 468),
                               ("left_hand_landmarks", LEFT_HAND_START, 21),
                               ("pose_landmarks", POSE_START, 33),
                               ("right_hand_landmarks", RIGHT_HAND_START, 21)):
        part = getattr(results, name, None)
        if part is None:
            continue
        points = part.landmark
        # Browser assembly only writes a face/pose when its full list exists.
        if name in ("face_landmarks", "pose_landmarks") and len(points) < count:
            continue
        for i, point in enumerate(points[:count]):
            frame[start + i] = (point.x, point.y, point.z)
    frame[~np.isfinite(frame)] = np.nan
    return frame


def hands_missing(frame):
    """Match upstream handsMissing: presence is determined by the two wrists."""
    return not (np.isfinite(frame[LEFT_HAND_START, 0])
                or np.isfinite(frame[RIGHT_HAND_START, 0]))


def frame_status(frame, poses):
    """Match browser frameStatus, including the shoulder framing tolerance."""
    if poses > 1:
        return "crowded"
    if hands_missing(frame):
        return "nothing" if poses == 0 else "no-hands"
    shoulders = frame[[POSE_START + 11, POSE_START + 12], :2]
    if not (np.isfinite(shoulders).all()
            and ((shoulders >= -0.02) & (shoulders <= 1.02)).all()):
        return "partial"
    return "ok"


@dataclass(frozen=True)
class TrackedFrame:
    frame: np.ndarray
    poses: int

    @property
    def status(self):
        return frame_status(self.frame, self.poses)

    @property
    def hands(self):
        return sum(int(np.isfinite(self.frame[start, 0]))
                   for start in (LEFT_HAND_START, RIGHT_HAND_START))

    @property
    def face_present(self):
        return bool(np.isfinite(self.frame[:468, 0]).any())


class HolisticTracker:
    """Legacy Python Holistic, using packaged standard assets and one person.

    MediaPipe is imported only on construction so formatting/segmenter tests
    and Phase 1 inference never initialize camera or tracking dependencies.
    """

    def __init__(self):
        import cv2
        import mediapipe as mp
        self._cv2 = cv2
        self._factory = mp.solutions.holistic.Holistic
        self._tracker = None
        self.reset()

    def reset(self):
        self.close()
        self._tracker = self._factory(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            enable_segmentation=False, refine_face_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, bgr):
        if self._tracker is None:
            raise RuntimeError("Holistic tracker is closed")
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("Expected an unmirrored uint8 BGR camera frame")
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._tracker.process(rgb)
        return TrackedFrame(format_holistic(results),
                            int(results.pose_landmarks is not None))

    def close(self):
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
