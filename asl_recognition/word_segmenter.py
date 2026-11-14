"""Python port of GestureSign web/src/ml/segmenter.ts.

Copyright (c) 2026 Shashank Yadav. MIT License; full notice reproduced in
WORD_RECOGNITION.md. Upstream defaults/state transitions and trimming are
preserved. Python additionally rejects non-increasing/nonfinite timestamps
and copies buffered frames so caller-owned arrays cannot change history.
"""

from dataclasses import dataclass

import numpy as np

from word_tracking import LEFT_HAND_START, RIGHT_HAND_START, POSE_START, hands_missing


@dataclass(frozen=True)
class SegmenterOptions:
    gap_ms: float = 300
    move_speed: float = 0.6
    min_frames: int = 8
    max_ms: float = 4000
    preroll_ms: float = 200
    leave_ms: float = 500


@dataclass(frozen=True)
class Segment:
    frames: np.ndarray
    timestamps: tuple[float, ...]
    fps: float
    duration_ms: float


@dataclass(frozen=True)
class SegmentEvent:
    type: str  # start, end, discard
    reason: str | None = None
    segment: Segment | None = None
    t: float | None = None


def hand_speed(previous, current, dt_ms):
    """Mean common hand-point travel in shoulder widths/sec, like handSpeed."""
    if dt_ms <= 0:
        return 0.0
    shoulders = current[[POSE_START + 11, POSE_START + 12], :2]
    scale = float(np.linalg.norm(shoulders[0] - shoulders[1]))
    if not np.isfinite(scale) or scale <= 1e-3:
        scale = 0.25  # upstream DEFAULT_ANCHOR.scale
    points = np.r_[LEFT_HAND_START:LEFT_HAND_START + 21,
                   RIGHT_HAND_START:RIGHT_HAND_START + 21]
    distances = np.linalg.norm(current[points, :2] - previous[points, :2], axis=1)
    distances = distances[np.isfinite(distances)]
    return float(distances.mean() / scale / (dt_ms / 1000)) if len(distances) else 0.0


class Segmenter:
    def __init__(self, options=None):
        self.options = options or SegmenterOptions()
        opts = self.options
        if (not all(np.isfinite(v) for v in (opts.gap_ms, opts.move_speed,
                                             opts.max_ms, opts.preroll_ms, opts.leave_ms))
                or min(opts.gap_ms, opts.move_speed, opts.max_ms) <= 0
                or min(opts.preroll_ms, opts.leave_ms) < 0
                or not isinstance(opts.min_frames, int) or opts.min_frames < 1):
            raise ValueError("Invalid segmenter options")
        self.reset()

    def reset(self):
        self.samples = []
        self.recent = []
        self.last = None
        self.speed = 0.0
        self.quiet_since = None
        self.quiet_reason = "absent"
        self.from_rest = False
        self.state = "idle"

    @property
    def frame_count(self):
        return len(self.samples)

    def push(self, frame, t, *, crowded=False):
        """Feed normalized landmarks and capture time in monotonic milliseconds."""
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (543, 3):
            raise ValueError("Expected landmarks with shape [543, 3]")
        if not np.isfinite(t) or (self.last is not None and t <= self.last[1]):
            raise ValueError("Timestamps must be finite and strictly increasing; reset after source changes")
        sample = (frame.copy(), float(t))
        present = not crowded and not hands_missing(frame)
        if present and self.last is not None and not hands_missing(self.last[0]):
            self.speed = 0.5 * self.speed + 0.5 * hand_speed(self.last[0], frame, t - self.last[1])
        elif not present:
            self.speed = 0.0
        self.last = sample
        moving = present and self.speed >= self.options.move_speed

        if self.state != "signing":
            if not present:
                self.state = "idle"
            self.recent = ([s for s in self.recent + [sample]
                            if t - s[1] <= self.options.preroll_ms] if present else [])
            if not moving:
                return None
            self.from_rest = self.state == "resting"
            self.state = "signing"
            self.samples = list(self.recent)
            self.quiet_since = None
            return SegmentEvent("start", t=t)

        self.samples.append(sample)
        if self.samples and t - self.samples[0][1] >= self.options.max_ms:
            event = self._finish("max", len(self.samples))
            self.recent = []
            return event
        if moving:
            self.quiet_since = None
            return None
        if self.quiet_since is None:
            self.quiet_since = t
            self.quiet_reason = "still" if present else "absent"
        if t - self.quiet_since < self.options.gap_ms:
            return None
        keep = sum(s[1] < self.quiet_since for s in self.samples)
        reason = self.quiet_reason
        if reason == "absent" and self.from_rest:
            kept = self.samples[:keep]
            span = (kept[-1][1] if kept else 0) - (kept[0][1] if kept else 0)
            if span < self.options.leave_ms:
                self._finish(reason, 0)
                return SegmentEvent("discard", "hands-leaving")
        event = self._finish(reason, keep + 1 if reason == "still" else keep)
        self.state = "resting" if present else "idle"
        return event

    def flush(self):
        """Finish an active sequence, trimming an unfinished quiet gap."""
        if self.state != "signing":
            return None
        keep = (len(self.samples) if self.quiet_since is None else
                sum(s[1] < self.quiet_since for s in self.samples))
        return self._finish("flush", keep)

    def _finish(self, reason, keep):
        kept = self.samples[:max(0, keep)]
        self.samples = []
        self.quiet_since = None
        self.state = "idle"
        if len(kept) < self.options.min_frames:
            return SegmentEvent("discard", "too-short")
        timestamps = tuple(s[1] for s in kept)
        duration = timestamps[-1] - timestamps[0]
        fps = (len(kept) - 1) * 1000 / duration if duration > 0 else 30
        segment = Segment(np.stack([s[0] for s in kept]), timestamps, fps, duration)
        return SegmentEvent("end", reason, segment)
