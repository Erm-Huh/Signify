"""Reusable ASL alphabet (A-Z) recognizer built on the vendored upstream pipeline.

Per-frame chain, identical to upstream ``realtime_translator.main`` (commit 45ff2c0e):

    BGR frame --flip--> MediaPipe Hands (1 hand, complexity 1, 0.5/0.5)
      -> landmarks_to_pixels -> OneEuroFilter (timestamped)
      -> MotionTracker.update (raw pixels / hand size; for J/Z)
      -> normalize_landmarks -> render_skeleton (96x96 RGB) -> ONNX logits
      -> softmax -> mean over the last 8 frames -> argmax
      -> MotionTracker.classify (J/Z override) -> SpellingBuffer.update (commit rule)

About the scores: the graph outputs raw logits. Upstream applies a plain softmax and
calls the result "confidence"; it is NOT temperature-calibrated. Upstream measured
precision 96.7% / coverage 94.6% at its COMMIT_CONFIDENCE (0.75) on its held-out test
dataset (static photos). We report the same softmax scores, labelled as uncalibrated,
and never compare them numerically with the word model's calibrated probabilities.

Frame-rate note: upstream's temporal constants are counted in frames (8-frame average,
10-frame commit, 45-frame motion window, per-frame speed threshold). Upstream processes
frames sequentially inside its camera loop, so its effective rate is whatever its own
per-frame work allows; we run the same work on a worker thread and report the achieved rate.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from app.asl_alphabet import preprocessing as pp
from app.asl_alphabet.motion import MotionTracker
from app.asl_alphabet.smoothing import OneEuroFilter
from app.asl_alphabet.spelling import (
    COMMIT_CONFIDENCE,
    COMMIT_FRAMES,
    CONFIDENCE_FLOOR,
    LOGITS,
    SMOOTHING_WINDOW,
    SpellingBuffer,
)

# Upstream prepare_dataset.py defaults (how training landmarks were extracted from photos).
STILL_IMAGE_PAD = 0.25
STILL_IMAGE_MIN_CONFIDENCE = 0.3
# Upstream realtime_translator.py Hands settings.
LIVE_MIN_DETECTION = 0.5
LIVE_MIN_TRACKING = 0.5


class AlphabetModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class LetterGuess:
    label: str  # "A".."Z"
    confidence: float  # softmax score (uncalibrated), see module docstring


@dataclass(frozen=True)
class AlphabetResult:
    timestamp: float  # capture time of the frame this result describes
    computed_at: float  # when the result became available (same clock)
    hand_detected: bool
    top: list[LetterGuess] = field(default_factory=list)  # smoothed, sorted, <= 3
    best: str | None = None  # after the J/Z motion override
    confidence: float = 0.0  # smoothed score of the classifier's top letter
    static_best: str | None = None  # classifier argmax before the motion override
    motion_override: str | None = None  # set when motion.py changed the letter
    uncertain: bool = True  # confidence <= CONFIDENCE_FLOOR (upstream "uncertain")
    passes_threshold: bool = False  # confidence >= COMMIT_CONFIDENCE
    moving: bool = False  # upstream MotionTracker.is_moving()
    stable: bool = False  # COMMIT_FRAMES agreeing still frames above the threshold
    progress: float = 0.0  # upstream SpellingBuffer.progress()
    committed: str | None = None  # letter the upstream commit rule fired on this frame
    infer_ms: float = 0.0
    points_px: npt.NDArray[np.float32] | None = None  # smoothed, mirrored-frame pixels
    skeleton_rgb: npt.NDArray[np.uint8] | None = None  # what the model saw


class AlphabetModel:
    """Loads the ONNX classifier once; thread-safe ``scores`` (ORT sessions are)."""

    def __init__(self, model_dir: Path, intra_op_threads: int = 2) -> None:
        onnx_path = model_dir / "asl_cnn_model.onnx"
        names_path = model_dir / "class_names.txt"
        for p in (onnx_path, names_path):
            if not p.is_file():
                raise AlphabetModelError(
                    f"Missing {p}. Run `python scripts/download_model.py` to fetch it."
                )
        self.class_names = names_path.read_text().strip().split(",")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # Shares the CPU with two MediaPipe graphs in the live app: a few threads and no
        # spin-waiting (ORT's default spins idle threads, starving MediaPipe). Outputs are
        # unaffected; measured 7.6 ms per frame on an M2 with 2 threads.
        opts.intra_op_num_threads = intra_op_threads
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = inp.shape[1] if isinstance(inp.shape[1], int) else pp.MODEL_INPUT_SIZE
        outputs = [o.name for o in self.session.get_outputs()]
        self.output = LOGITS if LOGITS in outputs else outputs[0]
        n_out = self.session.get_outputs()[outputs.index(self.output)].shape[-1]
        if n_out != len(self.class_names):
            raise AlphabetModelError(f"model has {n_out} outputs, {len(self.class_names)} labels")
        self.device = "cpu"
        self.logits(np.zeros((self.size, self.size, 3), np.uint8))  # warm-up

    def logits(self, skeleton_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        out = self.session.run([self.output], {self.input_name: pp.model_input(skeleton_rgb)})
        return out[0][0]

    @staticmethod
    def softmax(logits: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        # Upstream: "The graph emits raw logits; softmax happens here."
        exponentials = np.exp(logits - logits.max())
        return exponentials / exponentials.sum()

    def scores(self, skeleton_rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
        return self.softmax(self.logits(skeleton_rgb))

    def top_k(self, scores: npt.NDArray[np.floating], k: int = 3) -> list[LetterGuess]:
        order = np.argsort(-scores, kind="stable")[:k]
        return [LetterGuess(self.class_names[i].upper(), float(scores[i])) for i in order]


class HandDetector:
    """MediaPipe Hands, configured as upstream does for live video or for still photos."""

    def __init__(self, *, still_image: bool = False) -> None:
        import mediapipe as mp

        self.still_image = still_image
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=still_image,
            max_num_hands=1,
            model_complexity=pp.HAND_MODEL_COMPLEXITY,
            min_detection_confidence=STILL_IMAGE_MIN_CONFIDENCE if still_image else LIVE_MIN_DETECTION,
            min_tracking_confidence=LIVE_MIN_TRACKING,
        )

    def detect(
        self, bgr: npt.NDArray[np.uint8], max_side: int | None = None
    ) -> tuple[npt.NDArray[np.float32] | None, int, int]:
        """Raw pixel landmarks (21, 2) of the first hand, or None. ``bgr`` is used as given:
        the caller mirrors live frames first, exactly as upstream does.

        ``max_side`` (optional, live app only) runs MediaPipe on a uniformly downscaled copy.
        MediaPipe returns image-normalised coordinates, and they are converted with the
        ORIGINAL width/height, so the pixel units upstream's OneEuroFilter and MotionTracker
        thresholds expect are unchanged. Default None = upstream behaviour (full frame)."""
        h, w = bgr.shape[:2]
        small = bgr
        if max_side is not None and max(h, w) > max_side:
            scale = max_side / max(h, w)
            small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = self._hands.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return None, w, h
        return pp.landmarks_to_pixels(result.multi_hand_landmarks[0], w, h), w, h

    def close(self) -> None:
        self._hands.close()


class AlphabetStream:
    """Per-stream temporal state (one per webcam / video). Mirrors the body of upstream's
    camera loop. ``update`` must be called from one thread; the text-editing helpers take
    a lock so the UI thread can call them."""

    def __init__(self, model: AlphabetModel, *, clock=time.perf_counter) -> None:
        self.model = model
        self.clock = clock
        self.smoother = OneEuroFilter()
        self.tracker = MotionTracker()
        self.spelling = SpellingBuffer()
        self.score_history: list[npt.NDArray[np.float64]] = []
        self._lock = threading.Lock()

    def update(self, points_px: npt.NDArray[np.float32] | None, timestamp: float) -> AlphabetResult:
        start = time.perf_counter()
        with self._lock:
            if points_px is None:
                # upstream: else-branch when no hand is found
                self.score_history.clear()
                self.smoother.reset()
                self.tracker.reset()
                self.spelling.update(None, 0.0, False, False)
                return AlphabetResult(timestamp, self.clock(), hand_detected=False)

            points_px = self.smoother(points_px, timestamp)
            self.tracker.update(points_px)
            normalized = pp.normalize_landmarks(points_px)
            skeleton_rgb = pp.render_skeleton(normalized, size=self.model.size)
            self.score_history.append(self.model.scores(skeleton_rgb))
            if len(self.score_history) > SMOOTHING_WINDOW:
                self.score_history.pop(0)
            scores = np.mean(self.score_history, axis=0)
            letter = self.model.class_names[int(np.argmax(scores))]
            confidence = float(scores.max())
            static_letter = letter
            resolved = self.tracker.classify(letter)
            if resolved and resolved in self.model.class_names:
                letter = resolved
            moving = self.tracker.is_moving()
            committed = self.spelling.update(letter, confidence, True, moving)
            progress = self.spelling.progress()

        passes = confidence >= COMMIT_CONFIDENCE
        return AlphabetResult(
            timestamp=timestamp,
            computed_at=self.clock(),
            hand_detected=True,
            top=self.model.top_k(scores),
            best=letter.upper(),
            confidence=confidence,
            static_best=static_letter.upper(),
            motion_override=letter.upper() if letter != static_letter else None,
            uncertain=confidence <= CONFIDENCE_FLOOR,
            passes_threshold=passes,
            moving=moving,
            stable=progress >= 1.0 and passes and not moving,
            progress=progress,
            committed=committed.upper() if committed else None,
            infer_ms=(time.perf_counter() - start) * 1000,
            points_px=points_px,
            skeleton_rgb=skeleton_rgb,
        )

    # Upstream key actions, so its no-double-letter rule stays consistent with the text.
    def reset_repeat_guard(self) -> None:
        """After a space/backspace/clear the same letter may be committed again (upstream)."""
        with self._lock:
            self.spelling._last_committed = None  # what upstream space()/backspace()/clear() do

    def reset(self) -> None:
        with self._lock:
            self.score_history.clear()
            self.smoother.reset()
            self.tracker.reset()
            self.spelling = SpellingBuffer()


# --------------------------------------------------------------------------- offline helpers


@dataclass(frozen=True)
class StillImageResult:
    hand_detected: bool
    top: list[LetterGuess]
    best: str | None
    confidence: float
    passes_threshold: bool
    uncertain: bool


def classify_still_image(model: AlphabetModel, detector: HandDetector, bgr: npt.NDArray[np.uint8]) -> StillImageResult:
    """One photo, extracted the way upstream extracted its training photos
    (prepare_dataset.py: static mode, 25% black border, no mirroring). Single frame: no
    smoothing, no stability, and no J/Z motion (J and Z are movements)."""
    border = int(round(max(bgr.shape[:2]) * STILL_IMAGE_PAD))
    padded = cv2.copyMakeBorder(bgr, border, border, border, border, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    points_px, _, _ = detector.detect(padded)
    if points_px is None:
        return StillImageResult(False, [], None, 0.0, False, True)
    scores = model.scores(pp.render_skeleton(pp.normalize_landmarks(points_px), size=model.size))
    top = model.top_k(scores)
    conf = top[0].confidence
    return StillImageResult(True, top, top[0].label, conf, conf >= COMMIT_CONFIDENCE, conf <= CONFIDENCE_FLOOR)


def iter_video_frames(path: Path, max_seconds: float) -> Iterator[tuple[float, npt.NDArray[np.uint8]]]:
    """(timestamp_s, BGR frame) at <= ~30 fps; upstream's frame-count constants assume a
    webcam-like rate, so high-fps sources are strided like the word pipeline does."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("unreadable_video")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        stride = max(1, round(fps / 30.0)) if fps > 45 else 1
        i = 0
        while cap.grab():
            i += 1
            if (i - 1) % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            t = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if t > max_seconds:
                raise ValueError("video_too_long")
            yield t, frame
    finally:
        cap.release()


@dataclass
class VideoAlphabetResult:
    final: AlphabetResult | None
    committed: list[str]
    frames: int
    frames_with_hand: int
    fps: float
    motion_overrides: list[str]


def classify_video(model: AlphabetModel, path: Path, max_seconds: float = 15.0) -> VideoAlphabetResult:
    """Run the live pipeline over a recorded clip (mirrored frames, temporal smoothing,
    J/Z motion and the commit rule), as if it were the webcam."""
    detector = HandDetector(still_image=False)
    stream = AlphabetStream(model)
    committed: list[str] = []
    overrides: list[str] = []
    last: AlphabetResult | None = None
    last_hand: AlphabetResult | None = None
    n = with_hand = 0
    times: list[float] = []
    try:
        for t, frame in iter_video_frames(path, max_seconds):
            n += 1
            times.append(t)
            points, _, _ = detector.detect(cv2.flip(frame, 1))
            last = stream.update(points, t)
            if last.hand_detected:
                with_hand += 1
                last_hand = last
            if last.committed:
                committed.append(last.committed)
            if last.motion_override and (not overrides or overrides[-1] != last.motion_override):
                overrides.append(last.motion_override)
    finally:
        detector.close()
    fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
    return VideoAlphabetResult(last_hand or last, committed, n, with_hand, fps, overrides)
