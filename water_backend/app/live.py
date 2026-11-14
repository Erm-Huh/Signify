"""Continuous (sliding-window) live recognition, all in RAM.

What it is: every ``interval_s`` the last ``window_s`` seconds of webcam landmarks are
classified as one isolated sign, and the results are smoothed. What it is NOT: sign
segmentation or sentence recognition. The model is GestureSign's isolated-sign
classifier, so a window that straddles two signs, or catches half a sign, is
classified anyway. Smoothing plus the calibrated "not sure" threshold hide most of
those, but not all.

Pipeline (three threads, bounded hand-offs, nothing written to disk)::

    camera thread (caller)   --frame, t-->  [queue, maxsize=2, drop-oldest]
    landmark worker          Holistic (same graph as uploads) -> ring buffer of
                             (t, [543, 3]) landmarks, bounded by buffer_s
    inference worker (x1)    every interval_s, if idle: snapshot the last window_s,
                             resample to 30 fps on the timestamps, then the SAME
                             ``landmarks_to_features`` + ``SignRecognizer.predict``
                             as POST /predict

Why landmarks are buffered instead of raw frames: Holistic is a tracking graph that
costs ~50 ms per frame on an M2 CPU, so re-running it over 60 frames every 0.5 s is
impossible. Each frame is processed once, in order. When Holistic is slower than the
camera, frames are dropped before it, and the window is resampled on the real capture
timestamps. ``resample_window`` uses upstream's ``interpolate`` (same NaN rule) and
reduces exactly to upstream ``resample`` for evenly spaced frames.

Model contract (checked in ``LiveConfig.validate``): 30 fps, 8 <= T <= 64 frames
(the ONNX graph fails for T > 64), so window_s <= 63 / 30 = 2.1 s.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from app.config import settings as app_settings
from app.gesturesign.resample import interpolate
from app.model import SignRecognizer
from app.preprocessing import VideoError, frame_to_landmarks, landmarks_to_features, make_holistic

log = logging.getLogger(__name__)

Landmarks = npt.NDArray[np.float32]
LandmarkFn = Callable[[np.ndarray], Landmarks]


@dataclass(frozen=True)
class LiveConfig:
    window_s: float = 2.0  # seconds of landmarks per inference (<= 2.1 s, see module doc)
    interval_s: float = 0.5  # how often a window is scheduled
    smoothing_windows: int = 3  # average calibrated probabilities over the last N windows
    min_stable_windows: int = 2  # consecutive confident windows with the same label to "emit"
    cooldown_s: float = 1.5  # no new emitted sign within this time of the previous one
    frame_queue_size: int = 2  # camera -> landmark worker; oldest frame dropped when full
    buffer_s: float = 4.0  # landmark ring buffer length (RAM: ~6.5 KB per frame)
    max_frame_side: int = 640  # downscale before Holistic (coordinates are unchanged)
    holistic_complexity: int | None = None  # None = same as uploads (settings)
    history_size: int = 8  # emitted signs kept for display

    def validate(self, rec: SignRecognizer) -> LiveConfig:
        max_window = (rec.max_frames - 1) / rec.fps
        min_window = (rec.min_frames - 1) / rec.fps
        if not min_window <= self.window_s <= max_window + 1e-9:
            raise ValueError(
                f"window_s must be in [{min_window:.2f}, {max_window:.2f}] s: the model takes "
                f"{rec.min_frames}-{rec.max_frames} frames at {rec.fps:.0f} fps"
            )
        if self.interval_s <= 0 or self.smoothing_windows < 1 or self.min_stable_windows < 1:
            raise ValueError("interval_s > 0, smoothing_windows >= 1, min_stable_windows >= 1")
        if self.cooldown_s < 0 or self.frame_queue_size < 1 or self.buffer_s < self.window_s:
            raise ValueError("cooldown_s >= 0, frame_queue_size >= 1, buffer_s >= window_s")
        return self


# --------------------------------------------------------------------------- sampling


def resample_window(
    times: npt.NDArray[np.float64],
    frames: Landmarks,
    t_end: float,
    window_s: float,
    fps: float,
) -> Landmarks:
    """Landmarks captured at ``times`` (seconds, increasing) -> evenly spaced ``fps`` frames
    covering ``[t_end - window_s, t_end]`` (clipped to the data), last frame at ``t_end``.

    Positions are fractional source indices, fed to upstream ``interpolate``, so missing
    (NaN) neighbours take the nearer frame exactly as in training.
    """
    if len(times) == 0:
        return np.zeros((0,) + frames.shape[1:], dtype=np.float32)
    t_end = min(t_end, float(times[-1]))
    t_start = max(t_end - window_s, float(times[0]))
    n = int(np.floor((t_end - t_start) * fps + 1e-6)) + 1
    targets = t_end - (np.arange(n - 1, -1, -1, dtype=np.float64) / fps)
    positions = np.interp(targets, times, np.arange(len(times), dtype=np.float64))
    return interpolate(frames, positions)


class LandmarkBuffer:
    """Thread-safe ring buffer of (timestamp, landmarks), bounded by duration and count."""

    def __init__(self, seconds: float, max_items: int = 600) -> None:
        self.seconds = seconds
        self._items: deque[tuple[float, Landmarks]] = deque(maxlen=max_items)
        self._lock = threading.Lock()

    def append(self, t: float, landmarks: Landmarks) -> None:
        with self._lock:
            if self._items and t <= self._items[-1][0]:
                return  # out of order / duplicate timestamp
            self._items.append((t, landmarks))
            while self._items and t - self._items[0][0] > self.seconds:
                self._items.popleft()

    def snapshot(self, window_s: float) -> tuple[npt.NDArray[np.float64], Landmarks]:
        with self._lock:
            items = list(self._items)
        if not items:
            return np.zeros(0), np.zeros((0, 543, 3), dtype=np.float32)
        t_last = items[-1][0]
        # one extra frame before the window so interpolation has a left neighbour
        keep = [i for i, (t, _) in enumerate(items) if t >= t_last - window_s]
        first = max(0, keep[0] - 1)
        items = items[first:]
        return np.array([t for t, _ in items]), np.stack([f for _, f in items])

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# --------------------------------------------------------------------------- decisions


@dataclass(frozen=True)
class WindowResult:
    """One classified window. ``probs`` is None when there was nothing to classify."""

    t_end: float
    probs: npt.NDArray[np.float64] | None
    partial_view: bool = False
    reason: str = ""  # "no_hands_detected" / "sign_too_short" / ""
    infer_ms: float = 0.0


@dataclass
class LiveDecision:
    state: str = "waiting"  # waiting | idle | sign | not_sure
    label: str | None = None  # smoothed top-1 when state == "sign"
    confidence: float = 0.0
    top: list[tuple[str, float]] = field(default_factory=list)
    reason: str = ""
    emitted: list[str] = field(default_factory=list)  # recent accepted signs, oldest first
    just_emitted: str | None = None
    # Added for the unified app's router (defaults keep existing behaviour):
    stable: bool = False  # confident with the same label for >= min_stable_windows
    t_end: float = 0.0  # capture time of the window's last landmark frame
    decided_at: float = 0.0  # when this decision was made (LiveRecognizer clock)


class SignSmoother:
    """Temporal smoothing + calibrated rejection + cooldown, independent of threads.

    * smoothed probs = mean of the calibrated probability vectors of the last N windows;
      a window without hands clears the history (the signer stopped).
    * confident  <=> smoothed top-1 >= the model's calibrated ``notSureBelow`` and the
      signer is not partly out of frame (the same rule as POST /predict).
    * A label is *emitted* to the history after ``min_stable_windows`` consecutive confident
      windows with that label, at least ``cooldown_s`` after the previous emission. The same
      label is not emitted twice in a row until it has been interrupted (a not-sure, idle,
      or different-label window), so holding one sign yields one entry.
    """

    def __init__(self, labels: list[str], not_sure_below: float, cfg: LiveConfig) -> None:
        self.labels = labels
        self.threshold = not_sure_below
        self.cfg = cfg
        self._probs: deque[npt.NDArray[np.float64]] = deque(maxlen=cfg.smoothing_windows)
        self._streak_label: str | None = None
        self._streak = 0
        self._last_emit_t = -np.inf
        self._last_emitted: str | None = None
        self._armed = True  # may the last emitted label be emitted again?
        self.history: deque[str] = deque(maxlen=cfg.history_size)

    def update(self, r: WindowResult) -> LiveDecision:
        if r.probs is None:
            self._probs.clear()
            self._streak_label, self._streak, self._armed = None, 0, True
            return LiveDecision(state="idle", reason=r.reason, emitted=list(self.history))
        self._probs.append(r.probs)
        mean = np.mean(np.stack(self._probs), axis=0)
        order = np.argsort(-mean)[:3]
        top = [(self.labels[i], float(mean[i])) for i in order]
        label, conf = top[0]
        confident = conf >= self.threshold and not r.partial_view
        if not confident:
            self._streak_label, self._streak, self._armed = None, 0, True
            reason = "partial_view" if r.partial_view else "low_confidence"
            return LiveDecision("not_sure", None, conf, top, reason, list(self.history))

        if label == self._streak_label:
            self._streak += 1
        else:
            self._streak_label, self._streak = label, 1
            if label != self._last_emitted:
                self._armed = True
        just = None
        if (
            self._streak >= self.cfg.min_stable_windows
            and r.t_end - self._last_emit_t >= self.cfg.cooldown_s
            and (label != self._last_emitted or self._armed)
        ):
            self.history.append(label)
            self._last_emit_t, self._last_emitted, self._armed = r.t_end, label, False
            just = label
        return LiveDecision(
            "sign", label, conf, top, "", list(self.history), just,
            stable=self._streak >= self.cfg.min_stable_windows,
        )


# --------------------------------------------------------------------------- scheduling


class InferenceScheduler:
    """Runs at most one job at a time on one worker thread, at most every ``interval_s``.
    A tick that arrives while the previous job is still running is skipped, not queued."""

    def __init__(self, interval_s: float, clock: Callable[[], float] = time.perf_counter) -> None:
        self.interval_s = interval_s
        self.clock = clock
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-infer")
        self._future: Future[None] | None = None
        self._last = -np.inf
        self.submitted = 0
        self.skipped_busy = 0
        self._closed = False

    @property
    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def maybe_submit(self, job: Callable[[], None], now: float | None = None) -> bool:
        if self._closed:
            return False
        now = self.clock() if now is None else now
        if now - self._last < self.interval_s:
            return False
        if self.busy:
            self.skipped_busy += 1
            self._last = now  # skip this slot instead of firing as soon as the job ends
            return False
        self._last = now
        self.submitted += 1
        self._future = self._pool.submit(job)
        return True

    def shutdown(self, wait: bool = True) -> None:
        self._closed = True
        self._pool.shutdown(wait=wait, cancel_futures=True)


# --------------------------------------------------------------------------- engine


@dataclass
class LiveStats:
    frames_in: int = 0
    frames_dropped: int = 0
    landmark_frames: int = 0
    landmark_ms: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    infer_ms: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    # capture time of a window's last frame -> decision available
    result_lag_ms: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    stale_windows: int = 0  # scheduled, but no new landmarks since the previous window


class LiveRecognizer:
    """Feed frames with ``submit_frame``; call ``tick`` often; read ``decision``."""

    def __init__(
        self,
        rec: SignRecognizer,
        cfg: LiveConfig | None = None,
        *,
        landmark_fn_factory: Callable[[], tuple[LandmarkFn, Callable[[], None]]] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.rec = rec
        self.cfg = (cfg or LiveConfig()).validate(rec)
        self.clock = clock
        self.stats = LiveStats()
        self.buffer = LandmarkBuffer(self.cfg.buffer_s)
        labels = [s["label"] for s in rec.signs]
        self.smoother = SignSmoother(labels, rec.not_sure_below, self.cfg)
        self.scheduler = InferenceScheduler(self.cfg.interval_s, clock)
        self._frames: queue.Queue[tuple[float, np.ndarray] | None] = queue.Queue(
            maxsize=self.cfg.frame_queue_size
        )
        self._decision = LiveDecision()
        self._last_t_end: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._factory = landmark_fn_factory or self._holistic_factory
        self._ready = threading.Event()
        self._worker_error: BaseException | None = None
        self._worker = threading.Thread(target=self._landmark_loop, name="live-landmarks", daemon=True)
        self._worker.start()
        self._closed = False

    # -- landmark worker ------------------------------------------------------------------

    def _holistic_factory(self) -> tuple[LandmarkFn, Callable[[], None]]:
        holistic = make_holistic(app_settings, self.cfg.holistic_complexity)
        side = self.cfg.max_frame_side
        return (lambda bgr: frame_to_landmarks(holistic, bgr, side)), holistic.close

    def _landmark_loop(self) -> None:
        try:
            fn, close = self._factory()
        except BaseException as exc:  # surfaced via .error
            self._worker_error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                try:
                    item = self._frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                t, frame = item
                start = time.perf_counter()
                landmarks = fn(frame)
                self.stats.landmark_ms.append((time.perf_counter() - start) * 1000)
                self.stats.landmark_frames += 1
                self.buffer.append(t, landmarks)
        except BaseException as exc:
            log.exception("landmark worker crashed")
            self._worker_error = exc
        finally:
            close()

    def wait_ready(self, timeout: float = 30.0) -> None:
        self._ready.wait(timeout)
        if self._worker_error:
            raise RuntimeError(f"landmark worker failed: {self._worker_error}")

    @property
    def error(self) -> BaseException | None:
        return self._worker_error

    def submit_frame(self, frame: np.ndarray, t: float | None = None) -> bool:
        """Non-blocking. Returns False if an older queued frame had to be dropped."""
        if self._closed:
            return False
        t = self.clock() if t is None else t
        self.stats.frames_in += 1
        dropped = False
        while True:
            try:
                self._frames.put_nowait((t, frame))
                return not dropped
            except queue.Full:
                try:
                    self._frames.get_nowait()
                    self.stats.frames_dropped += 1
                    dropped = True
                except queue.Empty:
                    pass

    # -- inference ------------------------------------------------------------------------

    def classify_window(self, times: npt.NDArray[np.float64], frames: Landmarks) -> WindowResult:
        """Pure function of a landmark window -> WindowResult (used by the job and tests)."""
        if len(times) == 0:
            return WindowResult(0.0, None, reason="no_frames")
        t_end = float(times[-1])
        uniform = resample_window(times, frames, t_end, self.cfg.window_s, self.rec.fps)
        start = time.perf_counter()
        try:
            prepared = landmarks_to_features(
                uniform,
                self.rec.fps,  # already at the model rate: upstream resample is a no-op
                feature_set=self.rec.feature_set,
                model_fps=self.rec.fps,
                min_frames=self.rec.min_frames,
                max_frames=self.rec.max_frames,
            )
        except VideoError as exc:
            return WindowResult(t_end, None, reason=exc.code)
        pred = self.rec.predict(prepared.features, k=len(self.rec.signs))
        probs = np.zeros(len(self.rec.signs))
        for g in pred.top:
            probs[g.index] = g.confidence
        ms = (time.perf_counter() - start) * 1000
        return WindowResult(t_end, probs, prepared.partial_view, "", ms)

    def _job(self) -> None:
        times, frames = self.buffer.snapshot(self.cfg.window_s)
        if len(times) and self._last_t_end is not None and times[-1] <= self._last_t_end:
            # MediaPipe hasn't produced a new frame (e.g. it stalled while re-detecting):
            # don't count the same evidence twice in smoothing / stability.
            self.stats.stale_windows += 1
            return
        if len(times):
            self._last_t_end = float(times[-1])
        result = self.classify_window(times, frames)
        decision = self.smoother.update(result)
        decision.t_end, decision.decided_at = result.t_end, self.clock()
        if result.probs is not None:
            self.stats.infer_ms.append(result.infer_ms)
            self.stats.result_lag_ms.append((self.clock() - result.t_end) * 1000)
        if decision.just_emitted:
            log.info("emitted %s (%.2f)", decision.just_emitted, decision.confidence)
        with self._lock:
            self._decision = decision

    def tick(self, now: float | None = None) -> bool:
        """Schedule a window if the interval elapsed and no job is running."""
        if self._closed or len(self.buffer) == 0:
            return False
        return self.scheduler.maybe_submit(self._job, now)

    @property
    def decision(self) -> LiveDecision:
        with self._lock:
            return self._decision

    # -- shutdown -------------------------------------------------------------------------

    def close(self, timeout: float = 10.0) -> None:
        """Stop both workers, drop buffered frames and landmarks. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self.scheduler.shutdown(wait=True)
        while True:  # drain, then wake the worker
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout)
        self.buffer.clear()

    @property
    def alive_threads(self) -> list[str]:
        names = []
        if self._worker.is_alive():
            names.append(self._worker.name)
        names += [t.name for t in threading.enumerate() if t.name.startswith("live-infer") and t.is_alive()]
        return names

    def __enter__(self) -> LiveRecognizer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def with_overrides(cfg: LiveConfig, **kw: object) -> LiveConfig:
    return replace(cfg, **{k: v for k, v in kw.items() if v is not None})
