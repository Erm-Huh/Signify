"""One webcam stream -> both recognisers -> one routed decision. No GUI code here.

Threads (all bounded; the caller owns the camera and the preview):

    caller (camera + preview) --frame--> word:     LiveRecognizer (Holistic landmark worker,
                                                   2 s sliding windows, 1 inference worker)
                              --frame--> alphabet: AlphabetWorker (Hands + CNN per frame,
                                                   queue of 1, drop-oldest)
                              SPACE ----> manual capture: the next ``manual_window_s`` of the
                                          SAME word landmark buffer, classified once
                                          (no temp file, no second Holistic pass)

``snapshot()`` returns the router decision plus new commit events; the caller (UI thread)
decides what goes into the text buffer. Measured on an M2: Holistic and Hands running in
parallel threads did not slow each other down, so both run in every mode by default; set
``run_word_in_spell`` / ``run_alphabet_in_word`` to False to save CPU.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.live import LiveConfig, LiveDecision, LiveRecognizer
from app.model import SignRecognizer
from app.models.alphabet_recognizer import AlphabetModel, AlphabetResult, AlphabetStream, HandDetector
from app.services.recognition_router import Kind, Mode, RouterConfig, RouterDecision, decide

log = logging.getLogger(__name__)

DetectorFactory = Callable[[], HandDetector]


@dataclass(frozen=True)
class SessionConfig:
    live: LiveConfig = LiveConfig()
    router: RouterConfig = RouterConfig()
    alphabet_queue_size: int = 1
    # MediaPipe Hands input size in the live app (None = full frame, as upstream). Pixel
    # units are preserved (see HandDetector.detect); this only reduces CPU.
    alphabet_max_side: int | None = 640
    run_word_in_spell: bool = True
    run_alphabet_in_word: bool = True
    manual_window_s: float = 2.0  # SPACE capture length (<= the word model's 2.1 s limit)
    manual_hold_s: float = 5.0  # how long a manual result stays the primary output


# --------------------------------------------------------------------------- alphabet worker


@dataclass
class AlphabetStats:
    frames_in: int = 0
    dropped: int = 0
    processed: int = 0
    with_hand: int = 0
    frame_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))  # detect+classify
    lag_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))  # capture -> result
    done_at: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def rate(self) -> float:
        d = list(self.done_at)
        return (len(d) - 1) / (d[-1] - d[0]) if len(d) > 1 and d[-1] > d[0] else 0.0


class AlphabetWorker:
    def __init__(self, model: AlphabetModel, *, queue_size: int = 1,
                 detector_factory: DetectorFactory | None = None, max_side: int | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.stream = AlphabetStream(model, clock=clock)
        self.max_side = max_side
        self.clock = clock
        self.stats = AlphabetStats()
        self._frames: queue.Queue[tuple[float, np.ndarray] | None] = queue.Queue(maxsize=queue_size)
        self._latest: AlphabetResult | None = None
        self._commits: deque[tuple[str, float]] = deque(maxlen=32)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._factory = detector_factory or (lambda: HandDetector(still_image=False))
        self._ready = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, name="alphabet-worker", daemon=True)
        self._thread.start()
        self._closed = False

    def wait_ready(self, timeout: float = 30.0) -> None:
        self._ready.wait(timeout)
        if self.error:
            raise RuntimeError(f"alphabet worker failed: {self.error}")

    def submit(self, frame: np.ndarray, t: float) -> None:
        if self._closed:
            return
        self.stats.frames_in += 1
        while True:
            try:
                self._frames.put_nowait((t, frame))
                return
            except queue.Full:
                try:
                    self._frames.get_nowait()
                    self.stats.dropped += 1
                except queue.Empty:
                    pass

    def _loop(self) -> None:
        try:
            detector = self._factory()
        except BaseException as exc:
            self.error = exc
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
                points, _, _ = detector.detect(cv2.flip(frame, 1), self.max_side)  # upstream mirrors first
                result = self.stream.update(points, t)
                self.stats.frame_ms.append((time.perf_counter() - start) * 1000)
                self.stats.processed += 1
                self.stats.with_hand += result.hand_detected
                now = self.clock()
                self.stats.lag_ms.append((now - t) * 1000)
                self.stats.done_at.append(now)
                with self._lock:
                    self._latest = result
                    if result.committed:
                        self._commits.append((result.committed, t))
        except BaseException as exc:
            log.exception("alphabet worker crashed")
            self.error = exc
        finally:
            detector.close()

    @property
    def latest(self) -> AlphabetResult | None:
        with self._lock:
            return self._latest

    def pop_commits(self) -> list[tuple[str, float]]:
        with self._lock:
            out = list(self._commits)
            self._commits.clear()
        return out

    def close(self, timeout: float = 10.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()


# --------------------------------------------------------------------------- manual capture


@dataclass(frozen=True)
class ManualResult:
    label: str | None
    confidence: float
    confident: bool  # calibrated threshold and in frame, as POST /predict
    top: list[tuple[str, float]]
    reason: str
    started: float
    done_at: float


@dataclass
class ManualCapture:
    state: str = "idle"  # idle | recording | processing
    started: float = 0.0
    future: Future[ManualResult] | None = None
    result: ManualResult | None = None


# --------------------------------------------------------------------------- session


@dataclass(frozen=True)
class Snapshot:
    decision: RouterDecision
    word: LiveDecision
    letter: AlphabetResult | None
    manual: ManualResult | None  # shown as primary while fresh
    capture_state: str
    capture_progress: float
    word_emitted: str | None  # new word-model emission since the last snapshot
    letter_commits: list[str]  # new upstream letter commits since the last snapshot


class UnifiedSession:
    def __init__(
        self,
        word_rec: SignRecognizer,
        alpha_model: AlphabetModel,
        cfg: SessionConfig = SessionConfig(),
        *,
        word_landmark_fn_factory=None,
        hand_detector_factory: DetectorFactory | None = None,
        clock: Callable[[], float] = time.perf_counter,
        mode: Mode = Mode.AUTO,
    ) -> None:
        if cfg.manual_window_s > cfg.live.window_s + 1e-9:
            raise ValueError("manual_window_s must be <= the live window (model frame limit)")
        self.cfg = cfg
        self.clock = clock
        self.mode = mode
        self.word = LiveRecognizer(word_rec, cfg.live, landmark_fn_factory=word_landmark_fn_factory, clock=clock)
        self.alphabet = AlphabetWorker(alpha_model, queue_size=cfg.alphabet_queue_size,
                                       detector_factory=hand_detector_factory,
                                       max_side=cfg.alphabet_max_side, clock=clock)
        self.capture = ManualCapture()
        self._manual_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="manual-capture")
        self._last_word_decision: LiveDecision | None = None
        self._closed = False

    def wait_ready(self, timeout: float = 30.0) -> None:
        self.word.wait_ready(timeout)
        self.alphabet.wait_ready(timeout)

    # -- frames ---------------------------------------------------------------------------

    def word_enabled(self) -> bool:
        return self.mode is not Mode.SPELL or self.cfg.run_word_in_spell or self.capture.state == "recording"

    def alphabet_enabled(self) -> bool:
        return self.mode is not Mode.WORD or self.cfg.run_alphabet_in_word

    def submit_frame(self, frame: np.ndarray, t: float | None = None) -> None:
        if self._closed:
            return
        t = self.clock() if t is None else t
        if self.word_enabled():
            self.word.submit_frame(frame, t)
        if self.alphabet_enabled():
            self.alphabet.submit(frame, t)

    def tick(self, now: float | None = None) -> None:
        if self._closed:
            return
        now = self.clock() if now is None else now
        if self.word_enabled():
            self.word.tick(now)
        cap = self.capture
        if cap.state == "recording" and now - cap.started >= self.cfg.manual_window_s:
            cap.state = "processing"
            cap.future = self._manual_pool.submit(self._classify_manual, cap.started, now)
        if cap.state == "processing" and cap.future is not None and cap.future.done():
            try:
                cap.result = cap.future.result()
            except Exception as exc:  # keep the app alive
                log.exception("manual capture failed")
                cap.result = ManualResult(None, 0.0, False, [], f"error: {exc}", cap.started, now)
            cap.state, cap.future = "idle", None

    # -- manual capture (SPACE) -----------------------------------------------------------

    def start_capture(self, now: float | None = None) -> bool:
        if self._closed or self.capture.state != "idle":
            return False
        self.capture = ManualCapture(state="recording", started=self.clock() if now is None else now)
        return True

    def _classify_manual(self, start: float, end: float) -> ManualResult:
        times, frames = self.word.buffer.snapshot(self.cfg.live.buffer_s)
        keep = (times >= start) & (times <= end)
        rec = self.word.rec
        if not keep.any():
            return ManualResult(None, 0.0, False, [], "no_frames", start, self.clock())
        r = self.word.classify_window(times[keep], frames[keep])
        if r.probs is None:
            return ManualResult(None, 0.0, False, [], r.reason, start, self.clock())
        order = np.argsort(-r.probs)[:3]
        top = [(rec.signs[i]["label"], float(r.probs[i])) for i in order]
        conf = top[0][1]
        confident = conf >= rec.not_sure_below and not r.partial_view
        reason = "partial_view" if r.partial_view else ("" if confident else "low_confidence")
        return ManualResult(top[0][0], conf, confident, top, reason, start, self.clock())

    def manual_result(self, now: float) -> ManualResult | None:
        r = self.capture.result
        if r is None or now - r.done_at > self.cfg.manual_hold_s:
            return None
        return r

    def dismiss_manual(self) -> None:
        self.capture.result = None

    # -- decisions ------------------------------------------------------------------------

    def snapshot(self, now: float | None = None) -> Snapshot:
        now = self.clock() if now is None else now
        word = self.word.decision
        letter = self.alphabet.latest
        decision = decide(self.mode, word, letter, now, self.cfg.router)
        emitted = None
        if word is not self._last_word_decision:  # a new window was classified
            emitted = word.just_emitted
            self._last_word_decision = word
        commits = [c for c, _ in self.alphabet.pop_commits()]
        manual = self.manual_result(now)
        if manual is not None:
            decision = manual_override(decision, manual)
        cap = self.capture
        progress = min(1.0, (now - cap.started) / self.cfg.manual_window_s) if cap.state == "recording" else 0.0
        return Snapshot(decision, word, letter, manual, cap.state, progress, emitted, commits)

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode

    # -- shutdown -------------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._manual_pool.shutdown(wait=True, cancel_futures=True)
        self.alphabet.close()
        self.word.close()

    @property
    def alive_threads(self) -> list[str]:
        names = list(self.word.alive_threads)
        if self.alphabet.alive:
            names.append("alphabet-worker")
        names += [t.name for t in threading.enumerate() if t.name.startswith("manual-capture") and t.is_alive()]
        return names

    def __enter__(self) -> UnifiedSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def manual_override(decision: RouterDecision, manual: ManualResult) -> RouterDecision:
    """A SPACE capture is an explicit request for a WORD; while fresh it is the primary
    output (in any mode), shown as WORD only if it passed the calibrated threshold."""
    from dataclasses import replace

    if manual.confident and manual.label:
        return replace(decision, kind=Kind.WORD, label=manual.label, stable=True,
                       reason="manual 2 s capture (word model above its calibrated threshold)",
                       alternatives=[])
    kind = Kind.NOT_SURE if manual.reason not in ("no_hands_detected", "no_frames") else Kind.NO_HANDS
    return replace(decision, kind=kind, label=None, stable=False,
                   reason=f"manual capture: {manual.reason or 'low_confidence'}", alternatives=[])
