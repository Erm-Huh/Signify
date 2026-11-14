"""Tests for the continuous live mode (app/live.py). No camera or MediaPipe needed:
threading tests inject a fake landmark function; model tests use the real checkpoint."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.gesturesign import landmarks as lm
from app.gesturesign.features import sequence_features
from app.gesturesign.resample import fit_length, resample
from app.live import (
    InferenceScheduler,
    LandmarkBuffer,
    LiveConfig,
    LiveRecognizer,
    SignSmoother,
    WindowResult,
    resample_window,
)
from app.model import SignRecognizer
from app.preprocessing import landmarks_to_features

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def rec() -> SignRecognizer:
    return SignRecognizer(settings.model_dir)


def heldout(sign: str) -> np.ndarray:
    replay = json.loads((DATA / "heldout_clips.json").read_text())
    clip = next(c for c in replay["clips"] if c["sign"] == sign)
    T = len(clip["frames"])
    out = np.full((T, lm.N_POINTS, 3), np.nan, dtype=np.float32)
    arr = np.array([[np.nan if v is None else v for v in f] for f in clip["frames"]], dtype=np.float32)
    out[:, replay["points"], :2] = arr.reshape(T, len(replay["points"]), 2)
    return out


def no_hands_like(frames: np.ndarray, n: int) -> np.ndarray:
    """n frames of the same body with both hands missing (signer at rest)."""
    rest = np.repeat(frames[:1], n, axis=0).copy()
    rest[:, lm.LEFT_HAND_START : lm.LEFT_HAND_START + lm.N_HAND] = np.nan
    rest[:, lm.RIGHT_HAND_START : lm.RIGHT_HAND_START + lm.N_HAND] = np.nan
    return rest


# --------------------------------------------------------------------------- sampling


def test_window_respects_model_frame_contract(rec: SignRecognizer) -> None:
    LiveConfig(window_s=2.1).validate(rec)  # 64 frames at 30 fps: the maximum
    with pytest.raises(ValueError):
        LiveConfig(window_s=2.2).validate(rec)  # 67 frames: ONNX graph fails above 64
    with pytest.raises(ValueError):
        LiveConfig(window_s=0.1).validate(rec)  # < 8 frames
    times = np.arange(200) / 30.0
    frames = np.random.default_rng(0).random((200, lm.N_POINTS, 3)).astype(np.float32)
    for w in (0.5, 1.0, 2.0, 2.1):
        out = resample_window(times, frames, times[-1], w, rec.fps)
        assert rec.min_frames <= out.shape[0] <= rec.max_frames
        assert out.shape[0] == int(np.floor(w * 30 + 1e-6)) + 1


def test_uniform_window_equals_upstream_resample() -> None:
    """At an even source rate the timestamp sampler reduces to upstream ``resample``."""
    rng = np.random.default_rng(1)
    for src_fps in (30.0, 24.0, 15.0):
        T = int(src_fps * 2) + 1  # exactly 2 s
        frames = rng.random((T, lm.N_POINTS, 3)).astype(np.float32)
        frames[5, 10:40] = np.nan  # missing points use the same nearest-frame rule
        times = np.arange(T) / src_fps
        ours = resample_window(times, frames, times[-1], 2.0, 30.0)
        ref = resample(frames, src_fps, 30.0)
        np.testing.assert_array_equal(np.isnan(ours), np.isnan(ref))
        np.testing.assert_allclose(ours, ref, atol=1e-5, equal_nan=True)


def test_irregular_timestamps_interpolate_in_time() -> None:
    times = np.array([0.0, 0.03, 0.2, 0.25, 0.6, 1.0])  # dropped frames -> uneven gaps
    frames = np.zeros((6, lm.N_POINTS, 3), dtype=np.float32)
    frames[:, 0, 0] = times * 10  # x = 10 t: linear in time
    out = resample_window(times, frames, 1.0, 1.0, 30.0)
    np.testing.assert_allclose(out[:, 0, 0], 10 * (1.0 - np.arange(30, -1, -1) / 30), atol=1e-5)


def test_buffer_is_bounded_by_time_and_snapshot_covers_window() -> None:
    buf = LandmarkBuffer(seconds=1.0)
    for i in range(300):
        buf.append(i / 30, np.zeros((lm.N_POINTS, 3), np.float32))
    assert len(buf) <= 31
    buf.append(0.0, np.zeros((lm.N_POINTS, 3), np.float32))  # out of order: ignored
    times, frames = buf.snapshot(0.5)
    assert times[-1] == pytest.approx(299 / 30)
    assert times[0] <= times[-1] - 0.5 and len(times) == frames.shape[0]


def test_live_window_matches_upload_pipeline_on_heldout_clip(rec: SignRecognizer) -> None:
    """A real held-out 'hello' clip streamed in at an uneven ~20 fps with rest frames around
    it: the live window gives the same answer as the upload path on the clean clip."""
    clip = heldout("hello")  # 30 fps
    rest = no_hands_like(clip, 15)
    stream = np.concatenate([rest, clip, rest])
    times = np.arange(len(stream)) / 30.0
    keep = np.arange(len(stream)) % 3 != 2  # drop every 3rd frame (Holistic too slow)
    live = LiveRecognizer(rec, LiveConfig(window_s=2.0), landmark_fn_factory=lambda: (lambda f: f, lambda: None))
    try:
        result = live.classify_window(times[keep], stream[keep])
    finally:
        live.close()
    upload = landmarks_to_features(
        clip, 30.0, feature_set=rec.feature_set, model_fps=30.0,
        min_frames=rec.min_frames, max_frames=rec.max_frames,
    )
    ref = rec.predict(upload.features, k=1).top[0]
    assert result.probs is not None
    assert rec.signs[int(np.argmax(result.probs))]["label"] == ref.label == "hello"
    assert result.probs.max() >= rec.not_sure_below


def test_window_without_hands_is_idle(rec: SignRecognizer) -> None:
    rest = no_hands_like(heldout("hello"), 60)
    live = LiveRecognizer(rec, landmark_fn_factory=lambda: (lambda f: f, lambda: None))
    try:
        r = live.classify_window(np.arange(60) / 30.0, rest)
    finally:
        live.close()
    assert r.probs is None and r.reason == "no_hands_detected"


# --------------------------------------------------------------------------- scheduling


def test_scheduler_interval_and_skips_overlapping_jobs() -> None:
    sched = InferenceScheduler(interval_s=0.5)
    gate, started, runs = threading.Event(), threading.Event(), []

    def job() -> None:
        started.set()
        gate.wait(5)
        runs.append(1)

    try:
        assert sched.maybe_submit(job, now=0.0)
        started.wait(2)
        assert not sched.maybe_submit(job, now=0.2)  # interval not elapsed: not counted
        assert sched.skipped_busy == 0
        assert not sched.maybe_submit(job, now=0.5)  # due, but job still running: skipped
        assert not sched.maybe_submit(job, now=1.0)
        assert sched.skipped_busy == 2 and sched.submitted == 1
        gate.set()
        for _ in range(100):
            if not sched.busy:
                break
            time.sleep(0.01)
        assert not sched.maybe_submit(job, now=1.2)  # skipped slot reset the interval
        assert sched.maybe_submit(job, now=1.5)
        for _ in range(100):
            if not sched.busy:
                break
            time.sleep(0.01)
    finally:
        gate.set()
        sched.shutdown()
    assert len(runs) == 2


# --------------------------------------------------------------------------- rejection / smoothing


LABELS = ["hello", "thank you", "yes", "no"]


def probs(**p: float) -> np.ndarray:
    v = np.full(len(LABELS), 1e-3)
    for k, x in p.items():
        v[LABELS.index(k.replace("_", " "))] = x
    return v / v.sum()


def test_rejection_uses_calibrated_threshold_and_partial_view(rec: SignRecognizer) -> None:
    sm = SignSmoother(LABELS, rec.not_sure_below, LiveConfig(smoothing_windows=1))
    low = sm.update(WindowResult(0.0, probs(hello=0.2, yes=0.19, no=0.18, thank_you=0.17)))
    assert low.state == "not_sure" and low.label is None and low.reason == "low_confidence"
    assert low.confidence < rec.not_sure_below
    framed_out = sm.update(WindowResult(0.5, probs(hello=0.95), partial_view=True))
    assert framed_out.state == "not_sure" and framed_out.reason == "partial_view"
    ok = sm.update(WindowResult(1.0, probs(hello=0.95)))
    assert ok.state == "sign" and ok.label == "hello"
    idle = sm.update(WindowResult(1.5, None, reason="no_hands_detected"))
    assert idle.state == "idle"


def test_smoothing_averages_recent_windows() -> None:
    sm = SignSmoother(LABELS, 0.29, LiveConfig(smoothing_windows=3, min_stable_windows=1))
    sm.update(WindowResult(0.0, probs(hello=0.9)))
    sm.update(WindowResult(0.5, probs(hello=0.9)))
    d = sm.update(WindowResult(1.0, probs(yes=0.9)))  # one outlier window
    assert d.label == "hello"  # (0.9 + 0.9 + ~0) / 3 beats (~0 + ~0 + 0.9) / 3
    sm.update(WindowResult(1.5, probs(yes=0.9)))
    assert sm.update(WindowResult(2.0, probs(yes=0.9))).label == "yes"
    d = sm.update(WindowResult(2.5, None, reason="no_hands_detected"))  # hands down: reset
    d = sm.update(WindowResult(3.0, probs(no=0.9)))
    assert d.label == "no" and d.confidence > 0.8


def test_cooldown_and_no_duplicate_emissions() -> None:
    cfg = LiveConfig(smoothing_windows=1, min_stable_windows=2, cooldown_s=1.5)
    sm = SignSmoother(LABELS, 0.29, cfg)
    emitted = []
    # hello held for 3 s: needs 2 stable windows, then emitted once only
    for i in range(7):
        d = sm.update(WindowResult(i * 0.5, probs(hello=0.9)))
        emitted += [d.just_emitted] if d.just_emitted else []
    assert emitted == ["hello"]

    # cooldown: a new sign right after an emission is displayed but not emitted yet
    sm = SignSmoother(LABELS, 0.29, cfg)
    sm.update(WindowResult(0.0, probs(hello=0.9)))
    assert sm.update(WindowResult(0.5, probs(hello=0.9))).just_emitted == "hello"
    sm.update(WindowResult(1.0, probs(yes=0.9)))
    d = sm.update(WindowResult(1.5, probs(yes=0.9)))  # stable, but only 1.0 s since hello
    assert d.label == "yes" and d.state == "sign" and d.just_emitted is None
    d = sm.update(WindowResult(2.0, probs(yes=0.9)))  # 1.5 s since hello: allowed
    assert d.just_emitted == "yes"
    # hello again after an interruption (hands down) is a new emission
    sm.update(WindowResult(2.5, None, reason="no_hands_detected"))
    sm.update(WindowResult(4.0, probs(hello=0.9)))
    d = sm.update(WindowResult(4.5, probs(hello=0.9)))
    assert d.just_emitted == "hello" and d.emitted == ["hello", "yes", "hello"]
    # a not-sure flicker shorter than the cooldown does not re-emit the same sign
    sm.update(WindowResult(5.0, probs(hello=0.2, yes=0.2, no=0.2, thank_you=0.2)))
    sm.update(WindowResult(5.5, probs(hello=0.9)))
    d = sm.update(WindowResult(5.8, probs(hello=0.9)))
    assert d.just_emitted is None


# --------------------------------------------------------------------------- engine + shutdown


def test_frame_queue_is_bounded_and_drops_oldest(rec: SignRecognizer) -> None:
    gate = threading.Event()

    def factory():  # a landmark function stuck on its first frame
        def fn(frame: np.ndarray) -> np.ndarray:
            gate.wait(5)
            return np.full((lm.N_POINTS, 3), np.nan, np.float32)
        return fn, lambda: None

    live = LiveRecognizer(rec, LiveConfig(frame_queue_size=2), landmark_fn_factory=factory)
    try:
        live.wait_ready()
        for i in range(100):
            live.submit_frame(np.zeros((4, 4, 3), np.uint8), t=i / 30)
        assert live._frames.qsize() <= 2
        assert live.stats.frames_dropped >= 95
    finally:
        gate.set()
        live.close()


def test_end_to_end_stream_and_clean_shutdown(rec: SignRecognizer) -> None:
    """Stream a held-out clip through the threads (fake landmark fn = identity on
    precomputed landmarks) and check a confident 'hello' appears, then shut down."""
    clip = heldout("hello")
    stream = np.concatenate([no_hands_like(clip, 20), clip, clip, no_hands_like(clip, 20)])
    closed = threading.Event()
    live = LiveRecognizer(
        rec,
        LiveConfig(interval_s=0.1, min_stable_windows=1, frame_queue_size=len(stream)),
        landmark_fn_factory=lambda: ((lambda f: f), closed.set),
    )
    seen = set()
    try:
        live.wait_ready()
        for i, frame in enumerate(stream):
            live.submit_frame(frame, t=i / 30)
            time.sleep(1 / 300)  # 10x faster than real time
            live.tick(now=i / 30)
            if live.decision.label:
                seen.add(live.decision.label)
        deadline = time.time() + 5
        while live.scheduler.busy and time.time() < deadline:
            time.sleep(0.01)
        seen |= set(live.decision.emitted)
    finally:
        live.close()
    assert "hello" in seen
    assert live.scheduler.submitted > 5
    assert closed.is_set()  # landmark backend (Holistic in production) was closed
    assert live.alive_threads == []
    assert len(live.buffer) == 0
    assert not live.submit_frame(stream[0])  # closed: refuses new frames
    live.close()  # idempotent


def test_shutdown_while_inference_is_running(rec: SignRecognizer) -> None:
    live = LiveRecognizer(rec, landmark_fn_factory=lambda: ((lambda f: f), lambda: None))
    gate = threading.Event()
    live.scheduler.maybe_submit(lambda: gate.wait(0.5), now=0.0)
    start = time.perf_counter()
    live.close(timeout=5)
    assert time.perf_counter() - start < 3
    assert live.alive_threads == []


def test_heldout_window_sanity_with_upstream_order(rec: SignRecognizer) -> None:
    """Guard: the live path must equal sequence_features(fit_length(resample(...)))."""
    clip = heldout("police")
    times = np.arange(len(clip)) / 30.0
    uniform = resample_window(times, clip, times[-1], 2.0, 30.0)
    np.testing.assert_allclose(uniform, clip, atol=1e-6, equal_nan=True)  # <2 s clip: unchanged
    ref = sequence_features(fit_length(resample(clip, 30.0, 30.0), 64), rec.feature_set)
    got = landmarks_to_features(
        uniform, 30.0, feature_set=rec.feature_set, model_fps=30.0, min_frames=8, max_frames=64
    ).features
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_window_without_new_landmarks_is_not_reclassified(rec: SignRecognizer) -> None:
    clip = heldout("hello")
    live = LiveRecognizer(rec, landmark_fn_factory=lambda: ((lambda f: f), lambda: None))
    calls = []
    original = live.smoother.update
    live.smoother.update = lambda r: calls.append(r) or original(r)  # type: ignore[method-assign]
    try:
        for i, f in enumerate(clip):
            live.buffer.append(i / 30, f)
        live._job()
        live._job()  # nothing new arrived: skipped
        live.buffer.append(len(clip) / 30, clip[-1])
        live._job()
    finally:
        live.close()
    assert len(calls) == 2 and live.stats.stale_windows == 1
