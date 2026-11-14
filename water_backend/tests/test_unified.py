"""Unified session: one frame stream -> both pipelines, gating, bounded queues, manual
capture, shutdown. Uses the real models with injected landmark sources (no camera)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.live import LiveConfig
from app.model import SignRecognizer
from app.models.alphabet_recognizer import AlphabetModel
from app.services.recognition_router import Kind, Mode
from app.services.unified_session import SessionConfig, UnifiedSession
from tests.test_live import heldout, no_hands_like

ROOT = Path(__file__).resolve().parent.parent
FIX = json.loads((Path(__file__).parent / "data" / "alphabet_reference_landmarks.json").read_text())


@pytest.fixture(scope="module")
def models() -> tuple[SignRecognizer, AlphabetModel]:
    return SignRecognizer(settings.model_dir), AlphabetModel(settings.model_dir / "alphabet")


class ScriptedHands:
    """Stands in for MediaPipe Hands: returns fixture landmarks for the given letter."""

    def __init__(self, letter: str | None = "b", gate: threading.Event | None = None) -> None:
        self.letter, self.gate, self.closed, self.calls = letter, gate, False, 0

    def detect(self, bgr, max_side=None):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(5)
        if self.letter is None:
            return None, 640, 480
        return np.array(FIX[self.letter]["points_px"], np.float32), 640, 480

    def close(self) -> None:
        self.closed = True


def identity_landmarks():
    return (lambda f: f), (lambda: None)


def make(models, mode=Mode.AUTO, hands=None, **cfg) -> UnifiedSession:
    word_rec, alpha = models
    hands = hands or ScriptedHands()
    s = UnifiedSession(word_rec, alpha, SessionConfig(**cfg), mode=mode,
                       word_landmark_fn_factory=identity_landmarks,
                       hand_detector_factory=lambda: hands)
    s.wait_ready()
    return s


def frame() -> np.ndarray:
    return no_hands_like(heldout("hello"), 1)[0]  # a [543, 3] landmark "frame"


def test_spell_mode_commits_stable_letter_once(models) -> None:
    s = make(models, Mode.SPELL)
    commits, letters = [], []
    try:
        for i in range(60):
            t = time.perf_counter()
            s.submit_frame(frame(), t)
            s.tick(t)
            time.sleep(0.01)
            snap = s.snapshot()
            commits += snap.letter_commits
            if snap.decision.kind is Kind.LETTER:
                letters.append(snap.decision.label)
    finally:
        s.close()
    assert commits == ["B"]  # held B -> exactly one commit (upstream no-repeat rule)
    assert letters and set(letters) == {"B"}
    assert s.alphabet.stats.processed >= 10


def test_one_frame_feeds_both_pipelines_and_modes_gate_them(models) -> None:
    s = make(models, Mode.AUTO)
    try:
        for i in range(5):
            s.submit_frame(frame(), i / 30)
        assert s.word.stats.frames_in == 5 and s.alphabet.stats.frames_in == 5
    finally:
        s.close()
    s = make(models, Mode.SPELL, run_word_in_spell=False)
    try:
        for i in range(5):
            s.submit_frame(frame(), i / 30)
        assert s.word.stats.frames_in == 0 and s.alphabet.stats.frames_in == 5
        s.set_mode(Mode.WORD)
        s.submit_frame(frame(), 1.0)
        assert s.word.stats.frames_in == 1
    finally:
        s.close()
    s = make(models, Mode.WORD, run_alphabet_in_word=False)
    try:
        s.submit_frame(frame(), 0.0)
        assert s.alphabet.stats.frames_in == 0 and s.word.stats.frames_in == 1
    finally:
        s.close()


def test_alphabet_queue_is_bounded_and_drops_stale_frames(models) -> None:
    gate = threading.Event()
    s = make(models, Mode.SPELL, hands=ScriptedHands(gate=gate))
    try:
        for i in range(100):
            s.submit_frame(frame(), i / 30)
        assert s.alphabet._frames.qsize() <= 1
        assert s.alphabet.stats.dropped >= 97
    finally:
        gate.set()
        s.close()


def test_stale_letter_result_is_not_shown_as_current(models) -> None:
    s = make(models, Mode.SPELL)
    try:
        for _ in range(15):
            s.submit_frame(frame(), time.perf_counter())
            time.sleep(0.01)
        time.sleep(0.1)
        fresh = s.snapshot()
        later = s.snapshot(time.perf_counter() + 5.0)  # no new frames for 5 s
    finally:
        s.close()
    assert fresh.decision.kind is Kind.LETTER
    assert later.decision.kind is Kind.NO_HANDS and not later.decision.letter.fresh
    assert later.decision.letter.note == "stale"


def test_manual_capture_classifies_the_shared_buffer(models) -> None:
    """SPACE: the next 2 s of the SAME landmark buffer -> one word prediction."""
    clip = heldout("hello")
    stream = np.concatenate([no_hands_like(clip, 15), clip, no_hands_like(clip, 25)])
    s = make(models, Mode.AUTO, hands=ScriptedHands(None),
             live=LiveConfig(frame_queue_size=len(stream) + 5, interval_s=10.0))
    try:
        t0 = 1000.0
        assert s.start_capture(t0) and not s.start_capture(t0)  # one capture at a time
        for i, f in enumerate(stream):
            s.submit_frame(f, t0 + i / 30)
        deadline = time.time() + 5
        while len(s.word.buffer) < len(stream) and time.time() < deadline:
            time.sleep(0.01)
        s.tick(t0 + 1.0)
        assert s.capture.state == "recording"
        s.tick(t0 + 2.01)
        deadline = time.time() + 5
        while s.capture.state != "idle" and time.time() < deadline:
            s.tick(t0 + 2.02)
            time.sleep(0.01)
        snap = s.snapshot(t0 + 2.1)
    finally:
        s.close()
    assert snap.manual is not None and snap.manual.label == "hello" and snap.manual.confident
    assert snap.decision.kind is Kind.WORD and snap.decision.label == "hello"


def test_session_close_releases_everything(models) -> None:
    hands = ScriptedHands()
    s = make(models, Mode.AUTO, hands=hands)
    for i in range(10):
        s.submit_frame(frame(), time.perf_counter())
        s.tick()
    s.start_capture()
    s.close()
    assert s.alive_threads == []
    assert hands.closed  # MediaPipe Hands closed by its worker
    assert len(s.word.buffer) == 0
    s.close()  # idempotent
    s.submit_frame(frame())  # ignored after close, no error


def test_app_exits_cleanly_when_camera_cannot_open() -> None:
    r = subprocess.run([sys.executable, "scripts/signbridge_app.py", "--source", "missing.mp4"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "could not open video file" in r.stderr
