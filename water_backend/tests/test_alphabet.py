"""Alphabet recognizer: vendoring, model I/O, upstream parity, temporal logic (no camera).

``alphabet_reference_landmarks.json`` holds MediaPipe Hands pixel landmarks and the logits
upstream's own code produced, for the 23 letters whose hand MediaPipe found in upstream's
``docs/asl-alphabet-reference.png`` (extracted with upstream's prepare_dataset settings:
static mode, complexity 1, 25% pad). Those photos look like upstream's training data, so
this is a parity/wiring check, not an accuracy benchmark.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from app.asl_alphabet import preprocessing as pp
from app.asl_alphabet.spelling import COMMIT_CONFIDENCE, COMMIT_FRAMES, CONFIDENCE_FLOOR
from app.config import settings
from app.models.alphabet_recognizer import AlphabetModel, AlphabetStream

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = json.loads((Path(__file__).parent / "data" / "alphabet_reference_landmarks.json").read_text())
UPSTREAM_SHA256 = {  # upstream commit 45ff2c0e26925c50f4546073bd573fe3ca7a5e6e
    "preprocessing.py": "bdeb88681c9344fd5406bec089431e7ffdf3e54714be5146a358bf6e7819bb97",
    "smoothing.py": "c4dd8fc5d57a262bb5be0fa107d495f60d8014e81c718f94bb1a25dc9b81e570",
    "motion.py": "ae76a39723a8939c84fe779ea9d6e1767c67ebf271169c9ba33e34d99be64c5b",
}
SPELLING_BUFFER_SHA256 = "60f3dd58f915943d85289e0040a4394cd678dbe41e8304df9372dd9d83ef88af"
MODEL_SHA256 = "c3843cabc57032080a81e5995b578139c90631080af1e6d070df1c6dfdc1172b"


@pytest.fixture(scope="module")
def model() -> AlphabetModel:
    return AlphabetModel(settings.model_dir / "alphabet")


def pts(letter: str) -> np.ndarray:
    return np.array(FIXTURE[letter]["points_px"], dtype=np.float32)


def test_vendored_files_match_upstream() -> None:
    for name, digest in UPSTREAM_SHA256.items():
        data = (ROOT / "app" / "asl_alphabet" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
    s = (ROOT / "app" / "asl_alphabet" / "spelling.py").read_text()
    seg = s[s.index("class SpellingBuffer:"):]
    assert hashlib.sha256(seg.encode()).hexdigest() == SPELLING_BUFFER_SHA256
    assert (CONFIDENCE_FLOOR, COMMIT_CONFIDENCE, COMMIT_FRAMES) == (0.50, 0.75, 10)
    onnx = (settings.model_dir / "alphabet" / "asl_cnn_model.onnx").read_bytes()
    assert hashlib.sha256(onnx).hexdigest() == MODEL_SHA256


def test_model_io_and_labels(model: AlphabetModel) -> None:
    inp = model.session.get_inputs()[0]
    assert inp.shape[1:] == [96, 96, 3] and inp.type == "tensor(float)"
    out = {o.name: o.shape for o in model.session.get_outputs()}
    assert out["logits"][1:] == [26] and model.output == "logits"
    assert model.class_names == list("abcdefghijklmnopqrstuvwxyz")
    logits = model.logits(np.zeros((96, 96, 3), np.uint8))
    assert logits.shape == (26,)
    p = model.softmax(logits)
    assert p.sum() == pytest.approx(1.0) and (p >= 0).all()


def test_preprocessing_parity_with_upstream_logits(model: AlphabetModel) -> None:
    """Our model wrapper on upstream-extracted landmarks reproduces upstream's logits."""
    hits = 0
    for letter, row in FIXTURE.items():
        img = pp.render_skeleton(pp.normalize_landmarks(pts(letter)), size=model.size)
        assert img.shape == (96, 96, 3) and img.dtype == np.uint8
        logits = model.logits(img)
        np.testing.assert_allclose(logits, np.array(row["logits"]), atol=2e-3, err_msg=letter)
        hits += model.class_names[int(np.argmax(logits))] == letter
    assert len(FIXTURE) == 23 and hits == 21  # measured: h->j and w->u are the misses


def test_top3_sorted_and_uppercase(model: AlphabetModel) -> None:
    stream = AlphabetStream(model)
    r = stream.update(pts("w"), 0.0)  # an ambiguous one: several letters share mass
    confs = [g.confidence for g in r.top]
    assert len(r.top) == 3 and confs == sorted(confs, reverse=True)
    assert all(g.label.isupper() and len(g.label) == 1 for g in r.top)
    assert r.best == r.top[0].label and r.confidence == pytest.approx(confs[0])


def test_missing_hand_resets_state(model: AlphabetModel) -> None:
    stream = AlphabetStream(model)
    for i in range(5):
        stream.update(pts("b"), i / 30)
    r = stream.update(None, 5 / 30)
    assert not r.hand_detected and r.best is None and r.top == [] and not r.stable
    assert stream.score_history == [] and len(stream.tracker.buffer) == 0


def test_low_confidence_is_not_committed(model: AlphabetModel) -> None:
    stream = AlphabetStream(model)
    results = [stream.update(pts("w"), i / 30) for i in range(20)]  # w: top score 0.375
    assert all(not r.passes_threshold and r.committed is None for r in results)
    assert results[-1].uncertain  # <= CONFIDENCE_FLOOR
    assert not results[-1].stable


def test_stability_commit_and_no_repeats(model: AlphabetModel) -> None:
    stream = AlphabetStream(model)
    commits, stable_at = [], None
    for i in range(30):  # a still B held for a second
        r = stream.update(pts("b"), i / 30)
        if r.committed:
            commits.append((i, r.committed))
        if r.stable and stable_at is None:
            stable_at = i
    assert commits == [(COMMIT_FRAMES - 1, "B")]  # 10th agreeing frame, once
    assert stable_at == COMMIT_FRAMES - 1
    # hand leaves and comes back: upstream allows the same letter again
    stream.update(None, 1.1)
    later = [stream.update(pts("b"), 1.2 + i / 30).committed for i in range(12)]
    assert later.count("B") == 1
    # a different letter in between (held long enough to win the 8-frame average) also
    # re-arms B; one stray frame does not, because upstream averages the last 8 frames
    stream.update(pts("a"), 2.0)
    assert stream.update(pts("b"), 2.05).committed is None
    # (switching photos also jumps the hand position: upstream reads that as motion and
    # suppresses commits for a few frames, so each letter is held ~0.7 s)
    seq = [stream.update(pts("a"), 2.1 + i / 30).committed for i in range(20)]
    seq += [stream.update(pts("b"), 2.8 + i / 30).committed for i in range(20)]
    assert [c for c in seq if c] == ["A", "B"]


def _trajectory(base: np.ndarray, offsets: np.ndarray, hand: float) -> list[np.ndarray]:
    return [base + o * hand for o in offsets]


def test_j_motion_overrides_static_i(model: AlphabetModel) -> None:
    """J = I handshape + a descent and a hook (upstream motion.py). Static I stays I."""
    base = pts("i")
    hand = float((base.max(0) - base.min(0)).max())
    still = AlphabetStream(model)
    for i in range(30):
        r = still.update(base, i / 30)
    assert r.static_best == "I" and r.best == "I" and r.motion_override is None

    down = np.stack([np.zeros(25), np.linspace(0, 0.9, 25)], 1)
    hook = np.stack([np.linspace(0, -0.6, 15), np.linspace(0.9, 0.7, 15)], 1)  # curls back up
    hold = np.tile([[-0.6, 0.7]], (6, 1))
    moving = AlphabetStream(model)
    results = [moving.update(p, i / 30) for i, p in enumerate(_trajectory(base, np.vstack([down, hook, hold]), hand))]
    assert any(r.best == "J" and r.motion_override == "J" and r.static_best == "I" for r in results)


def test_z_motion_overrides_static_d(model: AlphabetModel) -> None:
    base = pts("d")
    hand = float((base.max(0) - base.min(0)).max())
    a = np.stack([np.linspace(0, 0.8, 14), np.zeros(14)], 1)
    b = np.stack([np.linspace(0.8, 0, 14), np.linspace(0, 0.6, 14)], 1)
    c = np.stack([np.linspace(0, 0.8, 14), np.full(14, 0.6)], 1)
    hold = np.tile([[0.8, 0.6]], (6, 1))
    stream = AlphabetStream(model)
    results = [stream.update(p, i / 30) for i, p in enumerate(_trajectory(base, np.vstack([a, b, c, hold]), hand))]
    assert any(r.best == "Z" and r.motion_override == "Z" for r in results)
    assert all(r.static_best == "D" for r in results)  # the classifier itself never saw a Z


def test_scores_are_documented_as_uncalibrated() -> None:
    import app.models.alphabet_recognizer as ar

    assert "NOT temperature-calibrated" in (ar.__doc__ or "")
