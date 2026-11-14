"""Checks that the vendored preprocessing and the pretrained model match upstream.

* ``feature_fixtures.json`` is upstream's train/serve parity fixture
  (shared/fixtures/features.json): our features must match it within its tolerance.
* ``heldout_clips.json`` holds six real asl-signs landmark clips from signers the model
  never trained on (upstream web/public/replay/clips.json, CC BY 4.0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.gesturesign import landmarks as lm
from app.gesturesign.features import frame_features, sequence_features
from app.gesturesign.resample import fit_length, resample
from app.model import SignRecognizer

DATA = Path(__file__).parent / "data"


def expand(points: list[int], flat_frames: list[list[float | None]]) -> np.ndarray:
    """Upstream JSON (x, y of selected points per frame) -> [T, 543, 3] with NaN."""
    T = len(flat_frames)
    out = np.full((T, lm.N_POINTS, 3), np.nan, dtype=np.float32)
    arr = np.array(
        [[np.nan if v is None else v for v in f] for f in flat_frames], dtype=np.float32
    ).reshape(T, len(points), 2)
    out[:, points, :2] = arr
    return out


@pytest.fixture(scope="module")
def recognizer() -> SignRecognizer:
    return SignRecognizer(settings.model_dir)


def test_features_match_upstream_fixture() -> None:
    fixture = json.loads((DATA / "feature_fixtures.json").read_text())
    tol = fixture["tolerance"]
    for case in fixture["cases"]:
        frames = expand(fixture["points"], case["frames"])
        expected = np.array(case["expected"]["features"], dtype=np.float64)
        if case["kind"] == "sequence":
            got = sequence_features(
                fit_length(resample(frames, case["srcFps"], case["fps"]), case["maxFrames"]),
                case["featureSet"],
            )
            assert got.shape[0] == case["expected"]["length"], case["name"]
        else:
            got = frame_features(frames[0], case["featureSet"])
        assert got.shape == expected.shape, case["name"]
        np.testing.assert_allclose(got, expected, atol=tol, err_msg=case["name"])


def test_model_loads_real_checkpoint(recognizer: SignRecognizer) -> None:
    assert recognizer.feature_dim == 460
    assert len(recognizer.signs) == 250
    assert recognizer.meta["metrics"]["top1"] > 0.5  # a trained model, not the stub


def test_heldout_clips(recognizer: SignRecognizer) -> None:
    replay = json.loads((DATA / "heldout_clips.json").read_text())
    hits_top1 = hits_top3 = 0
    for clip in replay["clips"]:
        frames = expand(replay["points"], clip["frames"])
        feats = sequence_features(
            fit_length(resample(frames, clip["fps"], recognizer.fps), recognizer.max_frames),
            recognizer.feature_set,
        )
        pred = recognizer.predict(feats, k=3)
        signs = [g.sign for g in pred.top]
        print(f"{clip['sign']:>8} -> {[(g.label, round(g.confidence, 3)) for g in pred.top]}")
        hits_top1 += signs[0] == clip["sign"]
        hits_top3 += clip["sign"] in signs
    print(f"top-1 {hits_top1}/6, top-3 {hits_top3}/6")
    # Upstream reports 72% top-1 / 85% top-3 on held-out signers. These six clips are
    # what upstream shows in its demo; a broken pipeline would score ~0/6 (chance 1/250).
    assert hits_top3 >= 5
