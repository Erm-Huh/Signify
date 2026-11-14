"""Alphabet API endpoints (real models, real MediaPipe). Existing endpoints are covered by
test_api.py; this file only adds."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def frame_png(index: int) -> bytes:
    cap = cv2.VideoCapture(str(EXAMPLES / "hello.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, f = cap.read()
    cap.release()
    assert ok
    return cv2.imencode(".png", f)[1].tobytes()


def test_health_reports_both_models(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["model_loaded"] is True and body["alphabet_model_loaded"] is True
    assert body["status"] == "ok"


def test_alphabet_labels(client: TestClient) -> None:
    body = client.get("/labels/alphabet").json()
    assert body["count"] == 26 and body["labels"][:3] == ["A", "B", "C"] and body["labels"][-1] == "Z"


def test_still_image_prediction(client: TestClient) -> None:
    # frame 30 of the PopSign "hello" clip: a flat hand, which IS the B handshape
    r = client.post("/predict/alphabet", files={"file": ("hand.png", frame_png(30), "image/png")})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["input_type"] == "image" and b["hand_detected"] and not b["motion_supported"]
    assert b["stable"] is None and b["threshold"] == 0.75
    confs = [p["confidence"] for p in b["top_predictions"]]
    assert len(confs) == 3 and confs == sorted(confs, reverse=True)
    assert b["prediction"] == b["top_predictions"][0]["label"] == "B"


def test_image_without_hand(client: TestClient) -> None:
    blank = cv2.imencode(".png", np.full((240, 320, 3), 128, np.uint8))[1].tobytes()
    b = client.post("/predict/alphabet", files={"file": ("blank.png", blank, "image/png")}).json()
    assert b["hand_detected"] is False and b["prediction"] is None and b["top_predictions"] == []


def test_video_prediction_runs_temporal_pipeline(client: TestClient) -> None:
    data = (EXAMPLES / "hello.mp4").read_bytes()
    r = client.post("/predict/alphabet", files={"file": ("hello.mp4", data, "video/mp4")})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["input_type"] == "video" and b["motion_supported"] is True
    assert b["frames_processed"] > 30 and b["frames_with_hand"] > 0
    assert isinstance(b["committed_letters"], list)


@pytest.mark.parametrize(
    ("name", "data", "status", "code"),
    [
        ("notes.txt", b"hi", 415, "unsupported_file_type"),
        ("broken.png", b"not an image at all", 422, "unreadable_image"),
        ("empty.jpg", b"", 400, "empty_file"),
        ("broken.mp4", b"not a video" * 50, 422, "unreadable_video"),
    ],
)
def test_alphabet_errors(client: TestClient, name, data, status, code) -> None:
    r = client.post("/predict/alphabet", files={"file": (name, data, "application/octet-stream")})
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_word_endpoint_unchanged(client: TestClient) -> None:
    r = client.post("/predict", files={"file": ("hello.mp4", (EXAMPLES / "hello.mp4").read_bytes(), "video/mp4")})
    assert r.status_code == 200 and r.json()["prediction"] == "hello"
