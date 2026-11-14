"""API tests against the real pretrained model (no mocks)."""

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
    with TestClient(app) as c:  # runs the lifespan -> loads the model
        yield c


def _post(client: TestClient, name: str, data: bytes, ctype: str = "video/mp4"):
    return client.post("/predict", files={"file": (name, data, ctype)})


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"


def test_labels(client: TestClient) -> None:
    body = client.get("/labels").json()
    assert body["count"] == 250
    labels = {item["label"] for item in body["labels"]}
    assert {"hello", "thank you", "yes", "no"} <= labels


@pytest.mark.parametrize(("name", "expected"), [("hello.mp4", "hello"), ("thank_you.mp4", "thank you")])
def test_predict_real_video(client: TestClient, name: str, expected: str) -> None:
    resp = _post(client, name, (EXAMPLES / name).read_bytes())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "success"
    assert len(body["top_predictions"]) == 3
    confs = [p["confidence"] for p in body["top_predictions"]]
    assert confs == sorted(confs, reverse=True) and 0 < confs[0] <= 1
    # Both example clips are recognised correctly by this checkpoint (verified manually).
    assert body["prediction"] == expected


def test_unsupported_extension(client: TestClient) -> None:
    resp = _post(client, "notes.txt", b"hello", "text/plain")
    assert resp.status_code == 415
    assert resp.json()["error"]["code"] == "unsupported_file_type"


def test_empty_file(client: TestClient) -> None:
    resp = _post(client, "empty.mp4", b"")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "empty_file"


def test_corrupt_video(client: TestClient) -> None:
    resp = _post(client, "corrupt.mp4", b"this is not a video" * 100)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] in {"unreadable_video", "empty_or_corrupt_video"}


def test_truncated_video(client: TestClient) -> None:
    data = (EXAMPLES / "hello.mp4").read_bytes()[:4000]
    resp = _post(client, "truncated.mp4", data)
    assert resp.status_code == 422
    assert resp.json()["status"] == "error"


def test_video_without_hands(client: TestClient, tmp_path: Path) -> None:
    path = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(45):
        writer.write(np.full((240, 320, 3), 40 + i, dtype=np.uint8))
    writer.release()
    resp = _post(client, "blank.mp4", path.read_bytes())
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "no_hands_detected"


def test_missing_file_field(client: TestClient) -> None:
    resp = client.post("/predict")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"
