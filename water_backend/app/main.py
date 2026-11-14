"""SignBridge recognition API.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.model import SignRecognizer
from app.preprocessing import VideoError, check_extension, preprocess_video
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    LabelItem,
    LabelsResponse,
    PredictResponse,
    TimingMs,
    TopPrediction,
)

logging.basicConfig(
    level=os.environ.get("SIGNBRIDGE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("signbridge")

_state: dict[str, object] = {"recognizer": None, "error": None}
_slots = threading.BoundedSemaphore(settings.max_concurrent_inference)
_CHUNK = 1 << 20


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        _state["recognizer"] = SignRecognizer(settings.model_dir)
        # Import MediaPipe now so the first request doesn't pay for it.
        import mediapipe  # noqa: F401
    except Exception as exc:  # keep serving /health so the failure is visible
        log.exception("Model failed to load")
        _state["error"] = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="SignBridge Recognition API", version="1.0.0", lifespan=lifespan)
# Hackathon demo: the frontend may be served from anywhere.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error={"code": code, "message": message})  # type: ignore[arg-type]
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.exception_handler(VideoError)
async def _video_error(_: Request, exc: VideoError) -> JSONResponse:
    return _error(exc.status_code, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    fields = ", ".join(".".join(str(p) for p in e["loc"]) for e in exc.errors())
    return _error(422, "invalid_request", f"Invalid request ({fields}). Send multipart field 'file'.")


def _recognizer() -> SignRecognizer:
    rec = _state["recognizer"]
    if rec is None:
        raise VideoError("model_unavailable", f"Model not loaded: {_state['error']}", 503)
    assert isinstance(rec, SignRecognizer)
    return rec


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    rec = _state["recognizer"]
    if isinstance(rec, SignRecognizer):
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=settings.model_name,
            model_version=rec.version,
            device=rec.device,
        )
    return HealthResponse(
        status="degraded",
        model_loaded=False,
        model_name=settings.model_name,
        error=str(_state["error"]),
    )


@app.get("/labels", response_model=LabelsResponse)
def labels() -> LabelsResponse:
    rec = _recognizer()
    items = [
        LabelItem(index=s["index"], label=s["label"], sign=s["sign"], held_out_top1=s["heldOutTop1"])
        for s in rec.signs
        if s["shipped"]
    ]
    return LabelsResponse(count=len(items), labels=items)


def _save_upload(file: UploadFile, suffix: str) -> Path:
    limit = int(settings.max_upload_mb * 1024 * 1024)
    fd, name = tempfile.mkstemp(prefix="signbridge-", suffix=suffix)
    path = Path(name)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := file.file.read(_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise VideoError(
                        "file_too_large",
                        f"File exceeds the {settings.max_upload_mb:.0f} MB limit.",
                        413,
                    )
                out.write(chunk)
        if size == 0:
            raise VideoError("empty_file", "The uploaded file is empty.", 400)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


# Sync handler: FastAPI runs it in a worker thread, so CPU-bound work doesn't block the loop.
@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
               415: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
               503: {"model": ErrorResponse}},
)
def predict(file: UploadFile = File(..., description="Video of ONE ASL sign (mp4/mov/webm)")) -> PredictResponse:
    start = time.perf_counter()
    rec = _recognizer()
    suffix = check_extension(file.filename, settings)
    path = _save_upload(file, suffix)
    try:
        with _slots:
            prepared, prep_ms = preprocess_video(
                path,
                settings,
                feature_set=rec.feature_set,
                model_fps=rec.fps,
                min_frames=rec.min_frames,
                max_frames=rec.max_frames,
            )
            pred = rec.predict(prepared.features, k=settings.top_k)
    finally:
        path.unlink(missing_ok=True)
    total = (time.perf_counter() - start) * 1000
    best = pred.top[0]
    log.info(
        "predict %s -> %s (%.3f, confident=%s) in %.0f ms",
        file.filename, best.label, best.confidence, pred.confident and not prepared.partial_view, total,
    )
    return PredictResponse(
        prediction=best.label,
        confidence=round(best.confidence, 4),
        confident=pred.confident and not prepared.partial_view,
        not_sure_threshold=round(rec.not_sure_below, 4),
        top_predictions=[
            TopPrediction(label=g.label, sign=g.sign, confidence=round(g.confidence, 4))
            for g in pred.top
        ],
        inference_time_ms=round(total, 1),
        timing_ms=TimingMs(
            preprocessing=round(prep_ms, 1), model=round(pred.inference_ms, 2), total=round(total, 1)
        ),
        model_name=settings.model_name,
        model_version=rec.version,
        video=prepared.stats,
        warnings=prepared.warnings,
    )
