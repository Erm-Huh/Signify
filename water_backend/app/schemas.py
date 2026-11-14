from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str
    model_version: str | None = None
    device: str | None = None
    error: str | None = None


class TopPrediction(BaseModel):
    label: str = Field(description="Display name, e.g. 'call (on phone)'")
    sign: str = Field(description="Raw asl-signs class name, e.g. 'callonphone'")
    confidence: float = Field(description="Temperature-calibrated softmax probability")


class TimingMs(BaseModel):
    preprocessing: float = Field(description="Decode + MediaPipe Holistic + feature extraction")
    model: float = Field(description="ONNX forward pass only")
    total: float


class PredictResponse(BaseModel):
    status: Literal["success"] = "success"
    prediction: str
    confidence: float
    confident: bool = Field(
        description="False when confidence is below the calibrated 'not sure' threshold "
        "or the signer is only partly in view. Show 'Not sure' in the UI when false."
    )
    not_sure_threshold: float
    top_predictions: list[TopPrediction]
    inference_time_ms: float = Field(description="Total server-side time for the request")
    timing_ms: TimingMs
    model_name: str
    model_version: str
    video: dict[str, float | int | str | bool]
    warnings: list[str] = []


class LabelItem(BaseModel):
    index: int
    label: str
    sign: str
    held_out_top1: float | None = Field(
        description="Upstream per-sign top-1 accuracy on held-out signers"
    )


class LabelsResponse(BaseModel):
    count: int
    labels: list[LabelItem]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: ErrorDetail
