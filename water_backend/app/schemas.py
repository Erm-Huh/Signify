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
    # Added with the alphabet model (top-level fields above still describe the word model).
    alphabet_model_loaded: bool | None = None
    alphabet_model_name: str | None = None
    alphabet_error: str | None = None


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


# ----------------------------------------------------------------------------- alphabet


class LetterPrediction(BaseModel):
    label: str = Field(description="Letter A-Z")
    confidence: float = Field(description="Softmax score; NOT temperature-calibrated")


class AlphabetPredictResponse(BaseModel):
    status: Literal["success"] = "success"
    input_type: Literal["image", "video"]
    hand_detected: bool
    prediction: str | None = Field(description="Best letter (after J/Z motion override for video)")
    confidence: float = Field(description="Uncalibrated softmax score of the classifier's top letter")
    passes_threshold: bool = Field(description="confidence >= upstream COMMIT_CONFIDENCE (0.75)")
    uncertain: bool = Field(description="confidence <= upstream CONFIDENCE_FLOOR (0.50)")
    stable: bool | None = Field(
        description="Video only: upstream commit rule held at the end (10 still frames >= 0.75)"
    )
    threshold: float
    top_predictions: list[LetterPrediction]
    motion_supported: bool = Field(
        description="True for video: J/Z can be resolved from movement. False for still images."
    )
    motion_override: str | None = None
    committed_letters: list[str] | None = Field(
        default=None, description="Video only: letters the upstream spelling rule committed, in order"
    )
    frames_processed: int | None = None
    frames_with_hand: int | None = None
    inference_time_ms: float
    model_name: str
    warnings: list[str] = []


class AlphabetLabelsResponse(BaseModel):
    count: int
    labels: list[str]
