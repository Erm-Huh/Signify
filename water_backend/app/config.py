"""Backend settings. Override any of them with environment variables (prefix SIGNBRIDGE_)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"SIGNBRIDGE_{name}", default)


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(_env("MODEL_DIR", str(ROOT / "models")))
    model_name: str = "GestureSign (asl-signs, 250 signs)"

    # Upload validation.
    max_upload_mb: float = float(_env("MAX_UPLOAD_MB", "50"))
    allowed_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset({".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv"})
    )
    # Longer videos are rejected: the model classifies ONE isolated sign (~1-4 s).
    max_video_seconds: float = float(_env("MAX_VIDEO_SECONDS", "15"))

    # MediaPipe Holistic (legacy graph; the asl-signs training data was made with it).
    holistic_model_complexity: int = int(_env("HOLISTIC_COMPLEXITY", "1"))
    # Frames are downscaled so the long side is at most this many pixels. MediaPipe
    # landmarks are normalised to the image, so uniform downscaling leaves them unchanged.
    max_frame_side: int = int(_env("MAX_FRAME_SIDE", "960"))

    # Concurrent /predict requests allowed to run MediaPipe at once (CPU bound).
    max_concurrent_inference: int = int(_env("MAX_CONCURRENT", "2"))
    top_k: int = 3


settings = Settings()
