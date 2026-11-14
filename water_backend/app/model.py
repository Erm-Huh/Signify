"""ONNX Runtime wrapper around the pretrained GestureSign model.

Confidence = softmax(logits / temperature) over the shipped signs, with the
temperature and the "not sure" threshold that upstream fitted on held-out
validation signers (``model-meta.json`` -> ``calibration``). This mirrors
``web/src/ml/decision.ts`` in the upstream repo.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from app.gesturesign.features import FEATURE_VERSION, sequence_feature_dim

log = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Guess:
    index: int
    sign: str  # raw asl-signs class name, e.g. "callonphone"
    label: str  # display name, e.g. "call (on phone)"
    confidence: float


@dataclass(frozen=True)
class Prediction:
    top: list[Guess]
    confident: bool
    inference_ms: float


class SignRecognizer:
    def __init__(self, model_dir: Path) -> None:
        onnx_path = model_dir / "model.onnx"
        meta_path = model_dir / "model-meta.json"
        labels_path = model_dir / "labels.json"
        for p in (onnx_path, meta_path, labels_path):
            if not p.is_file():
                raise ModelLoadError(
                    f"Missing {p}. Run `python scripts/download_model.py` to fetch the checkpoint."
                )
        self.meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("metrics") is None or str(self.meta.get("version", "")).startswith("stub"):
            raise ModelLoadError("model-meta.json describes an untrained stub model; refusing.")
        if self.meta["featureVersion"] != FEATURE_VERSION:
            raise ModelLoadError(
                f"Model expects feature version {self.meta['featureVersion']}, "
                f"vendored preprocessing implements {FEATURE_VERSION}."
            )
        self.feature_set = self.meta["featureSet"]
        self.feature_dim = int(self.meta["featureDim"])
        if sequence_feature_dim(self.feature_set) != self.feature_dim:
            raise ModelLoadError("featureDim in model-meta.json does not match the preprocessing.")
        self.fps = float(self.meta["fps"])
        self.min_frames = int(self.meta["minFrames"])
        self.max_frames = int(self.meta["maxFrames"])
        self.temperature = float(self.meta["calibration"]["temperature"])
        self.not_sure_below = float(self.meta["calibration"]["notSureBelow"])

        signs = json.loads(labels_path.read_text(encoding="utf-8"))["signs"]
        signs.sort(key=lambda s: s["index"])
        if [s["index"] for s in signs] != list(range(int(self.meta["classes"]))):
            raise ModelLoadError("labels.json indices do not cover every model class.")
        self.signs: list[dict[str, Any]] = signs
        self.shipped = np.array([bool(s["shipped"]) for s in signs])

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # 1.4 M parameters: the CPU provider is a few ms; no accelerator needed.
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.device = "cpu"
        self.input_name = self.session.get_inputs()[0].name
        # Warm-up so the first request is not slower than the rest.
        self._logits(np.zeros((self.min_frames, self.feature_dim), dtype=np.float32))
        log.info(
            "Loaded %s (%d classes, feature set %s, %d-dim) on %s",
            self.meta["version"],
            len(signs),
            self.feature_set,
            self.feature_dim,
            self.device,
        )

    @property
    def version(self) -> str:
        return str(self.meta["version"])

    def _logits(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        x = np.ascontiguousarray(features[None], dtype=np.float32)
        logits: npt.NDArray[np.float32] = self.session.run(["logits"], {self.input_name: x})[0][0]
        return logits

    def probabilities(self, logits: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
        """Calibrated softmax over shipped signs (unshipped get 0), as upstream decision.ts."""
        z = logits.astype(np.float64) / self.temperature
        z = np.where(self.shipped, z, -np.inf)
        z -= z[self.shipped].max()
        e = np.exp(z)
        out: npt.NDArray[np.float64] = e / e.sum()
        return out

    def predict(self, features: npt.NDArray[np.float32], k: int = 3) -> Prediction:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"expected [T, {self.feature_dim}] features, got {features.shape}")
        if not self.min_frames <= features.shape[0] <= self.max_frames:
            raise ValueError(f"T={features.shape[0]} outside [{self.min_frames}, {self.max_frames}]")
        start = time.perf_counter()
        logits = self._logits(features)
        elapsed = (time.perf_counter() - start) * 1000
        if not np.isfinite(logits).all():
            raise RuntimeError("model returned non-finite logits")
        probs = self.probabilities(logits)
        order = np.argsort(-probs)[:k]
        top = [
            Guess(int(i), self.signs[i]["sign"], self.signs[i]["label"], float(probs[i]))
            for i in order
        ]
        return Prediction(top, top[0].confidence >= self.not_sure_below, elapsed)
