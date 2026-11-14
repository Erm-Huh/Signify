"""Inference-only GestureSign adapter and bundled-replay CLI (Python 3.11).

Reuses Shashank Yadav's MIT-licensed GestureSign preprocessing in the bundled
runtime subset without installing its training package. See WORD_RECOGNITION.md for
attribution, asset requirements, and the upstream license location.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import importlib
import json
from pathlib import Path
import sys

import numpy as np


DEFAULT_REPOSITORY = Path(__file__).resolve().parent / "vendor" / "GestureSign"


def _preprocessing(repository):
    """Import upstream's NumPy-only modules, never its dataset/training helpers."""
    source = repository / "pipeline" / "src"
    package = source / "gesturesign" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError(f"GestureSign preprocessing missing: {package}")
    existing = sys.modules.get("gesturesign")
    if existing and Path(existing.__file__).resolve() != package.resolve():
        raise ValueError("A different gesturesign package is already imported; use a fresh process")
    # Restore the import search path immediately; only these upstream modules
    # (features, landmarks, types, resample, and the empty package init) load.
    sys.path.insert(0, str(source))
    try:
        features = importlib.import_module("gesturesign.features")
        resampling = importlib.import_module("gesturesign.resample")
    finally:
        sys.path.pop(0)
    return features, resampling


@dataclass(frozen=True)
class Prediction:
    index: int
    sign: str
    label: str
    confidence: float


@dataclass(frozen=True)
class Recognition:
    status: str  # sign, not-sure, or nothing
    top: tuple[Prediction, ...] = ()
    reason: str | None = None
    input_frames: int = 0

    @property
    def accepted(self):
        return self.status == "sign"


def decide(logits, labels, temperature, threshold, *, partial=False):
    """Match upstream decision.ts: shipped-only calibrated softmax and top three.

    Partial framing rejects a segment while preserving its guesses. Nonfinite
    logits are never interpreted as a word. Confidence is not an accuracy claim.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(labels),):
        raise ValueError("logits must contain one value per label")
    shipped = np.array([label["shipped"] for label in labels], dtype=bool)
    if not shipped.any():
        return Recognition("nothing", reason="no-shipped-labels")
    if not np.isfinite(logits).all():
        return Recognition("nothing", reason="invalid-logits")
    indices = np.flatnonzero(shipped)
    scaled = logits[indices] / temperature
    probabilities = np.exp(scaled - scaled.max())
    probabilities /= probabilities.sum()
    ranking = np.argsort(-probabilities, kind="stable")[:3]
    top = tuple(Prediction(int(indices[i]), labels[indices[i]]["sign"],
                           labels[indices[i]]["label"], float(probabilities[i]))
                for i in ranking)
    if partial:
        return Recognition("not-sure", top, "partial")
    if top[0].confidence < threshold:
        return Recognition("not-sure", top, "low-confidence")
    return Recognition("sign", top)


class WordRecognizer:
    """Classify an already segmented [T, 543, 3] normalized landmark sequence.

    Missing points must be NaN. No hand centering, pixel conversion, landmark
    smoothing, segmentation, or letter recognition is performed here.
    """

    def __init__(self, repository=DEFAULT_REPOSITORY, model_dir=None):
        self.repository = Path(repository).resolve()
        self.features, self.resampling = _preprocessing(self.repository)
        assets = Path(model_dir) if model_dir is not None else self.repository / "web/public/models"
        for name in ("model.onnx", "model-meta.json", "labels.json"):
            if not (assets / name).is_file():
                raise FileNotFoundError(f"Missing GestureSign asset: {assets / name}")
        self.meta = json.loads((assets / "model-meta.json").read_text(encoding="utf-8"))
        self.labels = json.loads((assets / "labels.json").read_text(encoding="utf-8"))["signs"]
        self._validate_metadata()
        # ONNX Runtime is the only inference engine; no Torch/TensorFlow imports.
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(assets / "model.onnx"),
                                            providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = {output.name: output for output in self.session.get_outputs()}
        if (len(inputs) != 1 or inputs[0].name != "frames"
                or inputs[0].type != "tensor(float)"
                or inputs[0].shape != [1, "T", self.meta["featureDim"]]
                or "logits" not in outputs or "embedding" not in outputs
                or outputs["logits"].shape != [1, self.meta["classes"]]
                or outputs["embedding"].shape != [1, self.meta["embeddingDim"]]):
            raise ValueError("GestureSign ONNX I/O does not match its metadata")

    def _validate_metadata(self):
        meta = self.meta
        if meta["featureVersion"] != self.features.FEATURE_VERSION:
            raise ValueError("Unsupported GestureSign featureVersion")
        if meta["featureSet"] not in self.features.FEATURE_SETS:
            raise ValueError("Unsupported GestureSign featureSet")
        if meta["featureDim"] != self.features.sequence_feature_dim(meta["featureSet"]):
            raise ValueError("featureDim does not match featureSet")
        if (not np.isfinite(meta["fps"]) or meta["fps"] <= 0
                or not isinstance(meta["minFrames"], int)
                or not isinstance(meta["maxFrames"], int)
                or not 2 <= meta["minFrames"] <= meta["maxFrames"]):
            raise ValueError("Invalid model FPS or frame limits")
        if meta["io"] != {"input": "frames", "outputs": ["logits", "embedding"], "opset": 17}:
            raise ValueError("Unsupported GestureSign I/O metadata")
        if len(self.labels) != meta["classes"] or not self.labels:
            raise ValueError("Label count does not match model classes")
        for index, label in enumerate(self.labels):
            if (label.get("index") != index or not isinstance(label.get("sign"), str)
                    or not isinstance(label.get("label"), str)
                    or not isinstance(label.get("shipped"), bool)):
                raise ValueError(f"Invalid label entry at index {index}")
        calibration = meta["calibration"]
        # Share calibration validation with the actual decision path.
        decide(np.zeros(len(self.labels)), self.labels, calibration["temperature"],
               calibration["notSureBelow"])

    def prepare(self, frames, src_fps):
        """Preserve upstream order: resample -> fit length -> sequence features."""
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 3 or frames.shape[1:] != (543, 3):
            raise ValueError("frames must have shape [T, 543, 3]")
        if not np.isfinite(src_fps) or src_fps <= 0:
            raise ValueError("src_fps must be finite and positive")
        if np.isinf(frames).any():
            raise ValueError("Use NaN for missing landmarks, not infinity")
        fitted = self.resampling.fit_length(
            self.resampling.resample(frames, src_fps, self.meta["fps"]),
            self.meta["maxFrames"])
        return self.features.sequence_features(fitted, self.meta["featureSet"])

    def recognize(self, frames, src_fps, *, partial=False):
        features = self.prepare(frames, src_fps)
        length = len(features)
        if length < self.meta["minFrames"]:
            return Recognition("nothing", reason="too-short", input_frames=length)
        # Obvious empty input should not receive a learned class bias. Presence
        # flags and NaNs otherwise pass through the upstream features unchanged.
        frames = np.asarray(frames)
        if not (np.isfinite(frames[:, 468:489, 0]).any()
                or np.isfinite(frames[:, 522:543, 0]).any()):
            return Recognition("nothing", reason="no-hands", input_frames=length)
        logits = self.session.run(["logits"], {"frames": features[None]})[0]
        calibration = self.meta["calibration"]
        result = decide(logits[0], self.labels, calibration["temperature"],
                        calibration["notSureBelow"], partial=partial)
        return replace(result, input_frames=length)


def load_replay(path):
    """Decode upstream sparse x/y JSON to normalized frames; null means missing."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    points = data["points"]
    if (len(set(points)) != len(points)
            or any(not isinstance(p, int) or not 0 <= p < 543 for p in points)):
        raise ValueError("Replay point indices must be unique integers in [0, 543)")
    clips = []
    for clip in data["clips"]:
        values = np.asarray(clip["frames"], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 2 * len(points):
            raise ValueError(f"Invalid replay frame shape: {clip['id']}")
        frames = np.full((len(values), 543, 3), np.nan, dtype=np.float32)
        frames[:, points, :2] = values.reshape(len(values), len(points), 2)
        clips.append({**clip, "frames": frames})
    return data.get("credit"), clips


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run GestureSign's six replay clips on CPU")
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--model-dir", type=Path, help="Directory containing model.onnx and its two JSON files")
    parser.add_argument("--replay", type=Path, help="Replay JSON; defaults to the bundled clips")
    args = parser.parse_args(argv)
    try:
        recognizer = WordRecognizer(args.repository, args.model_dir)
        credit, clips = load_replay(args.replay or args.repository / "web/public/replay/clips.json")
        print(f"Model: {recognizer.meta['version']} | {len(recognizer.labels)} signs | CPU")
        print(f"Temperature: {recognizer.meta['calibration']['temperature']:.6f}; "
              f"reject below: {recognizer.meta['calibration']['notSureBelow']:.6f}")
        if credit:
            print(f"Replay credit: {credit}")
        for clip in clips:
            result = recognizer.recognize(clip["frames"], clip["fps"])
            guesses = "; ".join(f"{p.label} {p.confidence:.1%}" for p in result.top) or "no guesses"
            print(f"{clip['id']} expected={clip.get('sign', '?')} | {guesses} | "
                  f"{'ACCEPT' if result.accepted else 'REJECT'} "
                  f"({result.reason or 'calibrated-confidence'}; T={result.input_frames})")
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"Word recognition error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
