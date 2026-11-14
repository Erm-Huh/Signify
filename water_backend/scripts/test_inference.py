"""Run the full recognition pipeline on a local video, without the server.

    python scripts/test_inference.py --video examples/hello.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.model import SignRecognizer  # noqa: E402
from app.preprocessing import VideoError, check_extension, preprocess_video  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="video of one ASL sign")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="print a JSON result")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if not args.video.is_file():
        print(f"error: {args.video} not found", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    rec = SignRecognizer(settings.model_dir)
    import mediapipe  # noqa: F401  (one-off ~2 s import; the server does it at startup)
    load_ms = (time.perf_counter() - t0) * 1000

    try:
        check_extension(args.video.name, settings)
        prepared, prep_ms = preprocess_video(
            args.video,
            settings,
            feature_set=rec.feature_set,
            model_fps=rec.fps,
            min_frames=rec.min_frames,
            max_frames=rec.max_frames,
        )
    except VideoError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1
    pred = rec.predict(prepared.features, k=args.top_k)
    confident = pred.confident and not prepared.partial_view

    if args.json:
        print(json.dumps({
            "prediction": pred.top[0].label,
            "confidence": pred.top[0].confidence,
            "confident": confident,
            "top_predictions": [{"label": g.label, "confidence": g.confidence} for g in pred.top],
            "timing_ms": {"model_load": load_ms, "preprocessing": prep_ms, "model": pred.inference_ms},
            "video": prepared.stats,
            "warnings": prepared.warnings,
        }, indent=2))
        return 0

    print(f"Model:       {rec.version} ({len(rec.signs)} signs), loaded in {load_ms:.0f} ms")
    print(f"Video:       {args.video}  {prepared.stats}")
    print(f"Prediction:  {pred.top[0].label}  (confidence {pred.top[0].confidence:.3f}"
          f"{'' if confident else ', NOT SURE'})")
    print(f"Top-{args.top_k}:")
    for i, g in enumerate(pred.top, 1):
        print(f"  {i}. {g.label:<20} {g.confidence:.3f}")
    print(f"Latency:     preprocessing {prep_ms:.0f} ms + model {pred.inference_ms:.1f} ms"
          f" = {prep_ms + pred.inference_ms:.0f} ms")
    for w in prepared.warnings:
        print(f"Warning:     {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
