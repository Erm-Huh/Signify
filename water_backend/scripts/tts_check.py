"""Smoke-test ElevenLabs TTS: one sentence per voice, per model.

Usage:
    .venv/Scripts/python.exe scripts/tts_check.py [OUTPUT_DIR] [--models m1,m2,...]

Writes MP3s to OUTPUT_DIR (default: a fresh temp dir outside the repo) and
prints byte size and latency for each. Sends real requests (uses quota).
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.elevenlabs import (  # noqa: E402
    DEFAULT_MODEL,
    FALLBACK_MODEL,
    ElevenLabsError,
    ElevenLabsTTS,
)

SENTENCE = "Hi, I'm here to check in for my two o'clock appointment."
COMPARE_MODELS = [DEFAULT_MODEL, FALLBACK_MODEL, "eleven_flash_v2_5"]


def main(argv: list[str]) -> int:
    models = list(dict.fromkeys(COMPARE_MODELS))
    args = []
    for a in argv[1:]:
        if a.startswith("--models="):
            models = [m for m in a.split("=", 1)[1].split(",") if m]
        else:
            args.append(a)
    out = Path(args[0]) if args else Path(tempfile.mkdtemp(prefix="signify_tts_"))
    out.mkdir(parents=True, exist_ok=True)

    probe = ElevenLabsTTS()
    if not probe.available:
        print("ELEVENLABS_API_KEY not set (env or water_backend/.env)")
        return 2

    print(f"Output: {out}")
    print(f"Sentence: {SENTENCE!r}\n")
    failures = 0
    for model in models:
        tts = ElevenLabsTTS(model=model)
        print(f"[{model}]")
        for voice, vid in tts.voices.items():
            t0 = time.perf_counter()
            try:
                audio = tts.synthesize(SENTENCE, voice=voice)
            except ElevenLabsError as exc:
                failures += 1
                print(f"  {voice:<7} {vid}  FAILED: {exc}")
                continue
            ms = (time.perf_counter() - t0) * 1000
            path = out / f"{model}_{voice}.mp3"
            path.write_bytes(audio)
            print(f"  {voice:<7} {vid}  {len(audio):>7,} bytes  {ms:6.0f} ms  -> {path.name}")
        print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
