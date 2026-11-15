"""ElevenLabs text-to-speech client (stdlib only).

Speaks a Deaf patient's sentences aloud to clinic staff. Optimised for
naturalness on short sentences; latency is secondary but still reasonable
(~1 s for a 10-word sentence on eleven_v4).

The API key is read from ``ELEVENLABS_API_KEY`` or ``water_backend/.env``.
It is never logged, and never included in exception messages.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://api.elevenlabs.io/v1"
OUTPUT_FORMAT = "mp3_44100_128"
MAX_TEXT_CHARS = 500

# Newest flagship model on this account; verified working through the
# standard /v1/text-to-speech endpoint. Falls back to multilingual_v2 (the
# long-proven natural model) if the primary model is rejected.
DEFAULT_MODEL = "eleven_v4"
FALLBACK_MODEL = "eleven_multilingual_v2"

# Current ElevenLabs default (premade) voices. The account is on a free plan,
# where legacy/library voices (e.g. "Rachel") return 402, so only current
# premade voices are used. All three verified to synthesize on this key.
DEFAULT_VOICES: dict[str, str] = {
    "warm": "EXAVITQu4vr4xnSDxMaL",    # Sarah - calm, reassuring adult female
    "bright": "cgSgspJ2msm6clMCkdW9",  # Jessica - younger, clear and bright
    "deep": "nPczCjzI2devNBz1zQrb",    # Brian - deep, steady male
}

VOICE_SETTINGS = {
    "stability": 0.5,          # 0.5 = natural variation without drifting
    "similarity_boost": 0.75,
    "style": 0.0,              # style exaggeration adds latency and artifacts
    "use_speaker_boost": True,
}

SPEED_MIN, SPEED_MAX = 0.7, 1.2

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class ElevenLabsError(RuntimeError):
    """Short, user-safe error. Never contains the API key."""


def _read_env_file(path: Path, name: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if key != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # strip trailing inline comment on unquoted values
            value = value.split(" #", 1)[0].strip()
        return value or None
    return None


def load_api_key() -> str | None:
    """Return the ElevenLabs key from the environment, else water_backend/.env."""
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if key:
        return key
    return _read_env_file(_ENV_PATH, "ELEVENLABS_API_KEY")


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or _read_env_file(_ENV_PATH, name)


class ElevenLabsTTS:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        self._api_key = api_key or load_api_key()
        self.model = model or _env("ELEVENLABS_MODEL") or DEFAULT_MODEL
        self.timeout_s = timeout_s
        self.voices: dict[str, str] = {
            name: _env(f"ELEVENLABS_VOICE_{name.upper()}") or vid
            for name, vid in DEFAULT_VOICES.items()
        }

    def __repr__(self) -> str:  # never expose the key
        return f"ElevenLabsTTS(model={self.model!r}, available={self.available})"

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def synthesize(self, text: str, voice: str = "warm", speed: float = 1.0) -> bytes:
        """Return MP3 bytes for ``text``. Raises ElevenLabsError on failure."""
        if not self._api_key:
            raise ElevenLabsError("ElevenLabs API key is not configured")
        text = (text or "").strip()
        if not text:
            raise ElevenLabsError("Nothing to speak")
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]

        voice_id = self.voices.get(voice) or self.voices["warm"]
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            speed = 1.0
        speed = min(SPEED_MAX, max(SPEED_MIN, speed))

        try:
            return self._post(text, voice_id, self.model, speed)
        except _HTTPFailure as exc:
            # Primary model rejected (bad request / plan) -> retry once on the
            # proven model. Auth and rate-limit errors are not retried.
            if exc.status in (400, 402, 403, 422) and self.model != FALLBACK_MODEL:
                try:
                    return self._post(text, voice_id, FALLBACK_MODEL, speed)
                except _HTTPFailure as exc2:
                    raise ElevenLabsError(_message_for(exc2.status, exc2.body)) from None
            raise ElevenLabsError(_message_for(exc.status, exc.body)) from None

    # -- internals ---------------------------------------------------------

    def _post(self, text: str, voice_id: str, model: str, speed: float) -> bytes:
        url = (
            f"{API_BASE}/text-to-speech/{urllib.parse.quote(voice_id, safe='')}"
            f"?output_format={OUTPUT_FORMAT}"
        )
        payload = {
            "text": text,
            "model_id": model,
            "voice_settings": {**VOICE_SETTINGS, "speed": speed},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "xi-api-key": self._api_key or "",
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = resp.read()
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", "replace")[:2000]
            except Exception:
                body = ""
            raise _HTTPFailure(exc.code, body) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ElevenLabsError("Could not reach ElevenLabs") from None
        if not data:
            raise ElevenLabsError("ElevenLabs returned no audio")
        return data


class _HTTPFailure(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(status)
        self.status = status
        self.body = body


def _message_for(status: int, body: str) -> str:
    lowered = (body or "").lower()
    if status == 429 or "quota" in lowered or "rate limit" in lowered:
        return "ElevenLabs quota or rate limit reached"
    if status == 401:
        if "missing_permissions" in lowered or "permission" in lowered:
            return "ElevenLabs API key lacks text-to-speech permission"
        return "ElevenLabs rejected the API key"
    if status == 402:
        return "ElevenLabs plan does not allow this voice or model"
    if status == 404:
        return "ElevenLabs voice not found"
    if status in (400, 422):
        return "ElevenLabs rejected the request"
    if status >= 500:
        return "ElevenLabs is temporarily unavailable"
    return f"ElevenLabs request failed (HTTP {status})"
