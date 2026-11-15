"""Unit tests for the ElevenLabs TTS client. No network: urlopen is mocked."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from app.services import elevenlabs as el

FAKE_KEY = "test-key-not-real-123"


class _Resp:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    for name in (
        "ELEVENLABS_API_KEY",
        "ELEVENLABS_MODEL",
        "ELEVENLABS_VOICE_WARM",
        "ELEVENLABS_VOICE_BRIGHT",
        "ELEVENLABS_VOICE_DEEP",
    ):
        monkeypatch.delenv(name, raising=False)
    # point .env lookups at an empty temp location by default
    monkeypatch.setattr(el, "_ENV_PATH", tmp_path / "missing.env")


@pytest.fixture
def calls(monkeypatch):
    recorded: list = []

    def fake_urlopen(req, timeout=None):
        recorded.append((req, timeout))
        return _Resp(b"ID3fake-mp3")

    monkeypatch.setattr(el.urllib.request, "urlopen", fake_urlopen)
    return recorded


def _http_error(status: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.elevenlabs.io", status, "err", {}, io.BytesIO(body.encode())
    )


# -- key loading -----------------------------------------------------------

def test_key_from_env(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    assert el.load_api_key() == FAKE_KEY


def test_key_from_dotenv(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nOTHER=1\n\nexport ELEVENLABS_API_KEY=\"quoted-key\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(el, "_ENV_PATH", env)
    assert el.load_api_key() == "quoted-key"


def test_key_dotenv_single_quotes_and_env_wins(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("ELEVENLABS_API_KEY='file-key'\n", encoding="utf-8")
    monkeypatch.setattr(el, "_ENV_PATH", env)
    assert el.load_api_key() == "file-key"
    monkeypatch.setenv("ELEVENLABS_API_KEY", "env-key")
    assert el.load_api_key() == "env-key"


def test_no_key():
    assert el.load_api_key() is None
    tts = el.ElevenLabsTTS()
    assert tts.available is False
    with pytest.raises(el.ElevenLabsError):
        tts.synthesize("hello")


# -- request shape ---------------------------------------------------------

def test_request_shape(calls):
    tts = el.ElevenLabsTTS(api_key=FAKE_KEY, model="eleven_multilingual_v2", timeout_s=7)
    audio = tts.synthesize("Hello there.", voice="deep", speed=1.0)
    assert audio == b"ID3fake-mp3"
    req, timeout = calls[0]
    assert req.get_method() == "POST"
    assert req.full_url == (
        "https://api.elevenlabs.io/v1/text-to-speech/"
        f"{el.DEFAULT_VOICES['deep']}?output_format=mp3_44100_128"
    )
    assert req.get_header("Xi-api-key") == FAKE_KEY
    assert timeout == 7
    body = json.loads(req.data)
    assert body["text"] == "Hello there."
    assert body["model_id"] == "eleven_multilingual_v2"
    vs = body["voice_settings"]
    assert set(vs) == {"stability", "similarity_boost", "style", "use_speaker_boost", "speed"}
    assert vs["speed"] == 1.0


@pytest.mark.parametrize("given,expected", [(3.0, 1.2), (0.1, 0.7), (0.95, 0.95)])
def test_speed_clamped(calls, given, expected):
    el.ElevenLabsTTS(api_key=FAKE_KEY).synthesize("Hi", speed=given)
    assert json.loads(calls[0][0].data)["voice_settings"]["speed"] == expected


def test_default_model_and_env_override(monkeypatch, calls):
    el.ElevenLabsTTS(api_key=FAKE_KEY).synthesize("Hi")
    assert json.loads(calls[0][0].data)["model_id"] == el.DEFAULT_MODEL
    monkeypatch.setenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
    el.ElevenLabsTTS(api_key=FAKE_KEY).synthesize("Hi")
    assert json.loads(calls[1][0].data)["model_id"] == "eleven_flash_v2_5"


def test_unknown_voice_falls_back_to_warm(calls):
    el.ElevenLabsTTS(api_key=FAKE_KEY).synthesize("Hi", voice="nope")
    assert el.DEFAULT_VOICES["warm"] in calls[0][0].full_url


def test_voice_env_override(monkeypatch, calls):
    monkeypatch.setenv("ELEVENLABS_VOICE_BRIGHT", "customVoice123")
    tts = el.ElevenLabsTTS(api_key=FAKE_KEY)
    assert tts.voices["bright"] == "customVoice123"
    assert set(tts.voices) == {"warm", "bright", "deep"}
    tts.synthesize("Hi", voice="bright")
    assert "/text-to-speech/customVoice123?" in calls[0][0].full_url


def test_text_truncated(calls):
    el.ElevenLabsTTS(api_key=FAKE_KEY).synthesize("a" * 900)
    assert len(json.loads(calls[0][0].data)["text"]) == el.MAX_TEXT_CHARS


def test_empty_text_rejected(calls):
    with pytest.raises(el.ElevenLabsError):
        el.ElevenLabsTTS(api_key=FAKE_KEY).synthesize("   ")
    assert calls == []


# -- error mapping ---------------------------------------------------------

def _raising(monkeypatch, *errors):
    seq = list(errors)
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        err = seq.pop(0)
        if isinstance(err, bytes):
            return _Resp(err)
        raise err

    monkeypatch.setattr(el.urllib.request, "urlopen", fake_urlopen)
    return seen


@pytest.mark.parametrize(
    "status,body,expected",
    [
        (401, '{"detail":{"status":"invalid_api_key"}}', "ElevenLabs rejected the API key"),
        (429, "", "ElevenLabs quota or rate limit reached"),
        (401, '{"detail":{"status":"quota_exceeded"}}', "ElevenLabs quota or rate limit reached"),
        (500, "", "ElevenLabs is temporarily unavailable"),
    ],
)
def test_error_mapping(monkeypatch, status, body, expected):
    _raising(monkeypatch, _http_error(status, body))
    with pytest.raises(el.ElevenLabsError) as ei:
        el.ElevenLabsTTS(api_key=FAKE_KEY).synthesize("Hi")
    assert str(ei.value) == expected
    assert FAKE_KEY not in str(ei.value)


def test_network_error(monkeypatch):
    _raising(monkeypatch, urllib.error.URLError("boom"))
    with pytest.raises(el.ElevenLabsError, match="Could not reach ElevenLabs"):
        el.ElevenLabsTTS(api_key=FAKE_KEY).synthesize("Hi")


def test_model_rejection_falls_back_once(monkeypatch):
    seen = _raising(monkeypatch, _http_error(422, "bad model"), b"ID3ok")
    out = el.ElevenLabsTTS(api_key=FAKE_KEY, model="eleven_v4").synthesize("Hi")
    assert out == b"ID3ok"
    assert [json.loads(r.data)["model_id"] for r in seen] == ["eleven_v4", el.FALLBACK_MODEL]


def test_auth_error_not_retried(monkeypatch):
    seen = _raising(monkeypatch, _http_error(401, "invalid"))
    with pytest.raises(el.ElevenLabsError):
        el.ElevenLabsTTS(api_key=FAKE_KEY).synthesize("Hi")
    assert len(seen) == 1


def test_repr_hides_key():
    assert FAKE_KEY not in repr(el.ElevenLabsTTS(api_key=FAKE_KEY))
