"""Blake's part: the live bridge between the ASL backend and the web UI.

This file is READ-ONLY with respect to ``water_backend/``: it imports the backend package
(recognition pipeline, router, text buffer) and starts its own WebSocket server on 8765.
Nothing in ``water_backend/`` or ``joa_frontend/`` is modified, so a teammate can merge or
change those freely and this keeps working as long as these names stay:

    app.model.SignRecognizer                    (word model, ONNX)
    app.models.alphabet_recognizer.AlphabetModel (fingerspelling CNN)
    app.services.unified_session.UnifiedSession  (one camera -> both models -> router)
    app.services.recognition_router.auto_commit  (commit rules)
    app.services.text_buffer.TextBuffer          (committed sentence)

Run it with the backend's virtualenv (that is where mediapipe/onnxruntime live):

    # camera 0
    cd Signify/water_backend && .venv/bin/python ../Blake_frontend/server/ws_bridge.py

    # no camera: replay a clip as a fake camera
    .venv/bin/python ../Blake_frontend/server/ws_bridge.py --source examples/hello.mp4

    # another port / host (LAN demo)
    .venv/bin/python ../Blake_frontend/server/ws_bridge.py --host 0.0.0.0 --port 8765

Then serve the UI:  cd Blake_frontend && python3 -m http.server 8080  ->  http://localhost:8080

Why the camera lives here and not in the browser
------------------------------------------------
One process must own the webcam and run MediaPipe on the frames the models see. Two openers
(the browser and Python) fight over the device and the preview flickers or dies mid-demo.
So this process grabs the camera, encodes a small mirrored JPEG (~30 KB at 480 px, 14/s) and
ships it next to the JSON state; the page is a pure display.

Protocol (both directions, same as index.html/app.js expect)
-----------------------------------------------------------
server -> client, text, ~15/s::

    {"type":"state","t":12.34,"mode":"AUTO","kind":"WORD","label":"hello",
     "stable":true,"reason":"only the word model is confident","alternatives":[],
     "word":{"label":"hello","score":0.42,"fresh":true,"hands":true,"ok":true,
             "stable":true,"top":[["hello",0.42],["all",0.02]],"age_s":0.31,"note":"stable"},
     "letter":{"label":"B","score":0.99,...,"note":"holding"},
     "text":"HELLO MY","commits":[{"kind":"WORD","label":"hello"}],
     "capture":{"state":"idle","progress":0.0},"fps":28.4,"source":"0"}

``kind`` is WORD / LETTER / AMBIGUOUS / NOT_SURE / NO_HANDS; ``label`` is null for anything
but WORD / LETTER. ``text`` is AUTHORITATIVE — the commit rules (calibrated 0.29 threshold,
N agreeing windows, cooldown, duplicate guard) live in the backend and cannot be reproduced
client-side, because word scores are calibrated and letter scores are not. ``commits`` lists
only what was appended on this tick (for a "+HELLO" flash).

server -> client, binary, ~14/s:  one JPEG preview frame (480 px wide, mirrored).

client -> server, text, one command per message::

    {"type":"cmd","cmd":"mode"}          cycle AUTO -> WORD -> SPELL
    {"type":"cmd","cmd":"capture"}       SPACE in AUTO/WORD: record 2 s for the word model
    {"type":"cmd","cmd":"space"}         SPACE in SPELL: insert a space
    {"type":"cmd","cmd":"confirm"}       ENTER: commit the primary result
    {"type":"cmd","cmd":"word"}          W: commit the current top word (manual)
    {"type":"cmd","cmd":"letter"}        L: commit the current top letter (manual)
    {"type":"cmd","cmd":"repeat"}        R: repeat the last letter
    {"type":"cmd","cmd":"backspace"}     delete the last committed item
    {"type":"cmd","cmd":"clear"}         clear the text
    {"type":"cmd","cmd":"addspace"}      end the current word (smart-split the letters) in any mode
    {"type":"cmd","cmd":"pause"}         privacy: release the webcam (light off), stop reading
    {"type":"cmd","cmd":"resume"}        reopen the webcam

Under-the-hood viewer (the "Under the hood, live" section of joa_frontend/how.html)::

    client -> server:  {"type":"debug","on":true}
    every state message then carries "debug": raw word/letter internals, the 21 hand
    landmarks (0..1 of the mirrored frame) and, ~5/s, "skeleton": a base64 JPEG of the
    image the letter model actually classifies. Off again with {"type":"debug","on":false}
    or on disconnect. The landmarks and model internals are tiny, so they are always sent.

Pause is sticky for the life of this process (a reload does not silently turn the camera
back on). While paused, state messages carry ``"paused": true`` and no frames are sent.

Spacing: a word ends when the signer lowers their hand for ``SIGNBRIDGE_AUTO_SPACE_S``
(default 1.0 s, 0 = off), on ``addspace``/``space``, or when a word sign follows letters. The
letter run is then split into known words (``app.services.smart_spacing``):
"HELLOMYNAME" -> "HELLO MY NAME"; names it can't cover are left whole.

Text to speech (ElevenLabs, key in ``water_backend/.env``, never sent to the page)::

    on connect, server -> client:  {"type":"hello","tts":"elevenlabs"|null}
    client -> server:  {"type":"tts","id":7,"text":"Hello.","voice":"warm","speed":1.0}
    server -> client:  {"type":"tts_audio","id":7,"mime":"audio/mpeg","audio":"<base64>"}
                    or {"type":"tts_error","id":7,"error":"..."}   (the page falls back to the browser voice)

Measured on an M2 (CPU, ``examples/hello.mp4`` as a fake camera): 15 state messages/s,
14 preview frames/s, ~30 KB per frame, first ``WORD hello`` ~1.5 s after connect, and
``'HELLO'`` auto-committed. The camera loop runs below the clip's 30 fps because MediaPipe
Holistic saturates the CPU — the same trade-off the backend README documents.

Single presenter only: every connected tab shares one camera, one mode and one text buffer.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# --- read-only import of the teammate's backend package (no edits to it) -----------------
HERE = Path(__file__).resolve()
BACKEND = HERE.parents[2] / "water_backend"
if not (BACKEND / "app" / "services" / "unified_session.py").exists():
    sys.exit(f"error: backend package not found at {BACKEND}")
sys.path.insert(0, str(BACKEND))

from websockets.asyncio.server import serve  # noqa: E402

from app.config import settings  # noqa: E402
from app.live import LiveConfig  # noqa: E402
from app.model import SignRecognizer  # noqa: E402
from app.models.alphabet_recognizer import AlphabetModel  # noqa: E402
from app.services import smart_spacing  # noqa: E402
from app.services.recognition_router import Kind, Mode, RouterConfig, auto_commit  # noqa: E402
from app.services.text_buffer import TextBuffer  # noqa: E402
from app.services.unified_session import SessionConfig, Snapshot, UnifiedSession  # noqa: E402

log = logging.getLogger("blake.bridge")

PREVIEW_WIDTH = int(os.environ.get("SIGNBRIDGE_PREVIEW_WIDTH", "480"))
JPEG_QUALITY = int(os.environ.get("SIGNBRIDGE_JPEG_QUALITY", "70"))
FRAME_HZ = float(os.environ.get("SIGNBRIDGE_FRAME_HZ", "15"))
STATE_HZ = float(os.environ.get("SIGNBRIDGE_STATE_HZ", "20"))
AUTO_SPACE_S = float(os.environ.get("SIGNBRIDGE_AUTO_SPACE_S", "1.0"))

COMMANDS = frozenset({"mode", "capture", "space", "confirm", "word", "letter", "repeat",
                      "backspace", "clear", "addspace"})
TEXT_ONLY = frozenset({"backspace", "clear", "addspace", "space"})  # honoured while paused


def _end_word(text: TextBuffer, now: float) -> bool:
    """Close the letter run: smart-split it into known words, then add a space."""
    split = text.split_trailing_run(smart_spacing.split_run)
    return text.add_space(now) or split


def _add_word(text: TextBuffer, label: str, now: float, *, manual: bool = False) -> bool:
    text.split_trailing_run(smart_spacing.split_run)  # letters followed by a word sign
    return text.add_word(label, now, manual=manual)


# --------------------------------------------------------------------------- payload


def _view(v) -> dict[str, Any]:
    """One model's output in its own units (the two scores are never compared)."""
    return {
        "available": v.available,
        "fresh": v.fresh,
        "hands": v.hands,
        "ok": v.ok,
        "stable": v.stable,
        "label": v.label,
        "score": round(float(v.score), 4),
        "top": [[lab, round(float(s), 4)] for lab, s in list(v.top)[:3]],
        "age_s": None if v.age_s is None else round(float(v.age_s), 2),
        "note": v.note,
    }


def _payload(snap: Snapshot, text: TextBuffer, t: float, fps: float, source: str,
             commits: list[dict[str, str]]) -> dict[str, Any]:
    d = snap.decision
    return {
        "type": "state",
        "t": round(t, 3),
        "mode": d.mode.value,
        "kind": d.kind.value,
        "label": d.label,
        "stable": d.stable,
        "reason": d.reason,
        "alternatives": [[k.value, lab] for k, lab in d.alternatives],
        "word": _view(d.word),
        "letter": _view(d.letter),
        "text": text.text,
        "commits": commits,
        "capture": {"state": snap.capture_state, "progress": round(snap.capture_progress, 3)},
        "fps": round(fps, 1),
        "source": source,
        "paused": False,
    }


SKELETON_HZ = 5.0


def _debug(snap: Snapshot, fw: int, fh: int, skeleton: str | None) -> dict[str, Any]:
    """Raw model internals for the under-the-hood viewer (no UI shaping)."""
    w, lt = snap.word, snap.letter
    out: dict[str, Any] = {
        "frame": [fw, fh],
        "word": {
            "state": w.state, "label": w.label, "conf": round(float(w.confidence), 4),
            "stable": bool(w.stable), "reason": w.reason,
            "top": [[lab, round(float(sc), 4)] for lab, sc in list(w.top)[:5]],
            "recent": list(w.emitted)[-6:],
        },
        "letter": None,
    }
    if lt is not None:
        pts = None
        if lt.points_px is not None and fw and fh:
            arr = np.asarray(lt.points_px, dtype=np.float32).reshape(-1, 2)
            pts = [[round(float(x) / fw, 4), round(float(y) / fh, 4)] for x, y in arr]
        out["letter"] = {
            "hand": bool(lt.hand_detected), "best": lt.best, "conf": round(float(lt.confidence), 4),
            "static_best": lt.static_best, "motion_override": lt.motion_override,
            "moving": bool(lt.moving), "passes": bool(lt.passes_threshold),
            "stable": bool(lt.stable), "progress": round(float(lt.progress), 3),
            "infer_ms": round(float(lt.infer_ms), 1),
            "top": [[g.label, round(float(g.confidence), 4)] for g in lt.top],
            "points": pts,
        }
    if skeleton:
        out["skeleton"] = skeleton
    return out


def _skeleton_jpeg(snap: Snapshot) -> str | None:
    lt = snap.letter
    if lt is None or lt.skeleton_rgb is None:
        return None
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(lt.skeleton_rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


# --------------------------------------------------------------------------- hub


class LiveHub:
    """Owns the camera and one ``UnifiedSession`` on a private thread.

    The camera loop publishes ``(seq, state)`` and ``(seq, jpeg)`` pairs; WebSocket handlers
    only read the latest pair, so a slow or closed tab can never stall the models (no
    per-client queues: dropping old frames is the intended behaviour).
    """

    def __init__(self, word_rec: SignRecognizer, alphabet_model: AlphabetModel,
                 source: str) -> None:
        self.source = source
        self.word_rec, self.alphabet_model = word_rec, alphabet_model
        self.error: str | None = None
        self._state: dict[str, Any] = {}
        self._state_seq = 0
        self._frame: bytes | None = None
        self._frame_seq = 0
        self._lock = threading.Lock()
        self._cmds: queue.Queue[str] = queue.Queue(maxsize=32)
        self._clients = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()   # privacy pause; sticky across reconnects
        self.tts = None                    # ElevenLabsTTS when a key is configured
        self._debug_n = 0                  # tabs that asked for the skeleton image

    def set_debug(self, delta: int) -> None:
        with self._lock:
            self._debug_n = max(0, self._debug_n + delta)

    # -- lifecycle -----------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def failed(self) -> bool:
        return self.error is not None and not self.running

    def acquire(self) -> None:
        """Called by each connection; starts the camera on the first one.

        Blocking by design (it may join an exiting loop thread) — call it through
        ``asyncio.to_thread`` from the event loop.
        """
        with self._lock:
            self._clients += 1
            t = self._thread
            if t is not None and t.is_alive() and not self._stop.is_set():
                return                      # already streaming to another tab
            self._stop.set()                # cancel any shutdown that is in flight
        if t is not None and t.is_alive():
            t.join(timeout=5.0)
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="live-hub", daemon=True)
        self._thread.start()

    def release(self) -> None:
        """Called on disconnect; stops the camera when the last client leaves."""
        with self._lock:
            self._clients = max(0, self._clients - 1)
            stop = self._clients == 0
        if stop:
            self.close()

    def close(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=10.0)
        self._thread = None

    def command(self, cmd: str) -> bool:
        if cmd == "pause":                 # handled at once, not queued behind a frame
            self._paused.set()
            with self._lock:
                self._frame = None
            log.info("pause requested")
            return True
        if cmd == "resume":
            self._paused.clear()
            log.info("resume requested")
            return True
        if cmd not in COMMANDS:
            return False
        try:
            self._cmds.put_nowait(cmd)
            return True
        except queue.Full:
            return False

    def publish(self) -> tuple[int, dict[str, Any], int, bytes | None]:
        with self._lock:
            return self._state_seq, self._state, self._frame_seq, self._frame

    # -- camera loop (its own thread) -----------------------------------------------------

    def _open(self) -> tuple[cv2.VideoCapture | None, float]:
        source = self.source
        is_file = not source.isdigit()
        cap = cv2.VideoCapture(source if is_file else int(source))
        if not cap.isOpened():
            self.error = (f"could not open {source!r}" if is_file else
                          f"could not open camera {source!r}; check it isn't in use by another "
                          "app and that camera access is allowed for this terminal")
            log.error(self.error)
            cap.release()
            return None, 30.0
        if not is_file:
            cap.set(cv2.CAP_PROP_FPS, 30)
        return cap, (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 30.0

    def _publish_paused(self, text: TextBuffer, now: float) -> None:
        state = dict(self._state) if self._state else {"type": "state", "mode": "AUTO",
                                                        "word": {}, "letter": {}, "fps": 0}
        state.update(t=round(now, 3), kind="NO_HANDS", label=None, stable=False,
                     reason="camera paused", alternatives=[], text=text.text, commits=[],
                     capture={"state": "idle", "progress": 0.0}, source=self.source,
                     paused=True, fps=0)
        with self._lock:
            self._frame = None
            self._state, self._state_seq = state, self._state_seq + 1

    def _run(self) -> None:
        source = self.source
        is_file = not source.isdigit()
        cap: cv2.VideoCapture | None = None
        file_fps = 30.0
        if not self._paused.is_set():          # a sticky pause never opens the camera
            cap, file_fps = self._open()
            if cap is None:
                return

        session: UnifiedSession | None = None
        try:
            sess = UnifiedSession(
                self.word_rec, self.alphabet_model,
                SessionConfig(live=LiveConfig(), router=RouterConfig()),
            )
            session = sess
            sess.wait_ready()
            log.info("ready: source=%r mode=%s — open the UI and start signing",
                     source, sess.mode.value)
            text = TextBuffer()
            frame_times: deque[float] = deque(maxlen=4000)
            t_text = t_frame = t_skel = 0.0
            next_file_frame = time.perf_counter()
            last_hand_t = time.perf_counter()

            while not self._stop.is_set():
                if self._paused.is_set():
                    if cap is not None:
                        cap.release()
                        cap = None
                        log.info("camera paused: webcam released")
                    now = time.perf_counter()
                    self._drain_text_only(session, text, now)
                    self._publish_paused(text, now)
                    time.sleep(0.2)
                    continue
                if cap is None:
                    cap, file_fps = self._open()
                    if cap is None:
                        break
                    log.info("camera resumed")
                    frame_times.clear()
                    next_file_frame = last_hand_t = time.perf_counter()
                if is_file:
                    next_file_frame += 1.0 / file_fps
                    time.sleep(max(0.0, next_file_frame - time.perf_counter()))
                ok, frame = cap.read()
                if not ok and is_file:  # fake camera: loop the clip
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                if not ok or frame is None:
                    self.error = "lost the video source"
                    log.error(self.error)
                    break

                now = time.perf_counter()
                frame_times.append(now)
                fh, fw = frame.shape[:2]
                session.submit_frame(frame, now)   # both models; non-blocking, bounded
                session.tick(now)
                for err in (session.word.error, session.alphabet.error):
                    if err:
                        raise RuntimeError(f"worker stopped: {err}")
                snap = session.snapshot(now)
                commits = self._apply_commands(session, text, snap, now)

                # automatic commits: only stable, non-ambiguous results (router's rule)
                events = [auto_commit(snap.decision, snap.word_emitted, None)]
                events += [auto_commit(snap.decision, None, c) for c in snap.letter_commits]
                for ev in filter(None, events):
                    kind, label = ev
                    added = (_add_word(text, label, now) if kind is Kind.WORD
                             else text.add_letter(label, now))
                    if added:
                        commits.append({"kind": kind.value, "label": label})
                        log.info("commit %s %r -> %r", kind.value, label, text.text)

                # hand lowered after fingerspelling = end of word (auto space + smart split)
                if snap.decision.letter.hands:
                    last_hand_t = now
                elif (AUTO_SPACE_S > 0 and now - last_hand_t >= AUTO_SPACE_S
                      and text.trailing_letters()):
                    if _end_word(text, now):
                        session.alphabet.stream.reset_repeat_guard()
                        log.info("auto space -> %r", text.text)

                recent = [x for x in list(frame_times)[-40:] if now - x <= 1.0]
                if now - t_text >= 1.0 / STATE_HZ:
                    t_text = now
                    state = _payload(snap, text, now, len(recent), source, commits)
                    skel = None
                    if self._debug_n > 0 and now - t_skel >= 1.0 / SKELETON_HZ:
                        t_skel = now
                        skel = _skeleton_jpeg(snap)
                    state["debug"] = _debug(snap, fw, fh, skel)
                    with self._lock:
                        self._state, self._state_seq = state, self._state_seq + 1

                if now - t_frame >= 1.0 / FRAME_HZ:
                    t_frame = now
                    jpeg = self._encode(frame)
                    if jpeg is not None:
                        with self._lock:
                            self._frame, self._frame_seq = jpeg, self._frame_seq + 1
        except BaseException as exc:  # never leave a dead camera loop silently
            log.exception("live hub crashed")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if session is not None:
                session.close()
            if cap is not None:
                cap.release()
            log.info("camera released")

    @staticmethod
    def _encode(frame: np.ndarray) -> bytes | None:
        """Mirrored (selfie) preview, downscaled to a fixed width, JPEG."""
        h, w = frame.shape[:2]
        if w > PREVIEW_WIDTH:
            h = max(2, int(h * PREVIEW_WIDTH / w))
            frame = cv2.resize(frame, (PREVIEW_WIDTH, h), interpolation=cv2.INTER_AREA)
        frame = cv2.flip(frame, 1)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def _drain_text_only(self, session: UnifiedSession, text: TextBuffer, now: float) -> None:
        """While paused: editing the sentence still works; recognition commands are dropped."""
        while True:
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                return
            if cmd not in TEXT_ONLY:
                continue
            if cmd in ("addspace", "space"):
                _end_word(text, now)
            elif cmd == "backspace":
                text.delete_last()
            elif cmd == "clear":
                text.clear()
            session.alphabet.stream.reset_repeat_guard()

    def _apply_commands(self, session: UnifiedSession, text: TextBuffer, snap: Snapshot,
                        now: float) -> list[dict[str, str]]:
        """Drain client commands on the loop thread, mirroring the desktop key handler."""
        out: list[dict[str, str]] = []
        d = snap.decision
        while True:
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                return out
            if cmd == "mode":
                session.set_mode(session.mode.next())
            elif cmd == "capture":
                session.start_capture(now)
            elif cmd == "space":
                if session.mode is Mode.SPELL or text.trailing_letters():
                    _end_word(text, now)
                    session.alphabet.stream.reset_repeat_guard()
                else:
                    session.start_capture(now)
            elif cmd == "addspace":
                _end_word(text, now)
                session.alphabet.stream.reset_repeat_guard()
            elif cmd == "confirm":
                if d.kind is Kind.WORD and d.label:
                    _add_word(text, d.label, now, manual=True)
                    session.dismiss_manual()
                    out.append({"kind": "WORD", "label": d.label})
                elif d.kind is Kind.LETTER and d.label:
                    text.add_letter(d.label, now, manual=True)
                    out.append({"kind": "LETTER", "label": d.label})
            elif cmd == "word":
                label = snap.manual.label if snap.manual and snap.manual.label else (
                    d.word.label if d.word.fresh and d.word.hands else None)
                if label:
                    _add_word(text, label, now, manual=True)
                    session.dismiss_manual()
                    out.append({"kind": "WORD", "label": label})
            elif cmd == "letter":
                if d.letter.fresh and d.letter.hands and d.letter.label:
                    text.add_letter(d.letter.label, now, manual=True)
                    out.append({"kind": "LETTER", "label": d.letter.label})
            elif cmd == "repeat":
                if text.repeat_last_letter(now):
                    out.append({"kind": "LETTER", "label": "*"})
            elif cmd == "backspace":
                text.delete_last()
                session.alphabet.stream.reset_repeat_guard()
            elif cmd == "clear":
                text.clear()
                session.alphabet.stream.reset_repeat_guard()


# --------------------------------------------------------------------------- server


async def handler(ws, hub: LiveHub) -> None:
    """One browser tab: pushes state + preview, reads commands."""
    await asyncio.to_thread(hub.acquire)
    if hub.failed or hub.error:
        await ws.send(json.dumps({"type": "error", "error": hub.error}))
        await asyncio.to_thread(hub.release)
        return
    await ws.send(json.dumps({"type": "hello", "tts": "elevenlabs" if hub.tts else None}))
    tts_lock = asyncio.Lock()
    debug_on = [False]

    async def speak(msg: dict[str, Any]) -> None:
        """ElevenLabs runs in a worker thread; one request at a time per tab."""
        rid = msg.get("id")
        text = str(msg.get("text") or "").strip()[:500]
        try:
            speed = float(msg.get("speed") or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        voice = msg.get("voice") if msg.get("voice") in ("warm", "bright", "deep") else "warm"
        try:
            if hub.tts is None:
                raise RuntimeError("ElevenLabs is not configured on this laptop")
            if not text:
                raise RuntimeError("nothing to say")
            async with tts_lock:
                audio = await asyncio.to_thread(hub.tts.synthesize, text, voice, speed)
            await ws.send(json.dumps({"type": "tts_audio", "id": rid, "mime": "audio/mpeg",
                                      "audio": base64.b64encode(audio).decode("ascii")}))
        except Exception as exc:  # noqa: BLE001  (the page falls back to the browser voice)
            log.warning("tts failed: %s", exc)
            try:
                await ws.send(json.dumps({"type": "tts_error", "id": rid, "error": str(exc)[:200]}))
            except Exception:  # noqa: BLE001
                pass

    async def receive() -> None:
        async for raw in ws:
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "tts":
                asyncio.create_task(speak(msg))
            elif msg.get("type") == "debug":
                on = bool(msg.get("on"))
                if on != debug_on[0]:
                    debug_on[0] = on
                    hub.set_debug(1 if on else -1)
            elif msg.get("type") == "cmd" and isinstance(msg.get("cmd"), str):
                if not hub.command(msg["cmd"]):
                    await ws.send(json.dumps({"type": "warn",
                                              "error": f"unknown command {msg['cmd']!r}"}))

    recv = asyncio.create_task(receive())
    last_state = last_frame = -1
    try:
        while not recv.done():
            seq_s, state, seq_f, frame = hub.publish()
            if seq_s != last_state and state:
                await ws.send(json.dumps(state))
                last_state = seq_s
            if seq_f != last_frame and frame is not None:
                await ws.send(frame)        # binary: JPEG
                last_frame = seq_f
            if hub.failed:
                await ws.send(json.dumps({"type": "error", "error": hub.error}))
                break
            await asyncio.sleep(1.0 / 30.0)
    except Exception:  # normal disconnect (ConnectionClosed) included
        pass
    finally:
        recv.cancel()
        if debug_on[0]:
            hub.set_debug(-1)
        await asyncio.to_thread(hub.release)


async def run(args: argparse.Namespace) -> None:
    log.info("loading word model + alphabet model + MediaPipe ...")
    word_rec = SignRecognizer(settings.model_dir)
    alpha = AlphabetModel(settings.model_dir / "alphabet")
    hub = LiveHub(word_rec, alpha, args.source)
    try:  # word-model labels feed the smart-spacing vocabulary
        smart_spacing.add_words(s.get("label", "") for s in getattr(word_rec, "signs", []))
    except Exception:  # noqa: BLE001
        log.warning("could not add word labels to the spacing vocabulary")
    try:
        from app.services.elevenlabs import ElevenLabsTTS
        tts = ElevenLabsTTS()
        hub.tts = tts if tts.available else None
    except Exception as exc:  # noqa: BLE001
        log.warning("ElevenLabs unavailable: %s", exc)
    log.info("text to speech: %s", "ElevenLabs" if hub.tts else
             "browser voice (set ELEVENLABS_API_KEY in water_backend/.env for ElevenLabs)")

    async def handle(ws) -> None:
        await handler(ws, hub)

    async with serve(handle, args.host, args.port, max_size=None):
        log.info("listening on ws://%s:%d/ (source=%r)", args.host, args.port, args.source)
        log.info("now serve the UI:  cd Blake_frontend && python3 -m http.server 8080")
        try:
            await asyncio.Future()          # run until Ctrl+C
        finally:
            await asyncio.to_thread(hub.close)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=os.environ.get("SIGNBRIDGE_LIVE_SOURCE", "0"),
                   help="camera index or a video file used as a fake camera")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 for a LAN demo")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run(a))
    except KeyboardInterrupt:
        print("bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
