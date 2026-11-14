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

TTS is the browser's ``speechSynthesis`` on purpose: no audio over the socket.

Measured on an M2 (CPU, ``examples/hello.mp4`` as a fake camera): 15 state messages/s,
14 preview frames/s, ~30 KB per frame, first ``WORD hello`` ~1.5 s after connect, and
``'HELLO'`` auto-committed. The camera loop runs below the clip's 30 fps because MediaPipe
Holistic saturates the CPU — the same trade-off the backend README documents.

Single presenter only: every connected tab shares one camera, one mode and one text buffer.
"""

from __future__ import annotations

import argparse
import asyncio
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
from app.services.recognition_router import Kind, Mode, RouterConfig, auto_commit  # noqa: E402
from app.services.text_buffer import TextBuffer  # noqa: E402
from app.services.unified_session import SessionConfig, Snapshot, UnifiedSession  # noqa: E402

log = logging.getLogger("blake.bridge")

PREVIEW_WIDTH = int(os.environ.get("SIGNBRIDGE_PREVIEW_WIDTH", "480"))
JPEG_QUALITY = int(os.environ.get("SIGNBRIDGE_JPEG_QUALITY", "70"))
FRAME_HZ = float(os.environ.get("SIGNBRIDGE_FRAME_HZ", "15"))
STATE_HZ = float(os.environ.get("SIGNBRIDGE_STATE_HZ", "20"))

COMMANDS = frozenset({"mode", "capture", "space", "confirm", "word", "letter", "repeat",
                      "backspace", "clear"})


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
    }


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

    def _run(self) -> None:
        source = self.source
        is_file = not source.isdigit()
        cap = cv2.VideoCapture(source if is_file else int(source))
        if not cap.isOpened():
            self.error = (f"could not open {source!r}" if is_file else
                          f"could not open camera {source!r}; allow camera access for your "
                          "terminal in System Settings > Privacy & Security > Camera")
            log.error(self.error)
            return
        if not is_file:
            cap.set(cv2.CAP_PROP_FPS, 30)
        file_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 30.0

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
            t_text = t_frame = 0.0
            next_file_frame = time.perf_counter()

            while not self._stop.is_set():
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
                    added = (text.add_word(label, now) if kind is Kind.WORD
                             else text.add_letter(label, now))
                    if added:
                        commits.append({"kind": kind.value, "label": label})
                        log.info("commit %s %r -> %r", kind.value, label, text.text)

                recent = [x for x in list(frame_times)[-40:] if now - x <= 1.0]
                if now - t_text >= 1.0 / STATE_HZ:
                    t_text = now
                    state = _payload(snap, text, now, len(recent), source, commits)
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
                if session.mode is Mode.SPELL:
                    text.add_space(now)
                    session.alphabet.stream.reset_repeat_guard()
                else:
                    session.start_capture(now)
            elif cmd == "confirm":
                if d.kind is Kind.WORD and d.label:
                    text.add_word(d.label, now, manual=True)
                    session.dismiss_manual()
                    out.append({"kind": "WORD", "label": d.label})
                elif d.kind is Kind.LETTER and d.label:
                    text.add_letter(d.label, now, manual=True)
                    out.append({"kind": "LETTER", "label": d.label})
            elif cmd == "word":
                label = snap.manual.label if snap.manual and snap.manual.label else (
                    d.word.label if d.word.fresh and d.word.hands else None)
                if label:
                    text.add_word(label, now, manual=True)
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

    async def receive() -> None:
        async for raw in ws:
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "cmd" and isinstance(msg.get("cmd"), str):
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
        await asyncio.to_thread(hub.release)


async def run(args: argparse.Namespace) -> None:
    log.info("loading word model + alphabet model + MediaPipe ...")
    word_rec = SignRecognizer(settings.model_dir)
    alpha = AlphabetModel(settings.model_dir / "alphabet")
    hub = LiveHub(word_rec, alpha, args.source)

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
