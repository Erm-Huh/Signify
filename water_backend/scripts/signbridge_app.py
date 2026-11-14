"""SignBridge unified live app: one webcam, word + alphabet recognition, one result.

    python scripts/signbridge_app.py                 # camera 0, AUTO mode
    python scripts/signbridge_app.py --mode SPELL    # start in fingerspelling mode

Both recognisers run on the same camera frames; a conservative router
(app/services/recognition_router.py) picks what to show: WORD, LETTER, AMBIGUOUS,
NOT SURE or NO HANDS. Word recognition is sliding-window classification of isolated
signs (not segmentation or sentences); letters are per-frame with upstream smoothing.

Keys
  Q / Esc    quit                      TAB        mode AUTO -> WORD -> SPELL
  SPACE      WORD/AUTO: capture 2 s for the word model   |   SPELL: insert a space
  ENTER      confirm the primary result into the text
  W / L      confirm the current top WORD / top LETTER (manual fallback, e.g. AMBIGUOUS)
  R          repeat last letter (double letters)          D   toggle Other Possibilities
  BACKSPACE  delete last committed item                   C   clear text
  S          speak the committed text (macOS `say`, non-blocking)

Testing without a person:  --source examples/hello.mp4 --duration 15
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.live import LiveConfig  # noqa: E402
from app.model import SignRecognizer  # noqa: E402
from app.models.alphabet_recognizer import AlphabetModel  # noqa: E402
from app.services.recognition_router import Kind, Mode, RouterConfig, auto_commit  # noqa: E402
from app.services.speech import Speaker  # noqa: E402
from app.services.text_buffer import TextBuffer  # noqa: E402
from app.services.unified_session import SessionConfig, Snapshot, UnifiedSession  # noqa: E402

WINDOW = "SignBridge - Live ASL Recognition"
CAM_W, CAM_H = 960, 540
BG, PANEL, LINE = (18, 18, 18), (32, 32, 32), (70, 70, 70)
WHITE, GREY, DIM = (240, 240, 240), (160, 160, 160), (100, 100, 100)
GREEN, AMBER, RED, BLUE = (90, 200, 90), (40, 180, 250), (70, 70, 230), (230, 160, 60)
KIND_COLOR = {Kind.WORD: GREEN, Kind.LETTER: GREEN, Kind.AMBIGUOUS: AMBER, Kind.NOT_SURE: AMBER, Kind.NO_HANDS: GREY}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put(img, s, x, y, scale=0.55, color=WHITE, thick=1) -> None:
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def centered(img, s, y, scale, color, thick) -> None:
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    put(img, s, max(8, (img.shape[1] - tw) // 2), y, scale, color, thick)


def fmt_age(age: float | None) -> str:
    return "-" if age is None else f"{age:.1f}s ago"


def render(cam: np.ndarray, snap: Snapshot, mode: Mode, text: str, details: bool,
           fps: float, message: str, skeleton: np.ndarray | None) -> np.ndarray:
    # letterbox into a fixed 960x540 area so portrait sources don't make the window huge
    scale = min(CAM_W / cam.shape[1], CAM_H / cam.shape[0])
    resized = cv2.resize(cam, (int(cam.shape[1] * scale), int(cam.shape[0] * scale)))
    cam = np.full((CAM_H, CAM_W, 3), BG, np.uint8)
    y0, x0 = (CAM_H - resized.shape[0]) // 2, (CAM_W - resized.shape[1]) // 2
    cam[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    W = CAM_W
    d = snap.decision

    header = np.full((54, W, 3), BG, np.uint8)
    put(header, "SIGNBRIDGE", 14, 34, 0.9, WHITE, 2)
    put(header, "Live ASL Recognition", 210, 33, 0.6, GREY)
    badge = f"MODE: {mode.value}"
    cv2.rectangle(header, (W - 330, 12), (W - 180, 42), BLUE, 2)
    put(header, badge, W - 320, 33, 0.55, BLUE, 1)
    put(header, f"Camera {fps:4.1f} fps", W - 165, 33, 0.5, GREY)

    if skeleton is not None:  # what the alphabet model sees (upstream's "AI view")
        sk = cv2.resize(cv2.cvtColor(skeleton, cv2.COLOR_RGB2BGR), (110, 110), interpolation=cv2.INTER_NEAREST)
        cam[10:120, W - 120:W - 10] = sk
        cv2.rectangle(cam, (W - 121, 9), (W - 9, 121), LINE, 1)
    if snap.capture_state == "recording":
        cv2.rectangle(cam, (0, 0), (int(W * snap.capture_progress), 8), RED, -1)
        put(cam, "CAPTURING WORD... sign now", 12, 34, 0.7, RED, 2)
    elif snap.capture_state == "processing":
        put(cam, "Classifying capture...", 12, 34, 0.7, AMBER, 2)

    # --- primary result ---------------------------------------------------------------
    main = np.full((150, W, 3), PANEL, np.uint8)
    put(main, "RECOGNIZED SIGN", 14, 24, 0.5, GREY)
    color = KIND_COLOR[d.kind]
    if d.kind in (Kind.WORD, Kind.LETTER):
        big = (d.label or "").upper()
        centered(main, big, 82, 1.9 if len(big) < 12 else 1.2, color if d.stable else AMBER, 3)
        status = "Confident (stable)" if d.stable else "Tentative - holding..."
        info = f"Type: {d.kind.value}    Status: {status}"
    elif d.kind is Kind.AMBIGUOUS:
        alts = "   or   ".join(f"{k.value} '{lab.upper()}'" for k, lab in d.alternatives)
        centered(main, "AMBIGUOUS", 70, 1.4, color, 3)
        info = f"{alts}   -  press W or L to confirm, or TAB"
    elif d.kind is Kind.NOT_SURE:
        centered(main, "NOT SURE", 75, 1.5, color, 3)
        info = {Mode.WORD: "Status: no word above the word model's calibrated threshold",
                Mode.SPELL: "Status: no letter above the alphabet model's threshold",
                Mode.AUTO: "Status: neither model passed its own threshold"}[mode]
    else:
        centered(main, "NO HANDS", 75, 1.3, color, 2)
        info = "Show your hands, chest-up"
    centered(main, info, 112, 0.55, WHITE, 1)
    centered(main, d.reason[:110], 138, 0.42, DIM, 1)

    blocks = [header, cam, main]
    if details:  # --- Other Possibilities -------------------------------------------
        det = np.full((96, W, 3), BG, np.uint8)
        put(det, "OTHER POSSIBILITIES  (D to hide)", 14, 20, 0.45, GREY)
        for row, (name, view, unit) in enumerate(
            (("Words", d.word, "calibrated p"), ("Letters", d.letter, "softmax, uncal."))):
            y = 48 + row * 32
            stale = not view.fresh
            col = DIM if stale else WHITE
            put(det, f"{name}:", 14, y, 0.55, col, 1)
            tops = "   ".join(f"{i + 1}. {lab} {c:.2f}" for i, (lab, c) in enumerate(view.top[:3])) or "-"
            put(det, tops, 100, y, 0.55, col, 1)
            tag = f"[{unit}] {fmt_age(view.age_s)}" + ("  STALE" if stale and view.available else "")
            put(det, tag, W - 250, y, 0.42, AMBER if stale and view.available else GREY)
        blocks.append(det)

    bottom = np.full((70, W, 3), BG, np.uint8)
    cv2.line(bottom, (0, 0), (W, 0), LINE, 1)
    put(bottom, "Text:", 14, 26, 0.6, GREY)
    put(bottom, (text[-70:] or "-") + "_", 80, 26, 0.65, WHITE, 1)
    if message:
        put(bottom, message, W - 330, 26, 0.45, AMBER)
    space_label = "SPACE space" if mode is Mode.SPELL else "SPACE capture"
    put(bottom, f"Q quit | TAB mode | {space_label} | ENTER confirm | W/L pick | R repeat | "
                "D details | BKSP del | C clear | S speak", 14, 56, 0.42, GREY)
    blocks.append(bottom)
    return np.vstack(blocks)


def _pct(v, q) -> float:
    return float(np.percentile(np.asarray(v), q)) if len(v) else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="camera index or a video file (fake camera)")
    p.add_argument("--mode", default="AUTO", choices=[m.value for m in Mode])
    p.add_argument("--duration", type=float, default=0, help="testing: quit after N seconds")
    p.add_argument("--interval", type=float, default=0.5, help="word inference interval (s)")
    p.add_argument("--word-max-age", type=float, default=1.5, help="router: word result staleness (s)")
    p.add_argument("--letter-max-age", type=float, default=0.5, help="router: letter result staleness (s)")
    p.add_argument("--pause-word-in-spell", action="store_true", help="save CPU: no word model in SPELL")
    p.add_argument("--pause-alphabet-in-word", action="store_true", help="save CPU: no letters in WORD")
    p.add_argument("--no-details", action="store_true", help="start with Other Possibilities hidden")
    p.add_argument("--screenshot", type=Path, help="save the last rendered window to this PNG on exit")
    a = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("Loading word model, alphabet model and MediaPipe...", flush=True)
    word_rec = SignRecognizer(settings.model_dir)
    alpha = AlphabetModel(settings.model_dir / "alphabet")
    cfg = SessionConfig(
        live=LiveConfig(interval_s=a.interval),
        router=RouterConfig(word_max_age_s=a.word_max_age, letter_max_age_s=a.letter_max_age),
        run_word_in_spell=not a.pause_word_in_spell,
        run_alphabet_in_word=not a.pause_alphabet_in_word,
    )

    is_file = not a.source.isdigit()
    cap = cv2.VideoCapture(a.source if is_file else int(a.source))  # the ONLY camera owner
    if not cap.isOpened():
        msg = (f"could not open video file {a.source!r}" if is_file else
               f"could not open camera {a.source!r}; on macOS allow camera access for your "
               "terminal app in System Settings > Privacy & Security > Camera")
        print(f"error: {msg}", file=sys.stderr)
        return 1
    if not is_file:
        cap.set(cv2.CAP_PROP_FPS, 30)
    file_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 30.0

    session = UnifiedSession(word_rec, alpha, cfg, mode=Mode(a.mode))
    text = TextBuffer()
    speaker = Speaker()
    details = not a.no_details
    message, message_until = "", 0.0
    frame_times: deque[float] = deque(maxlen=4000)
    commits_log: list[str] = []
    last_canvas = None

    def say(msg: str, now: float, secs: float = 2.5) -> None:
        nonlocal message, message_until
        message, message_until = msg, now + secs

    try:
        session.wait_ready()
        print(f"Ready. Mode {session.mode.value}. Press Q to quit.", flush=True)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        start = next_t = time.perf_counter()
        while True:
            if is_file:
                next_t += 1.0 / file_fps
                time.sleep(max(0.0, next_t - time.perf_counter()))
            ok, frame = cap.read()
            if not ok and is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or frame is None:
                print("error: lost the camera feed", file=sys.stderr)
                break
            now = time.perf_counter()
            frame_times.append(now)
            session.submit_frame(frame, now)  # both pipelines; non-blocking, bounded
            session.tick(now)
            for err in (session.word.error, session.alphabet.error):
                if err:
                    raise RuntimeError(f"worker stopped: {err}")
            snap = session.snapshot(now)

            # automatic commits: only stable, non-ambiguous results (see router.auto_commit)
            events = [auto_commit(snap.decision, snap.word_emitted, None)]
            events += [auto_commit(snap.decision, None, c) for c in snap.letter_commits]
            for ev in filter(None, events):
                kind, label = ev
                added = (text.add_word(label, now) if kind is Kind.WORD else text.add_letter(label, now))
                if added:
                    commits_log.append(f"{kind.value}:{label}")
                    print(f"[commit] {kind.value} {label!r} -> text: {text.text!r}", flush=True)

            recent = [t for t in list(frame_times)[-40:] if now - t <= 1.0]
            preview = frame if is_file else cv2.flip(frame, 1)
            letter = snap.letter
            skeleton = letter.skeleton_rgb if (letter and letter.hand_detected and snap.decision.letter.fresh) else None
            canvas = render(preview, snap, session.mode, text.text, details, len(recent),
                            message if now < message_until else "", skeleton)
            cv2.imshow(WINDOW, canvas)
            last_canvas = canvas

            key = cv2.waitKey(1) & 0xFF
            d = snap.decision
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == 9:  # TAB
                session.set_mode(session.mode.next())
                say(f"mode -> {session.mode.value}", now)
            elif key == ord(" "):
                if session.mode is Mode.SPELL:
                    text.add_space(now)
                    session.alphabet.stream.reset_repeat_guard()
                elif session.start_capture(now):
                    say("capturing 2 s for the word model", now)
                else:
                    say("capture already running", now)
            elif key in (13, 10):  # ENTER: confirm the primary result
                if d.kind is Kind.WORD and d.label:
                    text.add_word(d.label, now, manual=True)
                    session.dismiss_manual()
                    say(f"confirmed word {d.label!r}", now)
                elif d.kind is Kind.LETTER and d.label:
                    text.add_letter(d.label, now, manual=True)
                    say(f"confirmed letter {d.label}", now)
                else:
                    say("nothing to confirm (use W / L)", now)
            elif key in (ord("w"), ord("W")):
                if snap.manual and snap.manual.label:
                    label = snap.manual.label
                else:
                    label = d.word.label if d.word.fresh and d.word.hands else None
                if label:
                    text.add_word(label, now, manual=True)
                    session.dismiss_manual()
                    say(f"confirmed word {label!r}", now)
                else:
                    say("no current word prediction", now)
            elif key in (ord("l"), ord("L")):
                if d.letter.fresh and d.letter.hands and d.letter.label:
                    text.add_letter(d.letter.label, now, manual=True)
                    say(f"confirmed letter {d.letter.label}", now)
                else:
                    say("no current letter prediction", now)
            elif key in (ord("r"), ord("R")):
                if text.repeat_last_letter(now):
                    say("repeated last letter", now)
            elif key in (ord("d"), ord("D")):
                details = not details
            elif key in (8, 127):
                text.delete_last()
                session.alphabet.stream.reset_repeat_guard()
            elif key in (ord("c"), ord("C")):
                text.clear()
                session.alphabet.stream.reset_repeat_guard()
                say("text cleared", now)
            elif key in (ord("s"), ord("S")):
                status = speaker.speak(text.text)
                say({"spoken": "speaking...", "busy": "still speaking", "empty": "no confirmed text",
                     "unavailable": "TTS not available"}[status], now)
            if a.duration and now - start >= a.duration:
                break
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...", flush=True)
        if a.screenshot is not None and last_canvas is not None:
            cv2.imwrite(str(a.screenshot), last_canvas)
        session.close()
        speaker.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        ft = np.array(frame_times)
        gaps = np.diff(ft) * 1000 if len(ft) > 1 else np.array([])
        ws, als = session.word.stats, session.alphabet.stats
        print("--- measured ---")
        print(f"preview: {len(ft)} frames, interval median {_pct(gaps, 50):.0f} ms, p95 {_pct(gaps, 95):.0f} ms")
        print(f"alphabet: {als.processed} of {als.frames_in} frames processed ({als.dropped} dropped), "
              f"rate {als.rate():.1f}/s, Hands+CNN median {_pct(als.frame_ms, 50):.0f} ms p95 {_pct(als.frame_ms, 95):.0f} ms, "
              f"capture->result median {_pct(als.lag_ms, 50):.0f} ms p95 {_pct(als.lag_ms, 95):.0f} ms, "
              f"hand in {als.with_hand} frames")
        print(f"word: Holistic {ws.landmark_frames} of {ws.frames_in} frames ({ws.frames_dropped} dropped), "
              f"median {_pct(ws.landmark_ms, 50):.0f} ms; {session.word.scheduler.submitted} windows, "
              f"{session.word.scheduler.skipped_busy} skipped busy, {ws.stale_windows} stale; "
              f"window end->decision median {_pct(ws.result_lag_ms, 50):.0f} ms p95 {_pct(ws.result_lag_ms, 95):.0f} ms")
        print(f"commits: {commits_log}")
        print(f"final text: {text.text!r}")
        print(f"threads still alive: {session.alive_threads}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
