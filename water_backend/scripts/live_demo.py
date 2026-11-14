"""Continuous live mode: no start/stop. Classifies a sliding window of the webcam stream.

    python scripts/live_demo.py                  # camera 0, 2.0 s windows every 0.5 s

Sign one word at a time, chest-up, then drop your hands. The big label is the smoothed
current guess ("Not sure" below the calibrated threshold); the strip underneath lists
signs accepted after smoothing + cooldown. This is sliding-window classification of
isolated signs, not sign segmentation or sentence recognition (see app/live.py).

Keys: Q / Esc quit, C clear the accepted-signs strip.

Testing without a person:
    python scripts/live_demo.py --source examples/hello.mp4 --duration 12
(a video file is played at its own frame rate, looped, as a fake camera)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.live import LiveConfig, LiveRecognizer, with_overrides  # noqa: E402
from app.model import SignRecognizer  # noqa: E402

WINDOW = "SignBridge LIVE (Q = quit, C = clear)"
PREVIEW_MAX_SIDE = 960
GREEN, YELLOW, WHITE, GREY, RED = (80, 200, 80), (0, 200, 255), (255, 255, 255), (170, 170, 170), (60, 60, 230)


def _text(img, s, org, color=WHITE, scale=0.6, thick=1) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _pct(values, q) -> float:
    return float(np.percentile(np.asarray(values), q)) if len(values) else float("nan")


def draw(view: np.ndarray, live: LiveRecognizer, preview_fps: float) -> None:
    d = live.decision
    h, w = view.shape[:2]
    overlay = view.copy()
    cv2.rectangle(overlay, (0, h - 140), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, view, 0.4, 0, view)
    y = h - 105
    if d.state == "sign":
        _text(view, f"{d.label}  ({d.confidence:.0%})", (12, y), GREEN, 1.0, 2)
    elif d.state == "not_sure":
        why = "partly out of frame" if d.reason == "partial_view" else f"best {d.top[0][0]} {d.confidence:.0%}"
        _text(view, f"Not sure  ({why})", (12, y), YELLOW, 0.85, 2)
    elif d.state == "idle":
        _text(view, "No sign (hands not visible)", (12, y), GREY, 0.8, 2)
    else:
        _text(view, "Warming up...", (12, y), GREY, 0.8, 2)
    if d.top:
        tops = "   ".join(f"{i + 1}. {lab} {c:.2f}" for i, (lab, c) in enumerate(d.top))
        _text(view, tops, (12, y + 30), WHITE, 0.5)
    _text(view, "Accepted: " + (" > ".join(d.emitted[-6:]) or "-"), (12, y + 60), WHITE, 0.55)
    s = live.stats
    lm_fps = 1000 / np.median(s.landmark_ms) if s.landmark_ms else 0.0
    _text(view, f"preview {preview_fps:.0f} fps | landmarks {lm_fps:.0f} fps | "
                f"result lag {_pct(s.result_lag_ms, 50):.0f} ms | skipped {live.scheduler.skipped_busy}",
          (12, h - 10), GREY, 0.45)
    _text(view, "LIVE  sliding window", (12, 28), RED, 0.7, 2)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="camera index or video file (fake camera)")
    p.add_argument("--window", type=float, help="window length in s (default 2.0, max 2.1)")
    p.add_argument("--interval", type=float, help="seconds between inferences (default 0.5)")
    p.add_argument("--smoothing", type=int, help="windows averaged (default 3)")
    p.add_argument("--stable", type=int, help="consecutive confident windows to accept (default 2)")
    p.add_argument("--cooldown", type=float, help="seconds between accepted signs (default 1.5)")
    p.add_argument("--max-side", type=int, help="downscale frames for MediaPipe (default 640)")
    p.add_argument("--complexity", type=int, choices=(0, 1, 2), help="Holistic model complexity")
    p.add_argument("--duration", type=float, default=0, help="testing: quit after N seconds")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = with_overrides(
        LiveConfig(), window_s=a.window, interval_s=a.interval, smoothing_windows=a.smoothing,
        min_stable_windows=a.stable, cooldown_s=a.cooldown, max_frame_side=a.max_side,
        holistic_complexity=a.complexity,
    )
    print("Loading model and MediaPipe...", flush=True)
    rec = SignRecognizer(settings.model_dir)
    try:
        cfg.validate(rec)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    is_file = not a.source.isdigit()
    cap = cv2.VideoCapture(a.source if is_file else int(a.source))
    if not cap.isOpened():
        msg = (f"could not open video file {a.source!r}" if is_file else
               f"could not open camera {a.source!r}. On macOS allow camera access for your terminal "
               "app in System Settings > Privacy & Security > Camera")
        print(f"error: {msg}", file=sys.stderr)
        return 1
    if not is_file:
        cap.set(cv2.CAP_PROP_FPS, 30)
    file_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 30.0

    live = LiveRecognizer(rec, cfg)
    frame_times: list[float] = []
    try:
        live.wait_ready()
        print(f"Live. window {cfg.window_s}s every {cfg.interval_s}s, smoothing {cfg.smoothing_windows}, "
              f"stable {cfg.min_stable_windows}, cooldown {cfg.cooldown_s}s. Q quits.", flush=True)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        start = next_t = time.perf_counter()
        last_emitted: list[str] = []
        while True:
            if is_file:  # play files in real time, looped
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
            live.submit_frame(frame, now)  # non-blocking; drops the oldest queued frame
            live.tick(now)  # non-blocking; skipped while a window is still running
            if live.error:
                print(f"error: landmark worker stopped: {live.error}", file=sys.stderr)
                break

            frame_times.append(now)
            recent = [t for t in frame_times[-31:] if now - t <= 1.0]
            view = frame if is_file else cv2.flip(frame, 1)
            scale = PREVIEW_MAX_SIDE / max(view.shape[:2])
            view = cv2.resize(view, None, fx=scale, fy=scale) if scale < 1 else view.copy()
            draw(view, live, len(recent))
            cv2.imshow(WINDOW, view)

            emitted = live.decision.emitted
            if emitted != last_emitted:
                d = live.decision
                if emitted and d.just_emitted:
                    print(f"[accepted] {emitted[-1]}  ({d.confidence:.2f})  "
                          f"top-3: {', '.join(f'{l} {c:.2f}' for l, c in d.top)}", flush=True)
                last_emitted = emitted

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                live.smoother.history.clear()
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
        live.close()
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        s = live.stats
        gaps = np.diff(frame_times) * 1000 if len(frame_times) > 1 else np.array([])
        print("--- measured ---")
        print(f"preview: {len(frame_times)} frames, median {_pct(gaps, 50):.0f} ms, p95 {_pct(gaps, 95):.0f} ms")
        print(f"landmarks: {s.landmark_frames} of {s.frames_in} frames processed "
              f"({s.frames_dropped} dropped before MediaPipe); Holistic median {_pct(s.landmark_ms, 50):.0f} ms, "
              f"p95 {_pct(s.landmark_ms, 95):.0f} ms per frame")
        print(f"inference: {live.scheduler.submitted} windows scheduled, {live.scheduler.skipped_busy} skipped (busy), "
              f"{s.stale_windows} skipped (no new landmarks); "
              f"features+model median {_pct(s.infer_ms, 50):.1f} ms, p95 {_pct(s.infer_ms, 95):.1f} ms")
        print(f"result lag (capture of window's last frame -> decision): median {_pct(s.result_lag_ms, 50):.0f} ms, "
              f"p95 {_pct(s.result_lag_ms, 95):.0f} ms")
        print(f"accepted signs: {live.decision.emitted}")
        print(f"threads still alive: {live.alive_threads}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
