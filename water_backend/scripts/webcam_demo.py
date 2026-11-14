"""Local webcam demo: capture a 2-second clip, then recognise ONE isolated ASL sign.

This is NOT continuous real-time recognition. Each prediction is "record a short clip,
then run the same pipeline as POST /predict on it", which takes a few seconds on a CPU.

    python scripts/webcam_demo.py              # default camera (index 0)

Keys:  SPACE = record a 2 s clip and predict    Q / Esc = quit

Every clip goes through the backend's own code: ``app.preprocessing.preprocess_video``
(decode -> MediaPipe Holistic -> upstream features) and ``app.model.SignRecognizer.predict``
(ONNX + calibrated softmax). The ``confident`` rule matches /predict.

Test options (no person needed):
    --source examples/hello.mp4   use a video file as a looping fake camera
    --auto-capture 1              press SPACE automatically N times, quit after the results
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.model import SignRecognizer  # noqa: E402
from app.preprocessing import VideoError, preprocess_video  # noqa: E402

WINDOW = "SignBridge webcam demo (SPACE = capture, Q = quit)"
CLIP_SECONDS = 2.0
PREVIEW_MAX_SIDE = 960
GREEN, RED, YELLOW, WHITE, GREY = (80, 200, 80), (60, 60, 230), (0, 200, 255), (255, 255, 255), (170, 170, 170)


@dataclass
class Result:
    ok: bool
    label: str = ""
    confidence: float = 0.0
    confident: bool = False
    top: list[tuple[str, float]] = field(default_factory=list)
    error: str = ""
    warnings: list[str] = field(default_factory=list)
    preprocess_ms: float = 0.0
    model_ms: float = 0.0
    total_ms: float = 0.0


def write_clip(frames: list[np.ndarray], timestamps: list[float], path: Path) -> float:
    """Write captured frames as MP4 at the frame rate actually measured during capture."""
    span = timestamps[-1] - timestamps[0]
    fps = (len(frames) - 1) / span if len(frames) > 1 and span > 0 else 30.0
    fps = float(min(max(fps, 5.0), 120.0))
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer")
    try:
        for f in frames:
            writer.write(f)
    finally:
        writer.release()
    return fps


def run_prediction(
    rec: SignRecognizer, frames: list[np.ndarray], timestamps: list[float], tmp_dir: Path, n: int
) -> Result:
    """Worker-thread job: temp MP4 -> the backend's preprocess_video + predict."""
    start = time.perf_counter()
    path = tmp_dir / f"clip-{n:03d}.mp4"
    try:
        fps = write_clip(frames, timestamps, path)
        logging.info("clip %d: %d frames, %.1f fps measured", n, len(frames), fps)
        prepared, prep_ms = preprocess_video(
            path,
            settings,
            feature_set=rec.feature_set,
            model_fps=rec.fps,
            min_frames=rec.min_frames,
            max_frames=rec.max_frames,
        )
        pred = rec.predict(prepared.features, k=settings.top_k)
        best = pred.top[0]
        return Result(
            ok=True,
            label=best.label,
            confidence=best.confidence,
            confident=pred.confident and not prepared.partial_view,  # same rule as /predict
            top=[(g.label, g.confidence) for g in pred.top],
            warnings=prepared.warnings,
            preprocess_ms=prep_ms,
            model_ms=pred.inference_ms,
            total_ms=(time.perf_counter() - start) * 1000,
        )
    except VideoError as exc:
        return Result(ok=False, error=f"{exc.code}: {exc.message}",
                      total_ms=(time.perf_counter() - start) * 1000)
    except Exception as exc:  # keep the demo alive; show the error on screen
        logging.exception("prediction failed")
        return Result(ok=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        path.unlink(missing_ok=True)


def print_result(r: Result) -> None:
    if not r.ok:
        print(f"[result] error: {r.error}", flush=True)
        return
    head = r.label if r.confident else f"Not sure (best guess: {r.label})"
    print(f"[result] {head}  confidence {r.confidence:.3f}", flush=True)
    for i, (label, conf) in enumerate(r.top, 1):
        print(f"         {i}. {label:<18} {conf:.3f}", flush=True)
    print(f"         latency: preprocessing {r.preprocess_ms:.0f} ms + model {r.model_ms:.1f} ms "
          f"(capture-to-result {r.total_ms:.0f} ms after recording)", flush=True)
    for w in r.warnings:
        print(f"         warning: {w}", flush=True)


# --------------------------------------------------------------------------- drawing


def _text(img: np.ndarray, s: str, org: tuple[int, int], color=WHITE, scale=0.6, thick=1) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(view: np.ndarray, status: str, status_color, progress: float | None,
                 result: Result | None) -> None:
    h, w = view.shape[:2]
    _text(view, status, (12, 28), status_color, 0.7, 2)
    if progress is not None:
        cv2.rectangle(view, (12, 40), (12 + int((w - 24) * progress), 50), RED, -1)
        cv2.rectangle(view, (12, 40), (w - 12, 50), WHITE, 1)
    if result is None:
        return
    panel_h = 150
    overlay = view.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, view, 0.4, 0, view)
    y = h - panel_h + 32
    if not result.ok:
        _text(view, "Could not recognise this clip", (12, y), YELLOW, 0.75, 2)
        _text(view, result.error[:90], (12, y + 30), WHITE, 0.5)
        return
    if result.confident:
        _text(view, f"{result.label}   ({result.confidence:.0%})", (12, y), GREEN, 0.95, 2)
    else:
        _text(view, f"Not sure   (best guess: {result.label}, {result.confidence:.0%})",
              (12, y), YELLOW, 0.8, 2)
    for i, (label, conf) in enumerate(result.top):
        _text(view, f"{i + 1}. {label}  {conf:.3f}", (12, y + 30 + 22 * i), WHITE, 0.55)
    _text(view, f"{result.total_ms / 1000:.1f} s after recording", (w - 230, h - 12), GREY, 0.5)
    if result.warnings:
        _text(view, result.warnings[0][:80], (12, h - 12), YELLOW, 0.45)


# --------------------------------------------------------------------------- main loop


def open_source(source: str) -> tuple[cv2.VideoCapture, bool]:
    is_file = not source.isdigit()
    cap = cv2.VideoCapture(source if is_file else int(source))
    return cap, is_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="camera index (default 0) or a video file")
    parser.add_argument("--auto-capture", type=int, default=0, metavar="N",
                        help="testing: trigger N captures automatically, then quit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("app.preprocessing",):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    print("Loading model and MediaPipe...", flush=True)
    rec = SignRecognizer(settings.model_dir)
    import mediapipe  # noqa: F401  (slow import done once, before the camera opens)

    cap, is_file = open_source(args.source)
    if not cap.isOpened():
        if is_file:
            print(f"error: could not open video file {args.source!r} (missing or unreadable).",
                  file=sys.stderr)
        else:
            print(f"error: could not open camera {args.source!r}. On macOS, allow camera access "
                  "for your terminal app in System Settings > Privacy & Security > Camera.",
                  file=sys.stderr)
        return 1
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        print("error: the camera opened but returned no frames (permission denied or in use?).",
              file=sys.stderr)
        return 1
    file_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 30.0

    tmp_dir = Path(tempfile.mkdtemp(prefix="signbridge-webcam-"))
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predict")
    future: Future[Result] | None = None
    result: Result | None = None
    recording = False
    rec_frames: list[np.ndarray] = []
    rec_times: list[float] = []
    rec_start = 0.0
    clip_no = 0
    auto_left = args.auto_capture
    next_frame_t = time.perf_counter()
    busy_gaps: list[float] = []  # preview frame intervals while a prediction runs
    last_shown = time.perf_counter()
    print("Ready. SPACE = capture 2 s and predict, Q = quit.", flush=True)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            if is_file:  # pace a file like a live camera, and loop it
                next_frame_t += 1.0 / file_fps
                time.sleep(max(0.0, next_frame_t - time.perf_counter()))
            ok, frame = cap.read()
            if (not ok or frame is None) and is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or frame is None:
                print("error: lost the camera feed.", file=sys.stderr)
                break
            now = time.perf_counter()

            busy = future is not None and not future.done()
            if future is not None and future.done():
                result = future.result()
                print_result(result)
                future = None
                if args.auto_capture and auto_left == 0:
                    break

            # Recording: keep raw (unmirrored) frames; preprocessing is mirror-invariant.
            if recording:
                rec_frames.append(frame.copy())
                rec_times.append(now)
                if now - rec_start >= CLIP_SECONDS:
                    recording = False
                    clip_no += 1
                    future = executor.submit(run_prediction, rec, rec_frames, rec_times, tmp_dir, clip_no)
                    rec_frames, rec_times = [], []
                    busy = True

            # Preview (mirrored like a selfie camera; display only).
            view = frame if is_file else cv2.flip(frame, 1)
            scale = PREVIEW_MAX_SIDE / max(view.shape[:2])
            if scale < 1.0:
                view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                view = view.copy()
            if recording:
                left = max(0.0, CLIP_SECONDS - (now - rec_start))
                draw_overlay(view, f"RECORDING... sign now ({left:.1f} s)", RED,
                             (now - rec_start) / CLIP_SECONDS, None)
            elif busy:
                draw_overlay(view, "Processing clip... (camera stays live; SPACE disabled)",
                             YELLOW, None, result)
            else:
                draw_overlay(view, "Ready: press SPACE and sign ONE word (chest-up, hands visible)",
                             GREEN, None, result)
            cv2.imshow(WINDOW, view)
            shown = time.perf_counter()
            if busy:
                busy_gaps.append(shown - last_shown)
            last_shown = shown

            key = cv2.waitKey(1) & 0xFF
            if auto_left and not recording and not busy:
                key, auto_left = ord(" "), auto_left - 1
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                if recording or busy:
                    print("(ignored SPACE: a clip is already being recorded/processed)", flush=True)
                else:
                    recording, rec_start = True, time.perf_counter()
                    rec_frames, rec_times = [], []
                    result = None
                    print("Recording 2 s...", flush=True)
            try:  # window closed with the mouse
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...", flush=True)
        cap.release()
        if future is not None and not future.done():
            print("(waiting for the running prediction to finish)", flush=True)
        executor.shutdown(wait=True)
        if busy_gaps:
            g = np.array(busy_gaps) * 1000
            print(f"Preview while processing: {len(g)} frames, median {np.median(g):.0f} ms, "
                  f"p95 {np.percentile(g, 95):.0f} ms, max {g.max():.0f} ms between frames", flush=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
