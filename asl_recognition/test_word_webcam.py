"""Standalone Phase 2 webcam demo. Importing this file never opens a camera.

Launch: .venv/bin/python -B test_word_webcam.py --camera 0 --debug
GestureSign segmentation/quality behavior is MIT, Copyright 2026 Shashank
Yadav; see WORD_RECOGNITION.md for the full notice and WORD_WEBCAM.md for use.
"""

import argparse
import time

import numpy as np

from word_recognizer import DEFAULT_REPOSITORY, WordRecognizer
from word_session import LiveWordSession
from word_tracking import HolisticTracker, LEFT_HAND_START, POSE_START, RIGHT_HAND_START



def display_point(point, width, height, mirror):
    """Map normalized unmirrored landmarks to preview pixels only."""
    if not np.isfinite(point[:2]).all():
        return None
    x, y = point[:2]
    if mirror:
        x = 1 - x
    return int(round(x * (width - 1))), int(round(y * (height - 1)))


def draw_preview(cv2, image, tracked, session, fps, mirror, landmark_layout):
    preview = cv2.flip(image, 1) if mirror else image.copy()
    h, w = preview.shape[:2]

    def line(a, b, color):
        pa = display_point(tracked.frame[a], w, h, mirror)
        pb = display_point(tracked.frame[b], w, h, mirror)
        if pa is not None and pb is not None:
            cv2.line(preview, pa, pb, color, 2, cv2.LINE_AA)

    for start, color, side in ((LEFT_HAND_START, (0, 220, 255), "anatomical L"),
                               (RIGHT_HAND_START, (255, 200, 80), "anatomical R")):
        for a, b in landmark_layout.HAND_CONNECTIONS:
            line(start + a, start + b, color)
        wrist = display_point(tracked.frame[start], w, h, mirror)
        if wrist is not None:
            cv2.putText(preview, side, wrist, cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    for a, b in ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (7, 2), (2, 0), (0, 5), (5, 8)):
        line(POSE_START + a, POSE_START + b, (120, 230, 120))
    for i in landmark_layout.LIPS:
        point = display_point(tracked.frame[i], w, h, mirror)
        if point is not None:
            cv2.circle(preview, point, 1, (200, 160, 255), -1)

    seg = session.segmenter
    result = session.result
    lines = [
        f"WORD ONLY | {tracked.status} | hands {tracked.hands}/2 | face {'yes' if tracked.face_present else 'no'} | {fps:.1f} FPS",
        f"{seg.state.upper()} | frames {seg.frame_count} | speed {seg.speed:.2f}/0.60 shoulder widths/s",
        f"Event: {session.last_event} | completed {session.completed}",
        "Last completed sign: " + ("none" if result is None else
            ("ACCEPT" if result.accepted else f"REJECT: {result.reason}")),
    ]
    if result is not None:
        lines += [f"{i + 1}. {guess.label}: {guess.confidence:.1%}"
                  for i, guess in enumerate(result.top)]
    else:
        lines += ["Sign, then hold still or lower your hands for 300 ms."]
    lines += ["Keep face, shoulders and both hands visible. q: quit | r: reset"]
    panel = np.full((len(lines) * 25 + 18, w, 3), 20, dtype=np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(panel, text, (10, 23 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .50, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack((preview, panel))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Standalone webcam isolated ASL word recognition")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-mirror", action="store_true", help="Disable preview mirroring; model input is always unmirrored")
    parser.add_argument("--debug", action="store_true", help="Print live tracking/segment state once per second")
    parser.add_argument("--repository", type=type(DEFAULT_REPOSITORY), default=DEFAULT_REPOSITORY)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args(argv)
    import cv2

    capture = None
    tracker = None
    window = "ASL Words - standalone Phase 2"
    try:
        recognizer = WordRecognizer(repository=args.repository)
        session = LiveWordSession(recognizer)
        tracker = HolisticTracker()
        # Only this standalone entry point owns camera access.
        capture = cv2.VideoCapture(args.camera)
        if not capture.isOpened():
            raise RuntimeError("Cannot open webcam. Check camera index and macOS camera permission for this terminal/app.")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("Word webcam ready. One person; keep face, shoulders and hands visible.", flush=True)
        print("Sign, then pause or lower hands for 300 ms. q quits; r resets. Preview mirroring never changes model input.", flush=True)
        print("Python Holistic tracks one person; it cannot reliably detect additional people.", flush=True)
        last_time = None
        last_debug = float('-inf')
        last_status = None
        fps = 0.0
        while True:
            ok, image = capture.read()
            t = time.monotonic_ns() / 1_000_000
            if not ok:
                raise RuntimeError("Webcam frame read failed")
            if last_time is not None:
                instant = 1000 / (t - last_time)
                fps = instant if fps == 0 else .9 * fps + .1 * instant
            last_time = t
            tracked = tracker.process(image)  # original unmirrored camera frame
            event, result = session.push(tracked, t)
            if tracked.status != last_status or (args.debug and t - last_debug >= 1000):
                print(f"tracking={tracked.status} hands={tracked.hands} face={tracked.face_present} "
                      f"state={session.segmenter.state} frames={session.segmenter.frame_count} "
                      f"speed={session.segmenter.speed:.3f} fps={fps:.1f}", flush=True)
                last_debug, last_status = t, tracked.status
            if event is not None:
                detail = "" if event.segment is None else (
                    f" frames={len(event.segment.frames)} fps={event.segment.fps:.1f} "
                    f"duration={event.segment.duration_ms:.0f}ms")
                print(f"segment={event.type} reason={event.reason or '-'}{detail}", flush=True)
            if result is not None:
                guesses = '; '.join(f"{g.label} {g.confidence:.1%}" for g in result.top)
                print(f"{'ACCEPT' if result.accepted else 'REJECT'} reason={result.reason or '-'} "
                      f"model_frames={result.input_frames} | {guesses or 'no guesses'}", flush=True)
            cv2.imshow(window, draw_preview(cv2, image, tracked, session, fps,
                                            not args.no_mirror, recognizer.features.lm))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('r'):
                session.reset()
                tracker.reset()
                last_time = None
                print("Reset: discarded open recording and cleared tracker history/results.", flush=True)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        print(f"Word webcam error: {error}", flush=True)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        if capture is not None:
            capture.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
