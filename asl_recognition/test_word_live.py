"""Camera-free Phase 2 tests, including upstream segmenter regression cases."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from word_recognizer import DEFAULT_REPOSITORY, Recognition, WordRecognizer, load_replay
from word_segmenter import Segmenter, SegmenterOptions, hand_speed
from word_tracking import (HolisticTracker, TrackedFrame, format_holistic, frame_status,
                           LEFT_HAND_START, POSE_START, RIGHT_HAND_START)
from test_word_webcam import LiveWordSession, display_point


DT = 1000 / 30


def body():
    frame = np.full((543, 3), np.nan, dtype=np.float32)
    frame[POSE_START + 11] = (.62, .6, 0)
    frame[POSE_START + 12] = (.38, .6, 0)
    return frame


def with_hand(x, y=.4, start=RIGHT_HAND_START):
    frame = body()
    for i in range(21):
        frame[start + i] = (x + i * .003, y - i * .004, 0)
    return frame


def moving(n, start=0):
    return [with_hand(.3 + .01 * (start + i)) for i in range(n)]


def still(n, x=.5):
    return [with_hand(x) for _ in range(n)]


def absent(n):
    return [body() for _ in range(n)]


def run(segmenter, frames, t0=0, dt=DT, **kwargs):
    return [event for i, frame in enumerate(frames)
            if (event := segmenter.push(frame, t0 + i * dt, **kwargs)) is not None]


class SegmenterTests(unittest.TestCase):
    def test_no_hands_stays_idle(self):
        seg = Segmenter()
        self.assertEqual(run(seg, absent(40)), [])
        self.assertEqual(seg.state, 'idle')

    def test_absence_ends_and_trims_gap(self):
        seg = Segmenter()
        events = run(seg, absent(3) + moving(20) + absent(12))
        self.assertEqual([e.type for e in events], ['start', 'end'])
        end = events[1]
        self.assertEqual(end.reason, 'absent')
        self.assertTrue(20 <= len(end.segment.frames) <= 22)
        self.assertAlmostEqual(end.segment.fps, 30)
        self.assertTrue(np.isfinite(end.segment.frames[-1, RIGHT_HAND_START, 0]))

    def test_bridges_short_dropout(self):
        events = run(Segmenter(), moving(10) + absent(5) + moving(10, 15) + absent(12))
        self.assertEqual([e.type for e in events], ['start', 'end'])
        self.assertGreaterEqual(len(events[1].segment.frames), 25)
        self.assertTrue(np.isnan(events[1].segment.frames[10:15, RIGHT_HAND_START, 0]).all())

    def test_still_hand_and_small_jitter_do_not_start(self):
        self.assertEqual(run(Segmenter(), still(40)), [])
        rng = np.random.default_rng(23)
        frames = [with_hand(.5 + jitter) for jitter in rng.normal(0, .0002, 60)]
        self.assertEqual(run(Segmenter(), frames), [])

    def test_still_ending_waits_for_new_motion(self):
        seg = Segmenter()
        events = run(seg, moving(20) + still(30))
        self.assertEqual([e.type for e in events], ['start', 'end'])
        self.assertEqual(events[1].reason, 'still')
        self.assertEqual(seg.state, 'resting')
        self.assertEqual(run(seg, still(20), 50 * DT), [])

    def test_trailing_motion_rejected(self):
        seg = Segmenter()
        run(seg, moving(20) + still(30))
        events = run(seg, moving(10, 20) + absent(12), 50 * DT)
        self.assertEqual([e.type for e in events], ['start', 'discard'])
        self.assertEqual(events[1].reason, 'hands-leaving')

    def test_full_sign_from_rest_is_kept(self):
        seg = Segmenter()
        run(seg, moving(20) + still(30))
        events = run(seg, moving(24, 20) + absent(12), 50 * DT)
        self.assertEqual([e.type for e in events], ['start', 'end'])

    def test_incomplete_short_movement_discarded(self):
        events = run(Segmenter(), moving(4) + absent(12))
        self.assertEqual([e.type for e in events], ['start', 'discard'])
        self.assertEqual(events[1].reason, 'too-short')

    def test_timeout(self):
        seg = Segmenter(SegmenterOptions(max_ms=2000))
        events = run(seg, moving(90))
        self.assertEqual([e.type for e in events], ['start', 'end', 'start'])
        self.assertEqual(events[1].reason, 'max')

    def test_estimates_source_fps(self):
        events = run(Segmenter(), moving(20) + absent(12), dt=50)
        self.assertAlmostEqual(events[1].segment.fps, 20)

    def test_crowded_frames_cannot_start(self):
        self.assertEqual(run(Segmenter(), moving(20), crowded=True), [])

    def test_flush_and_incomplete_flush(self):
        seg = Segmenter()
        run(seg, moving(15))
        event = seg.flush()
        self.assertEqual(event.type, 'end')
        self.assertEqual(event.reason, 'flush')
        self.assertIsNone(seg.flush())
        seg.reset()
        run(seg, moving(4))
        self.assertEqual(seg.flush().reason, 'too-short')

    def test_flush_trims_pending_absence(self):
        seg = Segmenter()
        run(seg, moving(15) + absent(3))
        self.assertEqual(len(seg.flush().segment.frames), 15)

    def test_preroll_keeps_recent_still_frames(self):
        seg = Segmenter()
        events = run(seg, still(15, .3) + moving(12))
        self.assertEqual(events[0].type, 'start')
        segment = seg.flush().segment
        self.assertLessEqual(events[0].t - segment.timestamps[0], 200)
        self.assertLess(segment.timestamps[0], events[0].t)
        self.assertGreater(segment.timestamps[0], 0)

    def test_reset_clears_all_pending_state(self):
        seg = Segmenter()
        run(seg, moving(15))
        self.assertEqual(seg.state, 'signing')
        seg.reset()
        self.assertEqual(seg.state, 'idle')
        self.assertEqual(seg.frame_count, 0)
        self.assertEqual(seg.speed, 0)
        self.assertEqual(seg.recent, [])
        self.assertIsNone(seg.flush())
        self.assertEqual(run(seg, still(20)), [])

    def test_timestamp_validation_and_frame_ownership(self):
        seg = Segmenter()
        frame = with_hand(.3)
        seg.push(frame, 100)
        frame[:] = np.nan
        self.assertTrue(np.isfinite(seg.last[0][RIGHT_HAND_START, 0]))
        for t in (100, 99, np.nan, np.inf):
            with self.assertRaises(ValueError):
                seg.push(body(), t)
        with self.assertRaises(ValueError):
            seg.push(np.zeros((21, 3)), 101)

    def test_speed_uses_shoulders_dt_and_common_points(self):
        previous, current = with_hand(.3), with_hand(.31)
        self.assertAlmostEqual(hand_speed(previous, current, DT), .01 / .24 * 30, places=5)
        self.assertEqual(hand_speed(body(), current, DT), 0)
        self.assertEqual(hand_speed(previous, current, 0), 0)
        current[POSE_START + 11] = np.nan
        self.assertAlmostEqual(hand_speed(previous, current, DT), .01 / .25 * 30, places=5)


def part(count, x):
    return SimpleNamespace(landmark=[SimpleNamespace(x=x + i * .001, y=.4, z=-.05)
                                     for i in range(count)])


class TrackingTests(unittest.TestCase):
    def test_exact_layout_normalized_coordinates_and_anatomical_sides(self):
        result = SimpleNamespace(face_landmarks=part(478, .1),
                                 left_hand_landmarks=part(21, .8),
                                 pose_landmarks=part(33, .2),
                                 right_hand_landmarks=part(21, .3))
        frame = format_holistic(result)
        self.assertEqual(frame.shape, (543, 3))
        self.assertEqual(frame.dtype, np.float32)
        for index, x in ((0, .1), (467, .567), (468, .8), (488, .82),
                         (489, .2), (521, .232), (522, .3), (542, .32)):
            np.testing.assert_allclose(frame[index], [x, .4, -.05], atol=1e-6)

    def test_missing_parts_stay_nan(self):
        self.assertTrue(np.isnan(format_holistic(SimpleNamespace())).all())
        frame = format_holistic(SimpleNamespace(left_hand_landmarks=part(21, .8)))
        self.assertTrue(np.isnan(frame[:468]).all())
        self.assertTrue(np.isnan(frame[489:]).all())
        self.assertTrue(np.isfinite(frame[468:489]).all())

    def test_incomplete_face_pose_and_nonfinite_coordinates(self):
        result = SimpleNamespace(face_landmarks=part(467, .1), pose_landmarks=part(32, .2),
                                 right_hand_landmarks=part(21, .3))
        result.right_hand_landmarks.landmark[0].x = np.inf
        frame = format_holistic(result)
        self.assertTrue(np.isnan(frame[:522]).all())
        self.assertTrue(np.isnan(frame[522, 0]))
        self.assertTrue(np.isfinite(frame[523:]).all())

    def test_quality_states(self):
        self.assertEqual(frame_status(np.full((543, 3), np.nan), 0), 'nothing')
        self.assertEqual(frame_status(body(), 1), 'no-hands')
        self.assertEqual(frame_status(with_hand(.3), 2), 'crowded')
        frame = with_hand(.3)
        self.assertEqual(frame_status(frame, 1), 'ok')
        frame[POSE_START + 11, 1] = 1.4
        self.assertEqual(frame_status(frame, 1), 'partial')
        frame[POSE_START + 11] = np.nan
        self.assertEqual(frame_status(frame, 1), 'partial')

    def test_preview_mirroring_does_not_change_model_coordinates(self):
        point = np.array([.2, .4, -.05])
        original = point.copy()
        self.assertEqual(display_point(point, 101, 101, False), (20, 40))
        self.assertEqual(display_point(point, 101, 101, True), (80, 40))
        np.testing.assert_array_equal(point, original)
        self.assertIsNone(display_point(np.full(3, np.nan), 101, 101, True))

    def test_tracker_rgb_conversion_reset_and_close_without_camera(self):
        outputs = SimpleNamespace(pose_landmarks=None)
        instances = []
        def factory(**kwargs):
            self.assertEqual(kwargs['model_complexity'], 1)
            self.assertFalse(kwargs['refine_face_landmarks'])
            engine = MagicMock()
            engine.process.return_value = outputs
            instances.append(engine)
            return engine
        fake_mp = SimpleNamespace(solutions=SimpleNamespace(holistic=SimpleNamespace(Holistic=factory)))
        fake_cv2 = SimpleNamespace(COLOR_BGR2RGB=1, cvtColor=lambda bgr, _: bgr[:, :, ::-1].copy())
        with patch.dict('sys.modules', {'mediapipe': fake_mp, 'cv2': fake_cv2}):
            tracker = HolisticTracker()
            image = np.full((2, 3, 3), [10, 20, 30], dtype=np.uint8)
            tracked = tracker.process(image)
            rgb = instances[0].process.call_args.args[0]
            np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])
            self.assertFalse(rgb.flags.writeable)
            self.assertEqual(tracked.status, 'nothing')
            tracker.reset()
            instances[0].close.assert_called_once()
            self.assertEqual(len(instances), 2)
            tracker.close()
            tracker.close()
            instances[1].close.assert_called_once()
            with self.assertRaises(RuntimeError):
                tracker.process(image)


class LiveSessionTests(unittest.TestCase):
    def test_six_replays_through_real_segmenter_and_onnx_adapter(self):
        recognizer = WordRecognizer()
        _, clips = load_replay(DEFAULT_REPOSITORY / 'web/public/replay/clips.json')
        for clip in clips:
            with self.subTest(sign=clip['sign']):
                session = LiveWordSession(recognizer)
                frames = list(clip['frames'])
                # Mirror browser camera-end behavior: shoulders remain, hands leave.
                tail = np.full((543, 3), np.nan, dtype=np.float32)
                tail[489:522] = frames[-1][489:522]
                frames += [tail.copy() for _ in range(15)]
                results = []
                for i, frame in enumerate(frames):
                    _, result = session.push(TrackedFrame(frame, 1), i * 1000 / clip['fps'])
                    if result is not None:
                        results.append(result)
                self.assertEqual(len(results), 1)
                self.assertTrue(results[0].accepted)
                self.assertEqual(len(results[0].top), 3)
                expected = 'shh' if clip['sign'] == 'quiet' else clip['sign']
                self.assertEqual(results[0].top[0].label, expected)

    def test_completed_sign_calls_unchanged_adapter_interface(self):
        recognizer = MagicMock()
        recognizer.recognize.return_value = Recognition('sign')
        session = LiveWordSession(recognizer)
        results = []
        for i, frame in enumerate(moving(20) + absent(12)):
            event, result = session.push(TrackedFrame(frame, 1), i * DT)
            if result is not None:
                results.append(result)
        self.assertEqual(len(results), 1)
        recognizer.recognize.assert_called_once()
        args, kwargs = recognizer.recognize.call_args
        self.assertEqual(args[0].shape[1:], (543, 3))
        self.assertAlmostEqual(args[1], 30)
        self.assertFalse(kwargs['partial'])

    def test_partial_framing_passed_to_rejection_logic(self):
        recognizer = MagicMock()
        recognizer.recognize.return_value = Recognition('not-sure', reason='partial')
        session = LiveWordSession(recognizer)
        frames = moving(20) + absent(12)
        for i, frame in enumerate(frames):
            frame[POSE_START + 11] = np.nan
            session.push(TrackedFrame(frame, 1), i * DT)
        self.assertTrue(recognizer.recognize.call_args.kwargs['partial'])
        self.assertFalse(session.result.accepted)

    def test_short_segments_and_reset_never_trigger_inference(self):
        recognizer = MagicMock()
        session = LiveWordSession(recognizer)
        for i, frame in enumerate(moving(4) + absent(12)):
            session.push(TrackedFrame(frame, 1), i * DT)
        recognizer.recognize.assert_not_called()
        session.reset()
        for i, frame in enumerate(moving(12)):
            session.push(TrackedFrame(frame, 1), i * DT)
        session.reset()
        self.assertIsNone(session.segmenter.flush())
        self.assertIsNone(session.result)
        self.assertEqual(session.completed, 0)
        recognizer.recognize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
