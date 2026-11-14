"""Combined-app regressions. All camera/GUI/tracking graph access is mocked."""

from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import realtime_translator as app
import ui
from motion import MotionTracker
from recognition_session import RecognitionSession
from smoothing import OneEuroFilter
from transcript import Transcript
from word_recognizer import Prediction, Recognition, WordRecognizer, load_replay, DEFAULT_REPOSITORY
from word_session import LiveWordSession
from word_tracking import TrackedFrame
from test_motion import pointing_hand
from test_word_live import moving, absent, still, DT


def prediction(label='hello', confidence=.9, accepted=True):
    return Recognition('sign' if accepted else 'not-sure',
                       (Prediction(0, label, label, confidence),),
                       None if accepted else 'partial')


def session(confidence=.75):
    return RecognitionSession(app.SpellingBuffer(), MotionTracker(), OneEuroFilter(), [], confidence)


class TranscriptTests(unittest.TestCase):
    def test_letters_words_spacing_and_resume_spelling(self):
        transcript = Transcript()
        transcript.append_letter('c')
        transcript.append_letter('a')
        transcript.append_letter('t')
        self.assertEqual(transcript.text, 'CAT')
        transcript.append_word('  thank   you ')
        transcript.append_word('hello')
        transcript.append_letter('a')
        transcript.append_letter('b')
        self.assertEqual(transcript.text, 'CAT thank you hello AB')

    def test_space_backspace_clear_repeat_and_undo(self):
        transcript = Transcript()
        transcript.space()
        transcript.append_letter('a')
        transcript.repeat()
        self.assertEqual(transcript.text, 'AA')
        transcript.space()
        transcript.space()
        self.assertEqual(transcript.text, 'AA ')
        transcript.append_word('hello')
        transcript.backspace()
        self.assertEqual(transcript.text, 'AA hell')
        transcript.undo()
        self.assertEqual(transcript.text, 'AA hello')
        transcript.clear()
        self.assertEqual(transcript.text, '')
        self.assertEqual(transcript.latest, '')
        transcript.undo()
        self.assertEqual(transcript.text, 'AA hello')
        self.assertEqual(transcript.latest, 'hello')

    def test_undo_restores_word_boundary(self):
        transcript = Transcript()
        transcript.append_word('hello')
        transcript.append_letter('a')
        transcript.undo()
        transcript.append_letter('b')
        self.assertEqual(transcript.text, 'hello B')

    def test_clear_after_deleting_last_character_clears_latest(self):
        transcript = Transcript()
        transcript.append_letter('a')
        transcript.backspace()
        transcript.clear()
        self.assertEqual(transcript.latest, '')
        transcript.undo()
        self.assertEqual(transcript.latest, 'A')


class ModeTests(unittest.TestCase):
    def test_switch_resets_pending_state_preserves_unfinished_spelling(self):
        flow = session()
        flow.word_session = LiveWordSession(MagicMock())
        flow.commit_letter('c')
        flow.commit_letter('a')
        flow.score_history.append(np.ones(26))
        flow.motion.update(pointing_hand()[:, :2])
        flow.smoother(np.ones((21, 2)))
        flow.spelling.update('t', .99, True, False)
        flow.switch_mode('word')
        self.assertEqual(flow.spelling.progress(), 0)
        self.assertFalse(flow.score_history)
        self.assertFalse(flow.motion.buffer)
        self.assertEqual(flow.word_session.segmenter.state, 'idle')
        flow.commit_letter('x')  # inactive pipeline cannot write
        flow.switch_mode('letter')
        flow.commit_letter('t')
        self.assertEqual(flow.transcript.text, 'CAT')

    def test_rejections_and_separate_commit_threshold(self):
        flow = session()
        flow.switch_mode('word')
        for i, result in enumerate((prediction(confidence=.74), prediction(accepted=False),
                                    Recognition('nothing'), prediction(confidence=float('nan')))):
            self.assertFalse(flow.commit_word(result, i, flow.generation))
        self.assertEqual(flow.transcript.text, '')
        self.assertTrue(flow.commit_word(prediction(confidence=.75), 10, flow.generation))
        self.assertEqual(flow.transcript.text, 'hello')
        flow = session(.60)
        flow.switch_mode('word')
        self.assertTrue(flow.commit_word(prediction(confidence=.61), 1, flow.generation))
        with self.assertRaises(ValueError):
            session(1.1)

    def test_duplicate_result_cannot_recommit_even_after_undo(self):
        flow = session()
        flow.switch_mode('word')
        self.assertTrue(flow.commit_word(prediction(), 'gesture1', flow.generation))
        self.assertFalse(flow.commit_word(prediction(), 'gesture1', flow.generation))
        flow.transcript.undo()
        self.assertFalse(flow.commit_word(prediction(), 'gesture1', flow.generation))
        self.assertEqual(flow.transcript.text, '')
        self.assertTrue(flow.commit_word(prediction(), 'gesture2', flow.generation))
        self.assertTrue(flow.commit_word(prediction(), 'gesture3', flow.generation))
        self.assertEqual(flow.transcript.text, 'hello hello')

    def test_old_generation_results_ignored(self):
        flow = session()
        flow.switch_mode('word')
        old = flow.generation
        flow.switch_mode('letter')
        self.assertFalse(flow.commit_word(prediction(), 1, old))
        flow.switch_mode('word')
        self.assertFalse(flow.commit_word(prediction(), 1, old))
        self.assertEqual(flow.transcript.text, '')

    def test_switch_during_inference_cannot_commit_or_change_new_mode(self):
        flow = session()
        recognizer = MagicMock()
        def infer(*args, **kwargs):
            flow.switch_mode('letter')
            return prediction()
        recognizer.recognize.side_effect = infer
        flow.word_session = LiveWordSession(recognizer)
        flow.switch_mode('word')
        for i, frame in enumerate(moving(20) + absent(12)):
            flow.push_word(TrackedFrame(frame, 1), i * DT)
        self.assertEqual(flow.mode, 'letter')
        self.assertEqual(flow.transcript.text, '')

    def test_stillness_and_trailing_exit_do_not_duplicate_commits(self):
        flow = session()
        recognizer = MagicMock()
        recognizer.recognize.return_value = prediction()
        flow.word_session = LiveWordSession(recognizer)
        flow.switch_mode('word')
        frames = moving(20) + still(30) + moving(10, 20) + absent(12)
        for i, frame in enumerate(frames):
            flow.push_word(TrackedFrame(frame, 1), i * DT)
        self.assertEqual(flow.transcript.text, 'hello')
        recognizer.recognize.assert_called_once()

    def test_timeout_waits_for_real_boundary_before_rearming(self):
        flow = session()
        recognizer = MagicMock()
        recognizer.recognize.return_value = prediction()
        flow.word_session = LiveWordSession(recognizer)
        flow.switch_mode('word')
        frames = moving(260) + absent(15) + moving(20) + absent(15)
        for i, frame in enumerate(frames):
            flow.push_word(TrackedFrame(frame, 1), i * DT)
        self.assertEqual(recognizer.recognize.call_count, 2)
        self.assertEqual(flow.transcript.text, 'hello hello')

    def test_real_word_pipeline_automatic_commit_gates(self):
        flow = session()
        recognizer = WordRecognizer()
        flow.word_session = LiveWordSession(recognizer)
        _, clips = load_replay(DEFAULT_REPOSITORY / 'web/public/replay/clips.json')
        for sign, should_commit in (('hello', True), ('smile', False)):
            flow.switch_mode('letter')
            flow.switch_mode('word')
            flow.transcript.clear()
            clip = next(c for c in clips if c['sign'] == sign)
            tail = np.full((543, 3), np.nan, np.float32)
            tail[489:522] = clip['frames'][-1, 489:522]
            for i, frame in enumerate(list(clip['frames']) + [tail] * 15):
                flow.push_word(TrackedFrame(frame, 1), i * 1000 / clip['fps'])
            self.assertEqual(bool(flow.transcript.text), should_commit)
            if should_commit:
                self.assertEqual(flow.transcript.text, 'hello')


class LetterCommitTests(unittest.TestCase):
    def test_original_ten_frame_gate_confidence_motion_and_repeat(self):
        flow = session()
        for i in range(app.COMMIT_FRAMES):
            committed = flow.spelling.update('a', .99, True, False)
            flow.commit_letter(committed)
            self.assertEqual(flow.transcript.text, 'A' if i == 9 else '')
        for _ in range(20):
            flow.commit_letter(flow.spelling.update('a', .99, True, False))
        self.assertEqual(flow.transcript.text, 'A')
        flow.spelling.reset_pending()
        for _ in range(10):
            self.assertIsNone(flow.spelling.update('b', .74, True, False))
        flow.spelling.reset_pending()
        for _ in range(10):
            self.assertIsNone(flow.spelling.update('b', .99, True, True))
        flow.transcript.repeat()
        self.assertEqual(flow.transcript.text, 'AA')


class CombinedUITests(unittest.TestCase):
    def test_original_geometry_and_mode_button_hit_testing(self):
        layout = ui.Layout()
        self.assertEqual((layout.width, layout.height), (1100, 696))
        rects = ui.button_rects(layout)
        self.assertEqual(rects['filters'], (938, 638, 1088, 666))
        self.assertEqual(rects['reference'], (780, 638, 930, 666))
        for name, (x1, y1, x2, y2) in rects.items():
            self.assertEqual(ui.hit_test(layout, (x1 + x2) // 2, (y1 + y2) // 2), name)

    def test_word_ui_never_draws_top_three_candidates(self):
        for vis in (False, True):
            layout = ui.Layout(show_vis=vis)
            state = ui.ViewState(mode='word', status='Recording sign', text='CAT hello',
                                 latest_committed='hello')
            with patch.object(ui, 'draw_top3', side_effect=AssertionError('word candidates shown')):
                canvas = ui.compose(state, layout)
            self.assertEqual(canvas.shape, (layout.height, layout.width, 3))
        with patch.object(ui, 'draw_top3', wraps=ui.draw_top3) as draw:
            ui.compose(ui.ViewState(), ui.Layout())
            draw.assert_called_once()


class CombinedMainLoopTests(unittest.TestCase):
    def test_one_camera_selected_pipeline_mirroring_and_m_switch(self):
        self._run_combined(use_mouse=False)

    def test_mode_buttons_switch_actual_application(self):
        self._run_combined(use_mouse=True)

    def _run_combined(self, use_mouse):
        # Ten letter frames -> word gesture -> letter again. Exercise main(),
        # not a separate demo loop, while mocking only hardware/inference/GUI.
        raw = np.zeros((120, 160, 3), np.uint8)
        raw[:, :80] = (10, 20, 30)
        raw[:, 80:] = (40, 50, 60)
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.side_effect = lambda: (True, raw.copy())
        landmarks = SimpleNamespace(landmark=[SimpleNamespace(x=p[0]/500, y=p[1]/500, z=p[2]/500)
                                               for p in pointing_hand()])
        engines = []
        def hands_factory(**kwargs):
            engine = MagicMock()
            def process(rgb):
                # Letter inference must still see the horizontally mirrored RGB.
                np.testing.assert_array_equal(rgb[0, 0], [60, 50, 40])
                return SimpleNamespace(multi_hand_landmarks=[landmarks])
            engine.process.side_effect = process
            engines.append(engine)
            return engine
        fake_mp = SimpleNamespace(solutions=SimpleNamespace(
            hands=SimpleNamespace(Hands=hands_factory, HAND_CONNECTIONS=()),
            drawing_utils=SimpleNamespace(draw_landmarks=lambda *a: None),
            drawing_styles=SimpleNamespace(get_default_hand_landmarks_style=lambda: None,
                                          get_default_hand_connections_style=lambda: None)))
        cnn = MagicMock()
        logits = np.full((1, 26), -20., np.float32)
        logits[0, 0] = 20
        cnn.run.return_value = [logits]
        words = MagicMock()
        words.recognize.return_value = prediction()
        holistic = MagicMock()
        tracked_frames = iter(moving(20) + absent(12))
        def track(image):
            np.testing.assert_array_equal(image, raw)
            return TrackedFrame(next(tracked_frames), 1)
        holistic.process.side_effect = track
        states = []
        callback = None
        def mouse_callback(window, function):
            nonlocal callback
            callback = function
        frame_count = 0
        def key(delay):
            nonlocal frame_count
            frame_count += 1
            if frame_count in (10, 42):
                if use_mouse:
                    target = 'word' if frame_count == 10 else 'letter'
                    x1, y1, x2, y2 = ui.button_rects(ui.Layout())[target]
                    callback(app.cv2.EVENT_LBUTTONDOWN, (x1+x2)//2, (y1+y2)//2, 0, None)
                    return -1
                return ord('m')
            return ord('q') if frame_count >= 43 else -1
        def compose(state, layout):
            states.append(state)
            return np.zeros((layout.height, layout.width, 3), np.uint8)
        with patch.dict('sys.modules', {'mediapipe': fake_mp}), \
             patch.object(app.cv2, 'VideoCapture', return_value=camera) as open_camera, \
             patch.object(app, 'load_session', return_value=(cnn, 'input', 96, ['logits'], False)), \
             patch.object(app, 'WordRecognizer', return_value=words), \
             patch.object(app, 'HolisticTracker', return_value=holistic), \
             patch.object(app.time, 'monotonic_ns', side_effect=[int(i*DT*1e6) for i in range(50)]), \
             patch.object(app.cv2, 'namedWindow'), patch.object(app.cv2, 'imshow'), \
             patch.object(app.cv2, 'setMouseCallback', side_effect=mouse_callback), \
             patch.object(app.cv2, 'waitKey', side_effect=key), \
             patch.object(app.cv2, 'getWindowProperty', return_value=1), \
             patch.object(app.cv2, 'destroyAllWindows'), \
             patch.object(ui, 'compose', side_effect=compose), redirect_stdout(io.StringIO()):
            app.main([])
        open_camera.assert_called_once_with(0)
        camera.release.assert_called_once()
        self.assertEqual(cnn.run.call_count, 11)
        self.assertEqual(sum(e.process.call_count for e in engines), 11)
        self.assertEqual(holistic.process.call_count, 32)
        words.recognize.assert_called_once()
        holistic.close.assert_called_once()
        self.assertEqual(states[-1].text, 'A hello')
        self.assertEqual(states[-1].mode, 'letter')
        self.assertEqual(states[-1].latest_committed, 'hello')
        for state in states:
            if state.mode == 'word':
                self.assertIsNone(state.scores)


if __name__ == '__main__':
    unittest.main()
