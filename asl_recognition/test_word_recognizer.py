"""Word inference regression tests using the real bundled model and fixtures."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np

from word_recognizer import DEFAULT_REPOSITORY, WordRecognizer, decide, load_replay, main


class WordRecognizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = WordRecognizer()
        cls.credit, cls.clips = load_replay(DEFAULT_REPOSITORY / "web/public/replay/clips.json")

    def test_real_model_contract(self):
        recognizer = self.recognizer
        self.assertEqual(recognizer.session.get_inputs()[0].shape, [1, 'T', 460])
        self.assertEqual(len(recognizer.labels), 250)
        self.assertTrue(all(label['shipped'] for label in recognizer.labels))
        self.assertEqual(recognizer.meta['featureVersion'], 2)
        for module in ('torch', 'tensorflow', 'mediapipe', 'gesturesign.dataset',
                       'gesturesign.train', 'gesturesign.fixtures'):
            self.assertNotIn(module, sys.modules)

    def test_all_six_replays(self):
        self.assertEqual(len(self.clips), 6)
        self.assertIn('CC BY 4.0', self.credit)
        expected = {'blue': 'blue', 'police': 'police', 'quiet': 'shh',
                    'giraffe': 'giraffe', 'smile': 'smile', 'hello': 'hello'}
        for clip in self.clips:
            with self.subTest(sign=clip['sign']):
                result = self.recognizer.recognize(clip['frames'], clip['fps'])
                self.assertTrue(result.accepted)
                self.assertEqual(result.top[0].label, expected[clip['sign']])
                self.assertEqual(len(result.top), 3)
                self.assertTrue(all(0 <= p.confidence <= 1 for p in result.top))
                self.assertEqual(sorted(p.confidence for p in result.top)[::-1],
                                 [p.confidence for p in result.top])

    def test_all_eleven_upstream_parity_fixtures(self):
        fixture = json.loads((DEFAULT_REPOSITORY / 'shared/fixtures/features.json').read_text())
        self.assertEqual(len(fixture['cases']), 11)
        self.assertEqual(fixture['featureVersion'], self.recognizer.features.FEATURE_VERSION)
        for case in fixture['cases']:
            with self.subTest(case=case['name']):
                frames = np.full((len(case['frames']), 543, 3), np.nan, dtype=np.float32)
                frames[:, fixture['points'], :2] = np.asarray(
                    case['frames'], dtype=np.float32).reshape(len(frames), len(fixture['points']), 2)
                if case['kind'] == 'frame':
                    got = self.recognizer.features.frame_features(frames[0], case['featureSet'])
                else:
                    fitted = self.recognizer.resampling.fit_length(
                        self.recognizer.resampling.resample(frames, case['srcFps'], case['fps']),
                        case['maxFrames'])
                    got = self.recognizer.features.sequence_features(fitted, case['featureSet'])
                    self.assertEqual(len(got), case['expected']['length'])
                    # Exercise adapter.prepare itself for every shipped-feature case.
                    if case['featureSet'] == self.recognizer.meta['featureSet']:
                        np.testing.assert_array_equal(
                            self.recognizer.prepare(frames, case['srcFps']), got)
                np.testing.assert_allclose(got, np.asarray(case['expected']['features']),
                                           atol=fixture['tolerance'], rtol=0)

    def test_adapter_resamples_and_squeezes_full_clip(self):
        frames = np.tile(self.clips[0]['frames'], (3, 1, 1))
        features = self.recognizer.prepare(frames, 25)
        self.assertEqual(features.shape, (64, 460))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(self.recognizer.recognize(frames, 25).input_frames, 64)

    def test_short_empty_and_missing_hands_rejected(self):
        for length in (0, 1, 7):
            result = self.recognizer.recognize(self.clips[0]['frames'][:length], 30)
            self.assertFalse(result.accepted)
            self.assertEqual(result.reason, 'too-short')
            self.assertFalse(result.top)
        frames = np.full((12, 543, 3), np.nan, dtype=np.float32)
        result = self.recognizer.recognize(frames, 30)
        self.assertEqual(result.reason, 'no-hands')
        self.assertFalse(result.top)

    def test_partial_frame_rejection_preserves_guesses(self):
        clip = self.clips[-1]
        result = self.recognizer.recognize(clip['frames'], clip['fps'], partial=True)
        self.assertEqual(result.status, 'not-sure')
        self.assertEqual(result.reason, 'partial')
        self.assertEqual(len(result.top), 3)

    def test_invalid_inputs(self):
        frames = self.clips[0]['frames']
        for fps in (0, -1, np.nan, np.inf):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                self.recognizer.prepare(frames, fps)
        for bad in (np.zeros((10, 21, 2)), np.zeros((543, 3)),
                    np.full((10, 543, 3), np.inf)):
            with self.assertRaises(ValueError):
                self.recognizer.prepare(bad, 30)

    def test_metadata_mismatch_rejected(self):
        for key, value in (('featureVersion', -1), ('featureDim', 1), ('classes', 249),
                           ('fps', 0), ('minFrames', 100)):
            with self.subTest(key=key):
                probe = object.__new__(WordRecognizer)
                probe.features = self.recognizer.features
                probe.labels = copy.deepcopy(self.recognizer.labels)
                probe.meta = copy.deepcopy(self.recognizer.meta)
                probe.meta[key] = value
                with self.assertRaises(ValueError):
                    probe._validate_metadata()

    def test_missing_clone_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, 'preprocessing missing'):
            WordRecognizer(Path(__file__).with_name('nonexistent-gesturesign-clone'))

    def test_cli_reports_every_clip_and_rejection_status(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main([]), 0)
        lines = [line for line in output.getvalue().splitlines() if 'expected=' in line]
        self.assertEqual(len(lines), 6)
        for line in lines:
            self.assertIn('ACCEPT', line)
            self.assertEqual(line.count('%'), 3)
        self.assertIn('expected=quiet | shh', output.getvalue())


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.labels = [dict(index=i, sign=f'sign{i}', label=f'word {i}', shipped=True)
                       for i in range(4)]

    def test_temperature_matches_independent_softmax(self):
        logits = np.array([1., 2., 3., 4.])
        result = decide(logits, self.labels, .5, .29)
        reference = np.exp((logits - logits.max()) / .5)
        reference /= reference.sum()
        self.assertTrue(result.accepted)
        self.assertEqual([p.index for p in result.top], [3, 2, 1])
        self.assertAlmostEqual(result.top[0].confidence, reference[3])

    def test_low_confidence_and_threshold_boundary(self):
        result = decide(np.zeros(4), self.labels, 1, .29)
        self.assertEqual(result.reason, 'low-confidence')
        self.assertEqual(len(result.top), 3)
        self.assertTrue(decide(np.zeros(4), self.labels, 1, .25).accepted)

    def test_unshipped_labels_excluded_from_softmax(self):
        self.labels[3]['shipped'] = False
        result = decide([0, 0, 0, 1000], self.labels, 1, .29)
        self.assertEqual([p.index for p in result.top], [0, 1, 2])
        self.assertAlmostEqual(result.top[0].confidence, 1 / 3)
        for label in self.labels:
            label['shipped'] = False
        self.assertEqual(decide([0] * 4, self.labels, 1, .29).reason, 'no-shipped-labels')

    def test_invalid_logits_and_calibration(self):
        for bad in (np.nan, np.inf, -np.inf):
            result = decide([bad, 0, 0, 0], self.labels, 1, .29)
            self.assertEqual(result.reason, 'invalid-logits')
        with self.assertRaises(ValueError):
            decide([0], self.labels, 1, .29)
        for temperature, threshold in ((0, .29), (np.nan, .29), (1, 1.1), (1, -.1)):
            with self.assertRaises(ValueError):
                decide([0] * 4, self.labels, temperature, threshold)

    def test_display_label_not_raw_sign(self):
        labels = [dict(index=0, sign='thankyou', label='thank you', shipped=True)]
        result = decide([10], labels, 1, .29)
        self.assertEqual(result.top[0].sign, 'thankyou')
        self.assertEqual(result.top[0].label, 'thank you')


if __name__ == '__main__':
    unittest.main()
