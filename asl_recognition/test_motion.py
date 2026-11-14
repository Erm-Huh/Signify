"""Synthetic pixel-landmark regression tests; no webcam or model required."""

import ast
from pathlib import Path
import unittest

import numpy as np

from motion import MotionTracker, INDEX_TIP, PINKY_TIP, _reversals, z_handshape


def trajectory(vertices, samples=14):
    return np.concatenate([
        np.linspace(a, b, samples, endpoint=False)
        for a, b in zip(vertices, vertices[1:])
    ] + [np.asarray(vertices[-1:])])


def pointing_hand():
    hand = np.zeros((21, 3))
    hand[:] = (250, 270, 0)
    hand[0] = (250, 300, 0)
    for i, x in zip((5, 9, 13, 17), (230, 250, 270, 290)):
        hand[i:i + 4] = [(x, 250, 0), (x, 230, 0),
                          (x, 245, -5), (x, 255, -5)]
    hand[5:9] = [(230, 250, 0), (225, 230, 0),
                 (220, 210, 0), (215, 190, 0)]
    return hand


def feed(path, tip=INDEX_TIP, translate=True, shape=None):
    tracker = MotionTracker()
    # Fixed 100px hand extent: path coordinates are in hand widths.
    hand = np.full((21, 2), 250.0)
    hand[0] = (200, 300)
    hand[12] = (300, 200)
    if shape is not None:
        hand = shape[:, :2].copy()
    for offset in path:
        points = hand.copy()
        if translate:
            points += np.asarray(offset) * 100
        else:
            points[tip] += np.asarray(offset) * 100
        tracker.update(points, handshape_points=shape)
    return tracker, points


class MotionTests(unittest.TestCase):
    def test_landmark_gate_overrides_n_only_for_complete_z(self):
        shape = pointing_hand()
        path = trajectory([(0, 0), (.9, 0), (0, .6), (.9, .6)], samples=10)
        for mirror in (1, -1):
            hand = shape.copy()
            hand[:, 0] *= mirror
            tracker, _ = feed(path * (mirror, 1), shape=hand)
            self.assertTrue(tracker.z_shape_ready())
            self.assertEqual(tracker.classify('n'), 'z')

    def test_wrong_handshapes_cannot_override_n(self):
        path = trajectory([(0, 0), (.9, 0), (0, .6), (.9, .6)], samples=10)
        for variant in ('fist', 'two_fingers', 'open_hand', 'bent_index'):
            hand = pointing_hand()
            if variant in ('fist', 'bent_index'):
                hand[6:9] = [(230, 230, 0), (230, 245, -5), (230, 255, -5)]
            if variant in ('two_fingers', 'open_hand'):
                for base in ((9,) if variant == 'two_fingers' else (9, 13, 17)):
                    hand[base:base + 4] = hand[5:9] + (20, 0, 0)
            tracker, _ = feed(path, shape=hand)
            with self.subTest(variant=variant):
                self.assertFalse(tracker.z_shape_ready())
                self.assertIsNone(tracker.classify('n'))

    def test_shape_check_rotation_depth_and_degenerate_points(self):
        hand = pointing_hand()
        for angle in (0, .5, 1.5, 3):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            self.assertTrue(z_handshape(hand @ rotation.T)['valid'])
        # Rotate out of the camera plane: 3D shape survives foreshortening.
        rotation = np.array([[1, 0, 0], [0, .5, -.866], [0, .866, .5]])
        self.assertTrue(z_handshape(hand @ rotation.T)['valid'])
        self.assertFalse(z_handshape(np.zeros((21, 3)))['valid'])

    def test_shape_must_accompany_gesture(self):
        tracker, points = feed(trajectory([(0, 0), (.9, 0), (0, .6), (.9, .6)]))
        tracker.update(points, handshape_points=pointing_hand())
        self.assertFalse(tracker.z_shape_ready())
        self.assertIsNone(tracker.classify('n'))

    def test_pointing_static_incomplete_and_random_paths_keep_n(self):
        rng = np.random.default_rng(13)
        paths = [np.zeros((45, 2)), trajectory([(0, 0), (.9, 0), (0, .6)])]
        paths += [np.cumsum(rng.normal(0, .08, (45, 2)), axis=0) for _ in range(100)]
        for path in paths:
            tracker, _ = feed(path, shape=pointing_hand())
            self.assertIsNone(tracker.classify('n'))

    def test_z_normalization_does_not_invent_motion_from_extent(self):
        tracker = MotionTracker()
        for size_change in np.linspace(0, 40, 45):
            hand = pointing_hand()[:, :2]
            hand[0, 1] += size_change
            tracker.update(hand)
        np.testing.assert_allclose(tracker._trajectory(INDEX_TIP), 0, atol=1e-6)
        self.assertAlmostEqual(tracker.stats(INDEX_TIP)['path'], 0)

    def test_slow_z_both_horizontal_directions(self):
        for mirror in (1, -1):
            path = trajectory([(0, 0), (.8, 0), (0, .5), (.8, .5)], samples=15)
            path[:, 0] *= mirror
            tracker, _ = feed(path)
            s = tracker.stats(INDEX_TIP)
            # Reproduce the old gate: a complete slow Z loses its reversals.
            track = tracker._trajectory(INDEX_TIP)
            old = _reversals(np.diff(track[:, 0]), max(s['path'] / 40, .001))
            self.assertLess(old, 2)
            for label in ('d', 'z', 'D'):
                self.assertEqual(tracker.classify(label), 'z')

    def test_fast_z_and_endpoint_hold(self):
        path = trajectory([(0, 0), (.8, 0), (0, .5), (.8, .5)], samples=8)
        tracker, points = feed(path)
        self.assertEqual(tracker.classify('d'), 'z')
        for _ in range(12):
            tracker.update(points)
            self.assertEqual(tracker.classify('d'), 'z')

    def test_incomplete_and_wrong_geometry(self):
        for vertices in (
            [(0, 0), (.8, 0)],
            [(0, 0), (.8, 0), (0, .5)],
            [(0, 0), (.8, 0), (0, .5), (.1, .5)],
            [(0, 0), (.8, 0), (0, 0), (.8, 0)],
            [(0, 0), (.8, 0), (0, -.5), (.8, -.5)],
        ):
            with self.subTest(vertices=vertices):
                tracker, _ = feed(trajectory(vertices))
                self.assertNotEqual(tracker.classify('d'), 'z')

    def test_random_movement(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            for path in (rng.normal(0, .025, (45, 2)),
                         np.cumsum(rng.normal(0, .08, (45, 2)), axis=0)):
                tracker, _ = feed(path)
                self.assertNotEqual(tracker.classify('d'), 'z')

    def test_static_and_incompatible_labels(self):
        tracker, _ = feed(np.zeros((45, 2)))
        self.assertIsNone(tracker.classify('d'))
        self.assertEqual(tracker.classify('z'), 'd')
        tracker, _ = feed(trajectory([(0, 0), (.8, 0), (0, .5), (.8, .5)]))
        for label in ('a', 'b', 'x', None):
            self.assertIsNone(tracker.classify(label))
        self.assertIn('gate blocked', tracker.debug_lines('x')[0])

    def test_j_regression(self):
        tracker, _ = feed(trajectory([(0, 0), (0, .5), (.4, .5)]), PINKY_TIP)
        self.assertEqual(tracker.classify('i'), 'j')
        self.assertEqual(tracker.classify('j'), 'j')
        tracker, _ = feed(np.zeros((45, 2)))
        self.assertIsNone(tracker.classify('i'))
        self.assertEqual(tracker.classify('j'), 'i')

    def test_buffer_readiness_and_reset(self):
        tracker, _ = feed(np.zeros((14, 2)))
        self.assertIsNone(tracker.stats(INDEX_TIP))
        tracker.reset()
        self.assertIsNone(tracker.classify('z'))

    def test_z_commits_after_stopping(self):
        # Load the actual spelling gate without importing camera/ONNX libraries.
        source = ast.parse(Path(__file__).with_name('realtime_translator.py').read_text())
        nodes = [node for node in source.body if
                 isinstance(node, ast.ClassDef) and node.name == 'SpellingBuffer'
                 or isinstance(node, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id in ('COMMIT_FRAMES', 'COMMIT_CONFIDENCE')
                     for t in node.targets)]
        scope = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<spelling gate>', 'exec'), scope)
        for label, shape in (('d', None), ('n', pointing_hand())):
            buffer = scope['SpellingBuffer']()
            tracker, points = feed(
                trajectory([(0, 0), (.8, 0), (0, .5), (.8, .5)], samples=8), shape=shape)
            buffer.update(tracker.classify(label), .9, True, tracker.is_moving())
            self.assertEqual(buffer.text, '')
            for _ in range(12):
                tracker.update(points, handshape_points=shape)
                buffer.update(tracker.classify(label), .9, True, tracker.is_moving())
            self.assertEqual(buffer.text, 'Z')

    def test_trace_matches_segmented_strokes(self):
        tracker, _ = feed(trajectory([(0, 0), (.8, 0), (0, .5), (.8, .5)]))
        path, vertices = tracker.z_trace()
        self.assertEqual(len(path), len(tracker.buffer))
        self.assertEqual(len(vertices), 4)
        np.testing.assert_array_equal(np.sign(np.diff(vertices[:, 0])), [1, -1, 1])
        np.testing.assert_allclose(path[0], vertices[0])
        tracker.reset()
        self.assertEqual(len(tracker.z_trace()[0]), 0)


if __name__ == '__main__':
    unittest.main()
