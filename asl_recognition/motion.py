"""Trajectory detection for J and Z, the two ASL letters that are movements.

The classifier works one frame at a time, so for J and Z it can only ever see the
handshape at the start or end of the gesture — the limitation the README documents.
This module watches the last second of hand positions and decides whether the motion
that accompanies a J-ish or Z-ish handshape is actually there.

Two things worth knowing about the implementation:

**It deliberately does not use the normalised landmarks.** `preprocessing.normalize_landmarks`
re-centres the hand on every frame, which is exactly what makes the classifier immune to
where the hand sits — and exactly what erases the motion we need here. So the tracker
takes raw pixel landmarks. Z subtracts its starting position and uses one median
hand size over the window, avoiding motion caused by per-frame scale changes.
J retains its original normalization. Distances are in "hand widths".

**It is a rule-based detector, not a learned model.** Training a temporal model needs
recorded sequences, and every ASL image dataset stores J and Z as single frames. The
thresholds below are the tunable part; run the translator with `--debug-motion` to see
the measured values live and adjust them to your camera and signing speed.
"""

from collections import deque

import numpy as np

WRIST = 0
INDEX_TIP = 8
PINKY_TIP = 20

# Handshapes the classifier is likely to report at the start or end of each gesture.
J_SHAPES = frozenset({"i", "j"})
Z_SHAPES = frozenset({"d", "z"})

# --- Tunables ------------------------------------------------------------------
# Sized for a signer moving deliberately rather than quickly. At 30 fps a fast gesture
# smears the hand across the frame badly enough that MediaPipe drops it, and the
# translator clears this buffer whenever the hand disappears, so a rushed J loses its own
# trajectory before it can be measured. Signing slowly avoids that, but then the gesture
# outlasts a short window, hence 1.5 s rather than 0.8 s.
HISTORY_FRAMES = 45          # ~1.5 s at 30 fps
MIN_PATH_LENGTH = 0.55       # total travel, in hand widths, before anything counts
# Per-frame speed above which the hand counts as "in motion", which suppresses letter
# commits. Do not lower this to try to catch slow gestures: measured against MediaPipe
# jitter on a hand held still, the two distributions overlap almost completely (a still
# hand medians ~0.011 with light jitter and ~0.023 with heavy jitter, a slow J ~0.023).
# Dropping to 0.018 leaves a still hand only a ~55% chance of holding the COMMIT_FRAMES
# consecutive frames a letter needs, which breaks spelling outright. Instantaneous speed
# simply cannot separate a slow gesture from a jittery still hand; path length over the
# window can, and `stats()['path']` is the number to use for that.
MOVING_THRESHOLD = 0.035
J_MIN_DESCENT = 0.20         # J drops before it hooks
J_MIN_HOOK = 0.12            # horizontal travel of the hook at the end
Z_MIN_REVERSALS = 2          # a Z has two corners
Z_MIN_HORIZONTAL = 0.30      # the strokes are mostly sideways
Z_MIN_DESCENT = 0.20         # diagonal connects the upper and lower strokes
Z_SHAPE_FRACTION = 0.80     # fallback shape must accompany the path, not just its end
# Raised with HISTORY_FRAMES to keep the same proportion of the window averaged; more
# frames of a slow gesture also means more idle wobble for `_reversals` to miscount.
SMOOTHING = 5                # frames averaged before measuring direction changes
# -------------------------------------------------------------------------------


def _reversals(values, deadzone):
    """Count sign changes in a sequence, ignoring wobble below `deadzone`."""
    sign = 0
    count = 0
    for value in values:
        if abs(value) < deadzone:
            continue
        current = 1 if value > 0 else -1
        if sign and current != sign:
            count += 1
        sign = current
    return count


def z_handshape(points):
    """Orientation-independent straight-index / folded-other-fingers check.

    Accepts pixel coordinates, preferably 3D with MediaPipe z scaled by image
    width like x. Chord/chain ratios distinguish a straight finger from a folded
    one; joint alignment also rejects an index bent mainly at one joint.
    """
    points = np.asarray(points, dtype=float)
    ratios = []
    alignment = []
    for base in (5, 9, 13, 17):
        joints = points[base:base + 4]
        bones = np.diff(joints, axis=0)
        lengths = np.linalg.norm(bones, axis=1)
        ratios.append(float(np.linalg.norm(joints[-1] - joints[0]) /
                            max(lengths.sum(), 1e-6)))
        alignment.append(float(min(
            np.dot(bones[i], bones[i + 1]) /
            max(lengths[i] * lengths[i + 1], 1e-6) for i in (0, 1))))
    index_extended = ratios[0] >= 0.85 and alignment[0] >= 0.65
    others_folded = all(r <= 0.70 for r in ratios[1:])
    index_axis = points[8] - points[5]
    return {"index_extended": index_extended, "others_folded": others_folded,
            "ratios": ratios, "alignment": alignment[0],
            "projection": float(np.linalg.norm(index_axis[:2]) /
                                max(np.linalg.norm(index_axis), 1e-6)),
            "angle": float(np.degrees(np.arctan2(index_axis[1], index_axis[0]))),
            "valid": index_extended and others_folded}


def _z_vertices(track):
    """Find horizontal turns by displacement, independent of per-frame speed.

    A turn is confirmed only after travelling Z_MIN_HORIZONTAL back from an
    extremum. Small jitter cannot manufacture corners, even at low frame speeds.
    """
    vertices = [track[0]]
    extreme = track[0]
    direction = 0
    for point in track[1:]:
        if direction == 0:
            dx = point[0] - extreme[0]
            if abs(dx) >= Z_MIN_HORIZONTAL:
                direction = 1 if dx > 0 else -1
                extreme = point
        elif direction * (point[0] - extreme[0]) >= 0:
            extreme = point
        elif direction * (point[0] - extreme[0]) <= -Z_MIN_HORIZONTAL:
            vertices.append(extreme)
            direction = -direction
            extreme = point
    vertices.append(extreme)
    return np.asarray(vertices)


def _z_strokes(track):
    return np.diff(_z_vertices(track), axis=0)


class MotionTracker:
    """Rolling window of hand positions, measured in hand widths."""

    def __init__(self, history=HISTORY_FRAMES):
        self.buffer = deque(maxlen=history)
        self.pixel_buffer = deque(maxlen=history)
        self.sizes = deque(maxlen=history)
        self.shapes = deque(maxlen=history)
        self.shape = None

    def reset(self):
        self.buffer.clear()
        self.pixel_buffer.clear()
        self.sizes.clear()
        self.shapes.clear()
        self.shape = None

    def update(self, points_px, handshape_points=None):
        """Feed raw pixel landmarks — not the normalised ones."""
        points_px = np.asarray(points_px, dtype=np.float32)
        extent = points_px.max(axis=0) - points_px.min(axis=0)
        hand_size = float(extent.max())
        if hand_size <= 0:
            return
        self.buffer.append(points_px / hand_size)
        self.pixel_buffer.append(points_px.copy())
        self.sizes.append(hand_size)
        self.shape = z_handshape(points_px if handshape_points is None else handshape_points)
        self.shapes.append(self.shape["valid"])

    def ready(self):
        return len(self.buffer) >= max(6, self.buffer.maxlen // 3)

    def _smoothed_track(self, index, pixels=False):
        frames = self.pixel_buffer if pixels else self.buffer
        track = np.stack([frame[index] for frame in frames])
        if SMOOTHING > 1 and len(track) >= SMOOTHING:
            # Pad with the edge values so smoothing keeps the trajectory's length and
            # its endpoints. A plain 'valid' convolution eats the first frames, which
            # drags the anchor below into the middle of the first stroke and quietly
            # under-reports every displacement measured from it.
            pad = SMOOTHING // 2
            padded = np.pad(track, ((pad, pad), (0, 0)), mode="edge")
            kernel = np.ones(SMOOTHING) / SMOOTHING
            track = np.stack([np.convolve(padded[:, 0], kernel, "valid"),
                              np.convolve(padded[:, 1], kernel, "valid")], axis=1)[:len(track)]
        return track

    def _trajectory(self, index):
        # Z uses one common scale AFTER pixel-space smoothing. Dividing absolute
        # coordinates by a different size each frame invents motion as size varies.
        track = self._smoothed_track(index, pixels=index == INDEX_TIP)
        if index == INDEX_TIP:
            track = track / float(np.median(self.sizes))
        return track - track[0]

    def z_shape_ready(self):
        return bool(self.ready() and self.shape and self.shape["valid"]
                    and np.mean(self.shapes) >= Z_SHAPE_FRACTION)

    def z_trace(self):
        """Smoothed fingertip path and confirmed-turn vertices in camera pixels."""
        if not self.pixel_buffer:
            return np.empty((0, 2)), np.empty((0, 2))
        track = self._smoothed_track(INDEX_TIP, pixels=True)
        scale = float(np.median(self.sizes))
        vertices = _z_vertices((track - track[0]) / scale) * scale + track[0]
        return track, vertices

    def speed(self):
        if len(self.buffer) < 2:
            return 0.0
        recent = np.stack([f[WRIST] for f in list(self.buffer)[-4:]])
        return float(np.linalg.norm(np.diff(recent, axis=0), axis=1).mean())

    def is_moving(self):
        return self.speed() > MOVING_THRESHOLD

    def stats(self, index):
        """Measurements for one fingertip's recent path."""
        if not self.ready():
            return None

        track = self._trajectory(index)
        steps = np.diff(track, axis=0)
        if len(steps) == 0:
            return None

        path_length = float(np.linalg.norm(steps, axis=1).sum())
        deadzone = max(path_length / 40.0, 1e-3)
        strokes = _z_strokes(track) if index == INDEX_TIP else None
        z_geometry = False
        if strokes is not None and len(strokes) == 3:
            dx, dy = strokes[:, 0], strokes[:, 1]
            z_geometry = bool(
                np.all(np.abs(dx) >= Z_MIN_HORIZONTAL)
                and dx[0] * dx[1] < 0 and dx[1] * dx[2] < 0
                and dy[1] >= Z_MIN_DESCENT
                and abs(dy[0]) <= abs(dx[0]) * 0.5
                and abs(dy[2]) <= abs(dx[2]) * 0.5)

        return {
            "path": path_length,
            "net": float(np.linalg.norm(track[-1])),
            # Image coordinates grow downward, so a positive dy is a descent.
            "descent": float(track[:, 1].max() - track[0, 1]),
            # Total width of the stroke, which is what a Z's zigzag makes large — and
            # stays meaningful even when the window opens mid-gesture.
            "horizontal": float(track[:, 0].max() - track[:, 0].min()),
            "hook": float(abs(track[-1, 0] - track[np.argmax(track[:, 1]), 0])),
            "reversals": (max(0, len(strokes) - 1) if strokes is not None
                          else _reversals(steps[:, 0], deadzone)),
            "z_geometry": z_geometry,
        }

    def classify(self, static_label):
        """Resolve motion using CNN shape labels or a sustained landmark Z shape."""
        label = (static_label or "").lower()

        if label in J_SHAPES:
            s = self.stats(PINKY_TIP)
            if s and s["path"] >= MIN_PATH_LENGTH and s["descent"] >= J_MIN_DESCENT and s["hook"] >= J_MIN_HOOK:
                return "j"
            # An 'i' that never moved is a genuine 'i'.
            return "i" if label == "j" and s and s["path"] < MIN_PATH_LENGTH else None

        if label in Z_SHAPES or self.z_shape_ready():
            s = self.stats(INDEX_TIP)
            if (s and s["path"] >= MIN_PATH_LENGTH and s["reversals"] >= Z_MIN_REVERSALS
                    and s["horizontal"] >= Z_MIN_HORIZONTAL and s["z_geometry"]):
                return "z"
            return "d" if label == "z" and s and s["path"] < MIN_PATH_LENGTH else None

        return None

    def debug_lines(self, static_label):
        """Live readout of the measured values, for tuning the thresholds above."""
        label = (static_label or "").lower()
        index, target = (PINKY_TIP, "J") if label in J_SHAPES else (INDEX_TIP, "Z")
        s = self.stats(index)
        if not s:
            return [f"motion: filling buffer ({len(self.buffer)})"]
        if target == 'Z':
            return [
                f"target Z label {label} gate {'yes' if label in Z_SHAPES or self.z_shape_ready() else 'blocked'}",
                f"CNN gate {'yes' if label in Z_SHAPES else 'no'}; landmark {'yes' if self.z_shape_ready() else 'no'}",
                f"path {s['path']:.2f}/{MIN_PATH_LENGTH} span {s['horizontal']:.2f}/{Z_MIN_HORIZONTAL}",
                f"corners {s['reversals']}/2 geometry {'pass' if s['z_geometry'] else 'fail'}",
            ] + self._z_debug() + [
                f"resolved {self.classify(label) or '-'} speed {self.speed():.3f}/{MOVING_THRESHOLD}"]
        return [
            f"target {target} label {label} gate {'yes' if label in J_SHAPES | Z_SHAPES or self.z_shape_ready() else 'blocked'}",
            f"speed {self.speed():.3f} / {MOVING_THRESHOLD} (commit)",
            f"path       {s['path']:.2f} / {MIN_PATH_LENGTH}",
            f"descent    {s['descent']:.2f} / {J_MIN_DESCENT}",
            f"hook       {s['hook']:.2f} / {J_MIN_HOOK}",
            f"horizontal {s['horizontal']:.2f} / {Z_MIN_HORIZONTAL}",
            f"reversals  {s['reversals']} / {Z_MIN_REVERSALS}",
            f"Z geometry {'pass' if s['z_geometry'] else 'fail'}; resolved {self.classify(label) or '-'}",
        ]

    def _z_debug(self):
        shape = self.shape
        strokes = _z_strokes(self._trajectory(INDEX_TIP))
        return [
            f"index {'straight' if shape['index_extended'] else 'bent'} "
            f"r {shape['ratios'][0]:.2f} a {shape['alignment']:.2f}",
            f"M/R/P {'folded' if shape['others_folded'] else 'OPEN'} "
            + '/'.join(f'{r:.2f}' for r in shape['ratios'][1:]),
            f"index angle {shape['angle']:.0f} plane {shape['projection']:.2f}",
            f"shape {np.mean(self.shapes):.0%}/{Z_SHAPE_FRACTION:.0%} size {np.median(self.sizes):.0f}px",
        ] + [f"{i + 1}: {'R' if dx > 0 else 'L'} dx {dx:+.2f} dy {dy:+.2f}"
             for i, (dx, dy) in enumerate(strokes[:4])]
