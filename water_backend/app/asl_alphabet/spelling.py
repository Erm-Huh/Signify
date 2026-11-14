"""Constants and ``SpellingBuffer`` extracted VERBATIM from upstream ``realtime_translator.py``
(commit 45ff2c0e26925c50f4546073bd573fe3ca7a5e6e). That script also imports MediaPipe's GUI
helpers and ``ui``, so only these pure-Python pieces were copied. Do not edit; see
``tests/test_alphabet.py::test_vendored_files_match_upstream``.
"""

LOGITS = "logits"
CONV_TAP = "conv1_relu"
DENSE_TAP = "dense_features"

CONFIDENCE_FLOOR = 0.50      # below this the reading is reported as uncertain
COMMIT_CONFIDENCE = 0.75     # and this much is needed to append a letter
COMMIT_FRAMES = 10           # consecutive agreeing frames before committing
SMOOTHING_WINDOW = 8         # rolling average over recent probability vectors


class SpellingBuffer:
    """Turns a stream of per-frame guesses into typed text.

    A letter is committed once it has been the top prediction for COMMIT_FRAMES frames
    in a row while the hand is still. The same letter will not be committed twice in a
    row — otherwise a hand held steady would spray repeats — so double letters need the
    'r' key, or a brief move away and back.
    """

    def __init__(self):
        self.text = ""
        self._candidate = None
        self._streak = 0
        self._last_committed = None

    def update(self, letter, confidence, hand_present, moving):
        if not hand_present:
            self._candidate = None
            self._streak = 0
            self._last_committed = None
            return None

        if moving:
            # Mid-gesture frames are meaningless for a static classifier.
            self._streak = 0
            return None

        if letter != self._candidate:
            self._candidate = letter
            self._streak = 1
            if letter != self._last_committed:
                self._last_committed = None
        else:
            self._streak += 1

        if (self._streak == COMMIT_FRAMES and confidence >= COMMIT_CONFIDENCE
                and letter != self._last_committed):
            self.text += letter.upper()
            self._last_committed = letter
            return letter
        return None

    def progress(self):
        return min(self._streak / COMMIT_FRAMES, 1.0)

    def repeat(self):
        if self.text and self.text[-1] != " ":
            self.text += self.text[-1]

    def space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "
        self._last_committed = None

    def backspace(self):
        self.text = self.text[:-1]
        self._last_committed = None

    def clear(self):
        self.text = ""
        self._last_committed = None
