"""Mode, transcript, and commit ownership for the combined application.

No camera, GUI, model loading, or tracking graph initialization occurs here.
"""

import math

from transcript import Transcript
from word_segmenter import hand_speed
from word_tracking import hands_missing


class RecognitionSession:
    def __init__(self, spelling, motion, smoother, score_history, word_confidence=.75):
        if not math.isfinite(word_confidence) or not 0 <= word_confidence <= 1:
            raise ValueError("Word commit confidence must be between 0 and 1")
        self.spelling, self.motion, self.smoother = spelling, motion, smoother
        self.score_history = score_history
        self.word_confidence = word_confidence
        self.transcript = Transcript()
        self.mode = "letter"
        self.generation = 0
        self.word_session = None
        self.status = "Ready"
        self._consumed = set()
        self._cooldown = False
        self._last_word_frame = None
        self._quiet_since = None
        self._release_speed = 0.0

    def switch_mode(self, mode):
        if mode not in ("letter", "word"):
            raise ValueError("Unknown recognition mode")
        if mode == self.mode:
            return False
        self.generation += 1
        self.mode = mode
        self.spelling.reset_pending()
        self.motion.reset()
        self.smoother.reset()
        self.score_history.clear()
        if self.word_session is not None:
            self.word_session.reset()
        self._consumed.clear()
        self._cooldown = False
        self._last_word_frame = None
        self._quiet_since = None
        self._release_speed = 0.0
        self.status = "Ready"
        return True

    def commit_letter(self, letter):
        if self.mode == "letter" and letter:
            self.transcript.append_letter(letter)

    def commit_word(self, result, segment_id, generation):
        key = (generation, segment_id)
        if self.mode != "word" or generation != self.generation or key in self._consumed:
            return False
        # Consume rejected results too; an edit/undo must never replay an old result.
        self._consumed.add(key)
        if not result.accepted or not result.top:
            self.status = f"Not recorded: {result.reason or 'rejected'}"
            return False
        top = result.top[0]
        if not math.isfinite(top.confidence) or top.confidence < self.word_confidence:
            self.status = f"Not recorded: confidence below {self.word_confidence:.2f}"
            return False
        self.transcript.append_word(top.label)
        self.status = f"Recorded: {top.label}"
        return True

    def push_word(self, tracked, t):
        if self.mode != "word" or self.word_session is None:
            return None, None
        if self._cooldown:
            # A timeout is not a gesture boundary. Don't turn a continued long
            # gesture into repeated automatic commits; wait for a real quiet gap.
            previous = self._last_word_frame
            if previous is not None:
                self._release_speed = .5 * self._release_speed + .5 * hand_speed(
                    previous[0], tracked.frame, t - previous[1])
            quiet = hands_missing(tracked.frame) or (
                previous is not None and self._release_speed < .6)
            self._last_word_frame = (tracked.frame.copy(), t)
            self._quiet_since = (t if self._quiet_since is None else self._quiet_since) if quiet else None
            self.status = "Finish gesture: pause or lower hands"
            if self._quiet_since is not None and t - self._quiet_since >= 300:
                self._cooldown = False
                self.word_session.reset()
                self.status = "Ready for next sign"
            return None, None
        generation = self.generation
        event, result = self.word_session.push(tracked, t)
        if generation != self.generation or self.mode != "word":
            return None, None
        if event is not None and event.type == "discard":
            self.status = f"Not recorded: {event.reason}"
        elif event is not None and event.type == "start":
            self.status = "Recording sign"
        if result is not None:
            segment_id = (event.segment.timestamps[0], event.segment.timestamps[-1])
            self.commit_word(result, segment_id, generation)
        if event is not None and event.type == "end" and event.reason == "max":
            self._cooldown = True
            self._last_word_frame = (tracked.frame.copy(), t)
            self._quiet_since = None
            self._release_speed = self.word_session.segmenter.speed
            self.word_session.reset()
        return event, result
