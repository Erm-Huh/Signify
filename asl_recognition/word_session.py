"""Live word coordination shared by the main app and standalone demo.

Quality voting follows GestureSign recognizer.ts, MIT, Copyright (c) 2026
Shashank Yadav. Full notice and attribution: WORD_RECOGNITION.md.
"""

from word_segmenter import Segmenter


class LiveWordSession:
    """Synchronous segment/quality coordinator; inference only on completed signs.

    Mirrors recognizer.ts's quality vote over ok/partial frames during capture.
    Missing-hand frames don't count against framing. No video or transcript is
    saved, and no result is fabricated when a segment is discarded/reset.
    """

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.segmenter = Segmenter()
        self.reset()

    def reset(self):
        self.segmenter.reset()
        self.statuses = []
        self.result = None
        self.last_event = "waiting for motion"
        self.completed = 0

    def push(self, tracked, t):
        event = self.segmenter.push(tracked.frame, t, crowded=tracked.status == "crowded")
        if self.segmenter.state == "signing":
            self.statuses.append(tracked.status)
        if event is None:
            return None, None
        if event.type == "start":
            self.statuses = []
            self.last_event = "recording"
            return event, None
        statuses, self.statuses = self.statuses, []
        self.last_event = f"{event.type}: {event.reason}"
        if event.type == "discard":
            return event, None
        framed = sum(status in ("ok", "partial") for status in statuses)
        partial = statuses.count("partial") > framed / 2
        segment = event.segment
        self.result = self.recognizer.recognize(segment.frames, segment.fps, partial=partial)
        self.completed += 1
        return event, self.result

