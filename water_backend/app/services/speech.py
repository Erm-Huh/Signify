"""Optional, non-blocking local text-to-speech.

Uses macOS's built-in ``say`` when present (no extra dependency). Speech runs in a child
process, so it can never block the camera loop. Only explicit calls speak; there is no
automatic per-frame speech. While one utterance is playing, further requests are refused
rather than queued, so nothing piles up.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


class Speaker:
    def __init__(self, command: list[str] | None = None) -> None:
        if command is None:
            say = shutil.which("say") if sys.platform == "darwin" else None
            command = [say] if say else []
        self.command = command
        self._proc: subprocess.Popen[bytes] | None = None
        self.last_spoken: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.command)

    @property
    def busy(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def speak(self, text: str) -> str:
        """Returns 'spoken', 'busy', 'empty' or 'unavailable'. Never blocks."""
        text = text.strip()
        if not self.available:
            return "unavailable"
        if not text:
            return "empty"
        if self.busy:
            return "busy"
        self._proc = subprocess.Popen(
            [*self.command, text], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.last_spoken = text
        return "spoken"

    def close(self, timeout: float = 1.0) -> None:
        if self.busy:
            assert self._proc is not None
            self._proc.terminate()
            try:
                self._proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
