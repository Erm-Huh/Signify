"""Committed recognised text, kept separate from whatever is currently on screen.

Items are words, letters or spaces. Letters join into words; words are space-separated:

    [WORD hello] [LETTER M] [LETTER Y]  ->  "HELLO MY"

Duplicate rules:
* An automatic word identical to the previous committed item is never appended (same
  principle as upstream's letters: no automatic repeat without a deliberate action). A
  word sign held or repeated across sliding windows would otherwise be typed again and
  again; to repeat a word on purpose, confirm it manually (ENTER / W).
* Automatic letters rely on upstream's rule (no same letter twice in a row without a
  break) and are additionally dropped if identical to the previous letter within
  ``duplicate_letter_window_s``.
* Manual commits (a keypress) are intentional and always accepted, which is how double
  letters are typed (upstream's ``r`` = repeat last letter).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    kind: str  # "word" | "letter" | "space"
    text: str
    t: float
    manual: bool


@dataclass
class TextBuffer:
    duplicate_letter_window_s: float = 1.0
    max_items: int = 200
    items: list[Item] = field(default_factory=list)

    def _push(self, item: Item) -> bool:
        self.items.append(item)
        if len(self.items) > self.max_items:
            del self.items[0]
        return True

    def _last(self, kind: str | None = None) -> Item | None:
        for item in reversed(self.items):
            if kind is None or item.kind == kind:
                return item
        return None

    def add_word(self, word: str, t: float, *, manual: bool = False) -> bool:
        word = word.strip().upper()
        if not word:
            return False
        last = self._last()
        if not manual and last is not None and last.kind == "word" and last.text == word:
            return False
        return self._push(Item("word", word, t, manual))

    def add_letter(self, letter: str, t: float, *, manual: bool = False) -> bool:
        letter = letter.strip().upper()
        if len(letter) != 1 or not letter.isalpha():
            return False
        last = self._last()
        if (not manual and last is not None and last.kind == "letter" and last.text == letter
                and t - last.t < self.duplicate_letter_window_s):
            return False
        return self._push(Item("letter", letter, t, manual))

    def repeat_last_letter(self, t: float) -> bool:
        """Upstream 'r': intentionally double the last letter (e.g. HELLO)."""
        last = self._last()
        if last is None or last.kind != "letter":
            return False
        return self._push(Item("letter", last.text, t, True))

    def add_space(self, t: float) -> bool:
        if not self.items or self.items[-1].kind == "space":
            return False
        return self._push(Item("space", " ", t, True))

    def trailing_letters(self) -> str:
        """The run of letters at the end of the text ("" if the last item isn't a letter)."""
        out = []
        for item in reversed(self.items):
            if item.kind != "letter":
                break
            out.append(item.text)
        return "".join(reversed(out))

    def split_trailing_run(self, split) -> bool:  # noqa: ANN001  (str -> list[str])
        """Intelligent spacing: space the closing letter run into words with ``split``
        (see ``smart_spacing.split_run``). Each letter stays its own item, so backspace
        still removes one letter (or one inserted space) at a time."""
        run = self.trailing_letters()
        pieces = split(run) if run else [run]
        if len(pieces) <= 1:
            return False
        start = len(self.items) - len(run)
        letters, rebuilt = self.items[start:], []
        k = 0
        for n, piece in enumerate(pieces):
            if n:
                rebuilt.append(Item("space", " ", letters[k].t, False))
            rebuilt.extend(letters[k:k + len(piece)])
            k += len(piece)
        self.items[start:] = rebuilt
        return True

    def delete_last(self) -> Item | None:
        return self.items.pop() if self.items else None

    def clear(self) -> None:
        self.items.clear()

    @property
    def text(self) -> str:
        out = ""
        prev: Item | None = None
        for item in self.items:
            if item.kind == "space":
                out = out.rstrip(" ") + " "
            elif item.kind == "word":
                if out and not out.endswith(" "):
                    out += " "
                out += item.text
            else:  # letter
                if prev is not None and prev.kind == "word":
                    out += " "
                out += item.text
            prev = item
        return out.strip()
