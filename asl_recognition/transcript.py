"""Shared text and reversible edit history; independent of recognition models."""


class Transcript:
    def __init__(self):
        self.text = ""
        self.latest = ""
        self._after_word = False
        self._history = []

    def _set(self, text, after_word, latest=None):
        if (text == self.text and after_word == self._after_word
                and (latest is None or latest == self.latest)):
            return
        self._history.append((self.text, self._after_word, self.latest))
        self.text, self._after_word = text, after_word
        if latest is not None:
            self.latest = latest

    def append_letter(self, letter):
        separator = " " if self._after_word and self.text and not self.text.endswith(" ") else ""
        self._set(self.text + separator + letter.upper(), False, letter.upper())

    def append_word(self, word):
        word = " ".join(word.split())
        if word:
            separator = " " if self.text and not self.text.endswith(" ") else ""
            self._set(self.text + separator + word, True, word)

    def space(self):
        if self.text and not self.text.endswith(" "):
            self._set(self.text + " ", False)

    def backspace(self):
        if self.text:
            self._set(self.text[:-1], False)

    def clear(self):
        self._set("", False, "")

    def repeat(self):
        # Preserve the original r key's character-repeat behavior.
        if self.text and self.text[-1] != " ":
            self._set(self.text + self.text[-1], False, self.text[-1])

    def undo(self):
        if self._history:
            self.text, self._after_word, self.latest = self._history.pop()
