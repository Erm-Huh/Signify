"""Recognition router, text buffer and speech: pure logic, no models or camera."""

from __future__ import annotations

import time

import pytest

from app.live import LiveDecision
from app.models.alphabet_recognizer import AlphabetResult, LetterGuess
from app.services.recognition_router import Kind, Mode, RouterConfig, auto_commit, decide
from app.services.speech import Speaker
from app.services.text_buffer import TextBuffer
from app.services.unified_session import ManualResult, manual_override

NOW = 100.0
CFG = RouterConfig(word_max_age_s=1.5, letter_max_age_s=0.5)


def word(state: str = "sign", label: str | None = "hello", conf: float = 0.8, stable: bool = True,
         age: float = 0.3, emitted: str | None = None) -> LiveDecision:
    top = [("hello", conf), ("all", 0.05), ("open", 0.01)]
    return LiveDecision(state=state, label=label if state == "sign" else None, confidence=conf, top=top,
                        reason="" if state == "sign" else ("low_confidence" if state == "not_sure" else "no_hands_detected"),
                        just_emitted=emitted, stable=stable, t_end=NOW - age, decided_at=NOW - age + 0.2)


def letter(best: str | None = "A", conf: float = 0.9, stable: bool = True, age: float = 0.05,
           hand: bool = True, committed: str | None = None) -> AlphabetResult:
    if not hand:
        return AlphabetResult(NOW - age, NOW - age, hand_detected=False)
    return AlphabetResult(
        timestamp=NOW - age, computed_at=NOW - age, hand_detected=True,
        top=[LetterGuess(best or "A", conf), LetterGuess("S", 0.05), LetterGuess("T", 0.02)],
        best=best, confidence=conf, static_best=best, passes_threshold=conf >= 0.75,
        uncertain=conf <= 0.5, stable=stable, progress=1.0 if stable else 0.5, committed=committed,
    )


NOT_SURE_WORD = word("not_sure", conf=0.2, stable=False)
IDLE_WORD = word("idle", conf=0.0, stable=False)
WEAK_LETTER = letter(conf=0.6, stable=False)


# --------------------------------------------------------------------------- routing table


@pytest.mark.parametrize(
    ("mode", "w", "ltr", "kind", "label"),
    [
        (Mode.AUTO, word(), WEAK_LETTER, Kind.WORD, "hello"),
        (Mode.AUTO, NOT_SURE_WORD, letter(), Kind.LETTER, "A"),
        (Mode.AUTO, word(), letter(), Kind.AMBIGUOUS, None),
        (Mode.AUTO, NOT_SURE_WORD, WEAK_LETTER, Kind.NOT_SURE, None),
        (Mode.AUTO, IDLE_WORD, letter(hand=False), Kind.NO_HANDS, None),
        (Mode.WORD, word(), letter(), Kind.WORD, "hello"),  # letters ignored for the primary
        (Mode.WORD, NOT_SURE_WORD, letter(), Kind.NOT_SURE, None),
        (Mode.SPELL, word(), letter(), Kind.LETTER, "A"),  # words ignored for the primary
        (Mode.SPELL, word(), WEAK_LETTER, Kind.NOT_SURE, None),
        (Mode.SPELL, IDLE_WORD, letter(hand=False), Kind.NO_HANDS, None),
    ],
)
def test_routing_table(mode, w, ltr, kind, label) -> None:
    d = decide(mode, w, ltr, NOW, CFG)
    assert d.kind is kind and d.label == label


def test_each_model_uses_only_its_own_threshold() -> None:
    # a word at 0.30 passes the word model's calibrated 0.29; a letter at 0.74 fails the
    # alphabet's 0.75 even though 0.74 > 0.30 -- scores are never compared across models
    d = decide(Mode.AUTO, word(conf=0.30), letter(conf=0.74, stable=False), NOW, CFG)
    assert d.kind is Kind.WORD and d.label == "hello"
    d = decide(Mode.AUTO, NOT_SURE_WORD, letter(conf=0.99), NOW, CFG)
    assert d.kind is Kind.LETTER


def test_ambiguous_offers_both_and_never_picks() -> None:
    for wc, lc in ((0.95, 0.76), (0.31, 0.99)):  # regardless of which score is "bigger"
        d = decide(Mode.AUTO, word(conf=wc), letter(conf=lc), NOW, CFG)
        assert d.kind is Kind.AMBIGUOUS and d.label is None and not d.stable
        assert d.alternatives == [(Kind.WORD, "hello"), (Kind.LETTER, "A")]
        assert auto_commit(d, "hello", "A") is None  # nothing auto-committed


def test_stale_results_are_ignored_and_flagged() -> None:
    d = decide(Mode.AUTO, word(age=2.0), letter(age=0.05), NOW, CFG)
    assert d.kind is Kind.LETTER and not d.word.fresh and d.word.note == "stale"
    d = decide(Mode.AUTO, word(age=0.3), letter(age=0.8), NOW, CFG)
    assert d.kind is Kind.WORD and not d.letter.fresh and d.letter.note == "stale"
    d = decide(Mode.AUTO, word(age=3.0), letter(age=3.0), NOW, CFG)
    assert d.kind is Kind.NO_HANDS  # stale hands don't count as hands
    assert d.word.age_s == pytest.approx(3.0) and d.letter.age_s == pytest.approx(3.0)


def test_unstable_results_are_tentative_and_not_auto_committed() -> None:
    d = decide(Mode.WORD, word(stable=False, emitted=None), None, NOW, CFG)
    assert d.kind is Kind.WORD and not d.stable
    assert auto_commit(d, None, None) is None
    strict = RouterConfig(require_stable_for_display=True)
    assert decide(Mode.WORD, word(stable=False), None, NOW, strict).kind is Kind.NOT_SURE


def test_auto_commit_rules() -> None:
    d = decide(Mode.WORD, word(emitted="hello"), None, NOW, CFG)
    assert auto_commit(d, "hello", None) == (Kind.WORD, "hello")
    assert auto_commit(d, "bye", None) is None  # event must match what is shown
    d = decide(Mode.SPELL, None, letter(committed="A"), NOW, CFG)
    assert auto_commit(d, None, "A") == (Kind.LETTER, "A")
    d = decide(Mode.AUTO, NOT_SURE_WORD, WEAK_LETTER, NOW, CFG)
    assert auto_commit(d, "hello", "A") is None


def test_manual_capture_override() -> None:
    base = decide(Mode.AUTO, NOT_SURE_WORD, letter(), NOW, CFG)
    ok = manual_override(base, ManualResult("thank you", 0.8, True, [("thank you", 0.8)], "", 0, NOW))
    assert ok.kind is Kind.WORD and ok.label == "thank you" and ok.stable
    low = manual_override(base, ManualResult("napkin", 0.2, False, [("napkin", 0.2)], "low_confidence", 0, NOW))
    assert low.kind is Kind.NOT_SURE and low.label is None
    none = manual_override(base, ManualResult(None, 0.0, False, [], "no_hands_detected", 0, NOW))
    assert none.kind is Kind.NO_HANDS


def test_mode_cycle() -> None:
    assert [Mode.AUTO.next(), Mode.WORD.next(), Mode.SPELL.next()] == [Mode.WORD, Mode.SPELL, Mode.AUTO]


# --------------------------------------------------------------------------- text buffer


def test_text_buffer_rendering_and_editing() -> None:
    tb = TextBuffer()
    assert tb.add_word("hello", 0) and tb.add_letter("m", 1) and tb.add_letter("y", 2)
    assert tb.text == "HELLO MY"
    tb.add_space(3)
    tb.add_letter("b", 4)
    assert tb.text == "HELLO MY B"
    tb.add_word("thank you", 5)
    assert tb.text == "HELLO MY B THANK YOU"
    assert tb.delete_last().text == "THANK YOU"
    assert tb.text == "HELLO MY B"
    tb.clear()
    assert tb.text == "" and tb.items == []
    assert not tb.add_letter("1", 0) and not tb.add_letter("ab", 0) and not tb.add_word("  ", 0)
    assert not tb.add_space(0)  # nothing to separate yet


def test_text_buffer_debounces_auto_but_keeps_intentional_repeats() -> None:
    tb = TextBuffer(duplicate_letter_window_s=1.0)
    assert tb.add_word("hello", 0)
    assert not tb.add_word("hello", 10)  # automatic repeat of the previous word: never
    assert tb.add_word("hello", 10.5, manual=True)  # deliberate repeat: allowed
    assert tb.text == "HELLO HELLO"
    tb.clear()
    assert tb.add_letter("l", 0) and not tb.add_letter("l", 0.5)  # bounce
    assert tb.add_letter("l", 2.0)  # later and upstream re-armed: allowed
    assert tb.repeat_last_letter(2.1)  # upstream 'r': intentional double letter
    assert tb.text == "LLL"


# --------------------------------------------------------------------------- speech


def test_speech_is_non_blocking_and_not_repeated_while_busy() -> None:
    sp = Speaker(command=["sleep"])  # "sleep 2" stands in for `say <text>`
    t = time.perf_counter()
    assert sp.speak("2") == "spoken"
    assert time.perf_counter() - t < 0.5  # returned immediately
    assert sp.busy and sp.speak("2") == "busy"
    assert sp.speak("   ") == "empty"
    sp.close()
    assert not sp.busy
    assert Speaker(command=[]).speak("hi") == "unavailable"
