"""Conservative decision layer between the word model and the alphabet model.

The two models have different output spaces, training data and calibration:

* Word (GestureSign): softmax with a temperature fitted on held-out signers; "confident"
  means smoothed probability >= the model's calibrated ``notSureBelow`` (0.29) and the
  signer in frame. Upstream: 77% of answers above it are correct on held-out signers.
* Letter (alphabet CNN): plain softmax, not calibrated; "confident" means the upstream
  ``COMMIT_CONFIDENCE`` (0.75) on the 8-frame average. Upstream: 96.7% precision at it on
  its held-out photo test set.

So the router NEVER compares a word score with a letter score and never produces a
combined confidence. Each model is judged only against its own threshold, and:

=============  ==============================================================================
AUTO           word ok & letter ok -> AMBIGUOUS (both shown, nothing auto-committed)
               word ok only -> WORD;  letter ok only -> LETTER
               neither -> NO_HANDS if no fresh source sees a hand, else NOT_SURE
WORD           word ok -> WORD, else NO_HANDS / NOT_SURE (letters only in the details)
SPELL          letter ok -> LETTER, else NO_HANDS / NOT_SURE (words only in the details)
=============  ==============================================================================

"ok" = fresh (newer than ``*_max_age_s``) and above that model's own threshold. Stability
(word: ``min_stable_windows`` agreeing windows; letter: upstream's 10 still frames) is
reported separately: a WORD/LETTER that isn't stable yet is shown as *tentative*, and only
stable results can be committed to the text automatically (see ``auto_commit``).

There is no validated way to tell a fingerspelled letter from a word sign automatically,
so AUTO shows AMBIGUOUS instead of picking a winner; WORD / SPELL and the manual keys are
the fallback. Every threshold lives in ``RouterConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.live import LiveDecision
from app.models.alphabet_recognizer import AlphabetResult


class Mode(str, Enum):
    AUTO = "AUTO"
    WORD = "WORD"
    SPELL = "SPELL"

    def next(self) -> Mode:
        order = [Mode.AUTO, Mode.WORD, Mode.SPELL]
        return order[(order.index(self) + 1) % len(order)]


class Kind(str, Enum):
    WORD = "WORD"
    LETTER = "LETTER"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_SURE = "NOT_SURE"
    NO_HANDS = "NO_HANDS"


@dataclass(frozen=True)
class RouterConfig:
    # A word decision describes a window ending at ``t_end``; older than this = stale.
    # Default: 0.5 s scheduling interval + ~0.15-0.2 s measured lag + margin.
    word_max_age_s: float = 1.5
    # Alphabet results are per frame (~20/s measured); older than this = stale.
    letter_max_age_s: float = 0.5
    # Require stability before showing a result at all (otherwise shown as tentative).
    require_stable_for_display: bool = False


@dataclass(frozen=True)
class ModelView:
    """One model's latest output as the UI should present it."""

    available: bool = False  # a result exists
    fresh: bool = False
    age_s: float | None = None
    hands: bool = False
    ok: bool = False  # fresh and above this model's own threshold
    stable: bool = False
    label: str | None = None
    score: float = 0.0  # in this model's own units; never compared across models
    top: list[tuple[str, float]] = field(default_factory=list)
    note: str = ""


@dataclass(frozen=True)
class RouterDecision:
    mode: Mode
    kind: Kind
    label: str | None
    stable: bool  # False -> tentative
    reason: str
    word: ModelView
    letter: ModelView
    alternatives: list[tuple[Kind, str]] = field(default_factory=list)


def word_view(d: LiveDecision | None, now: float, cfg: RouterConfig) -> ModelView:
    if d is None or d.state == "waiting" or d.decided_at <= 0:
        return ModelView(note="waiting")
    age = now - d.t_end
    fresh = age <= cfg.word_max_age_s
    hands = d.state in ("sign", "not_sure")
    ok = fresh and d.state == "sign" and (d.stable or not cfg.require_stable_for_display)
    label = d.label if d.state == "sign" else (d.top[0][0] if d.top else None)
    note = "stale" if not fresh else (d.reason or d.state)
    return ModelView(True, fresh, age, hands and fresh, ok, d.stable and ok, label,
                     d.confidence, list(d.top), note)


def letter_view(r: AlphabetResult | None, now: float, cfg: RouterConfig) -> ModelView:
    if r is None:
        return ModelView(note="waiting")
    age = now - r.timestamp
    fresh = age <= cfg.letter_max_age_s
    hands = r.hand_detected and fresh
    ok = hands and r.passes_threshold and (r.stable or not cfg.require_stable_for_display)
    if not fresh:
        note = "stale"
    elif not r.hand_detected:
        note = "no_hand"
    elif r.moving and not r.motion_override:
        note = "moving"
    elif not r.passes_threshold:
        note = "low_confidence"
    else:
        note = "stable" if r.stable else "holding"
    return ModelView(True, fresh, age, hands, ok, r.stable and ok, r.best, r.confidence,
                     [(g.label, g.confidence) for g in r.top], note)


def decide(
    mode: Mode,
    word: LiveDecision | None,
    letter: AlphabetResult | None,
    now: float,
    cfg: RouterConfig = RouterConfig(),
) -> RouterDecision:
    w, ltr = word_view(word, now, cfg), letter_view(letter, now, cfg)
    any_hands = w.hands or ltr.hands

    def none(reason: str) -> RouterDecision:
        kind = Kind.NOT_SURE if any_hands else Kind.NO_HANDS
        return RouterDecision(mode, kind, None, False, reason if any_hands else "no hands visible", w, ltr)

    if mode is Mode.WORD:
        if w.ok:
            return RouterDecision(mode, Kind.WORD, w.label, w.stable, "word model above its calibrated threshold", w, ltr)
        return none(f"word model: {w.note}")
    if mode is Mode.SPELL:
        if ltr.ok:
            return RouterDecision(mode, Kind.LETTER, ltr.label, ltr.stable, "letter model above its threshold", w, ltr)
        return none(f"letter model: {ltr.note}")

    # AUTO
    if w.ok and ltr.ok:
        return RouterDecision(
            mode, Kind.AMBIGUOUS, None, False,
            "both models are confident and there is no validated way to choose; "
            "confirm one (W / L) or switch mode (TAB)",
            w, ltr, [(Kind.WORD, w.label or "?"), (Kind.LETTER, ltr.label or "?")],
        )
    if w.ok:
        return RouterDecision(mode, Kind.WORD, w.label, w.stable, "only the word model is confident", w, ltr)
    if ltr.ok:
        return RouterDecision(mode, Kind.LETTER, ltr.label, ltr.stable, "only the letter model is confident", w, ltr)
    return none(f"word: {w.note}; letter: {ltr.note}")


def auto_commit(
    decision: RouterDecision,
    word_emitted: str | None,
    letter_committed: str | None,
) -> tuple[Kind, str] | None:
    """What may be appended to the text WITHOUT a keypress. ``word_emitted`` /
    ``letter_committed`` are the models' own stability+cooldown events (LiveRecognizer
    emission; upstream SpellingBuffer commit). Nothing is appended for AMBIGUOUS,
    NOT_SURE or NO_HANDS, and an event only counts if the router currently shows that
    same label as a stable result of that kind."""
    if decision.kind is Kind.WORD and word_emitted and decision.stable and word_emitted == decision.label:
        return (Kind.WORD, word_emitted)
    if decision.kind is Kind.LETTER and letter_committed and letter_committed == decision.label:
        # upstream's commit rule already required 10 still frames >= 0.75
        return (Kind.LETTER, letter_committed)
    return None
