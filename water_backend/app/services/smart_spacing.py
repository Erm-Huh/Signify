"""Intelligent spacing for fingerspelled runs.

When a run of letters is closed (the signer lowers their hand, presses Space, or signs a
word), ``split_run("HELLOMYNAME")`` returns ``["HELLO", "MY", "NAME"]`` if — and only if —
the whole run can be covered by known words. It is deliberately conservative:

* a run that is itself a known word is never split ("NOTEBOOK" stays whole if listed);
* a run that cannot be covered completely is left untouched, so names and rare words
  ("OLVERA", "JONATHAN") are never chopped into pieces;
* pieces must be at least two letters, except "A" and "I", which may not end the run;
* among complete covers, the one with the fewest pieces wins (ties: longer first piece).

The vocabulary is the most common English words plus clinic vocabulary, and the word
model's 250 sign labels are added at runtime (``add_words``). No third-party dependency.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

_COMMON = """
a about above after again against age ago all allergic allergies allergy almost alone along already also always am an and
another any anyone anything appointment are area arm around as ask asked at away baby back bad bandage bathroom be because
bed been before being believe best better between big bill bit bleeding blood body both bottle bought boy brain break
breath breathe breathing bring brother but by call called came can cancer cannot car card care chair change check checked
chest child children clinic close cold come comes coming could cough cramp cut dad daughter day days dear deaf did die
diet different dizzy do doctor does doing done dont door down dr drink drive drug during each ear early eat eating else
emergency end english enough even evening ever every everything exam excuse eye eyes face fall family far fast father
feel feeling feet fever few fill find fine finger first floor flu follow food foot for form forward found free friday
friend from front full get gets getting girl give glad glass go goes going gone good got great had hair half hand hands
happen happened happy hard has have having he head headache health hear heard hearing heart heavy hello help her here
herself hi high him himself his history hold home hope hospital hot hour hours house how hungry hurt hurts husband i id
if ill in info information injury inside insurance interpreter into is it its itch itchy just keep kidney kind knee knew
know last late later leave left leg less let letter life light like line little live long look looking lose lost lot
loud love low lung made make man many may maybe me mean medical medicine medicines meet met might mind minute minutes
miss mom monday month more morning most mother mouth move much must my myself name near neck need needs never new next
nice night no none nose not note nothing now number nurse of off office often oh ok okay old on once one only open or
other our out over own pain paper parent pass patient pay people per person pharmacy phone pick pill pills place please
pm point pregnant prescription pressure pretty problem problems put question questions quick quiet rash read ready
really reason red refill remember rest result results right room run sad said same saturday saw say scared school
second see seen send sent she shot should show sick side sign signs since sister sit sleep slow slowly small so some
someone something sometimes son soon sore sorry speak stay still stomach stop strong such sugar sunday sure surgery
swelling symptoms take taking talk tell test tests than thank thanks that the their them then there these they thing
things think this those though three throat thursday time tired to today together told tomorrow tonight too took tooth
top touch tuesday turn two under understand until up upset urgent us use used very visit vomit wait waiting walk want
wanted was water way we wednesday week weeks weight well went were what when where which while who why wife will with
without woman word words work worse would write wrong x xray year years yes yesterday yet you young your yourself
""".split()

_vocab: set[str] = {w.upper() for w in _COMMON}
MAX_WORD = 20


def add_words(words: Iterable[str]) -> None:
    """Add single-token words (e.g. the word model's labels) to the vocabulary."""
    for w in words:
        w = "".join(ch for ch in str(w) if ch.isalpha()).upper()
        if w:
            _vocab.add(w)
    _split.cache_clear()


def is_word(w: str) -> bool:
    return w.upper() in _vocab


def _ok_piece(p: str) -> bool:
    return p in _vocab and (len(p) >= 2 or p in ("A", "I"))


@lru_cache(maxsize=512)
def _split(run: str) -> tuple[str, ...] | None:
    n = len(run)
    # best[i] = fewest-pieces cover of run[i:], or None
    best: list[tuple[str, ...] | None] = [None] * (n + 1)
    best[n] = ()
    for i in range(n - 1, -1, -1):
        for j in range(min(n, i + MAX_WORD), i, -1):  # longer pieces first
            tail = best[j]
            piece = run[i:j]
            # a lone A / I may not end the run: "MAYA" must not become "MAY A"
            if tail is not None and _ok_piece(piece) and (len(piece) > 1 or j < n):
                cand = (piece, *tail)
                if best[i] is None or len(cand) < len(best[i]):
                    best[i] = cand
    return best[0]


# Words after which an unknown remainder is taken to be a name: "MYNAMEISJON" -> "MY NAME IS JON".
NAME_INTRO = frozenset({"IS", "AM", "CALLED", "DR", "DOCTOR", "NURSE", "MR", "MRS", "MS"})


def split_run(run: str) -> list[str]:
    """Pieces to space a closed letter run into; ``[run]`` when it should stay whole."""
    run = run.upper()
    if len(run) < 3 or not run.isalpha() or run in _vocab:
        return [run]
    pieces = _split(run)
    if pieces and len(pieces) > 1:
        return list(pieces)
    # known words + a name at the end, but only right after a name-introducing word
    for cut in range(len(run) - 2, 1, -1):      # remainder (the name) is >= 2 letters
        head = _split(run[:cut])
        if head and head[-1] in NAME_INTRO:
            return [*head, run[cut:]]
    return [run]
