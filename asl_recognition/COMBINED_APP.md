> Packaging note: this application bundles the required GestureSign runtime subset; no external clone is needed. See README.md for current installation instructions. Historical phase instructions below describe development.

# Combined Letter / Word application — Phase 3

Launch the actual application in the existing inference environment:

```bash
.venv/bin/python -B realtime_translator.py --camera 0 --word-commit-confidence 0.75
```

It starts in LETTER mode. Click **LETTER**/**WORD** at the top-left of the
original preview, or press **M**. `--mode word` starts directly in WORD mode.
The original `--activations` and `--debug-motion` options remain available.
`--word-repository PATH` and `--word-model-dir PATH` override GestureSign paths.
No packages or training dependencies need to be installed for this phase.

## Behavior

- One `cv2.VideoCapture` belongs to `realtime_translator.py`. Only the selected
  tracker/model processes each camera frame. The old tracker is closed on a
  mode switch; switching back creates a fresh graph. Word model loading is lazy.
- LETTER preserves the existing CNN input renderer, mirrored camera input,
  adaptive landmark smoothing, eight-frame probability average, J/Z motion
  logic, confidence thresholds, ten-frame automatic commit, and repeat gate.
- WORD uses the original unmirrored frame through Holistic, normalized landmark
  packing, the Phase 2 segmenter, and the unchanged Phase 1 ONNX adapter.
  Landmark overlays are mirrored to match the original preview. Its candidates
  are never displayed in the combined UI. A single accepted top word is appended
  automatically after gesture completion, without confirmation.
- Automatic word recording requires BOTH upstream calibrated acceptance and
  `--word-commit-confidence` (default **0.75**). Changing the latter does not
  change the model's temperature or calibrated rejection threshold (~0.29028).
  Below-threshold and rejected segments never append transcript text.
- Completion timestamps and a mode generation identify each result. The same
  completion cannot write twice, including after undo. A fresh gesture may
  legitimately repeat the same word. The segmenter's resting/trailing-exit
  behavior remains active; it is not blindly reset after every word. After a
  four-second timeout, inference is disarmed until 300 ms of stillness or hand
  absence, preventing continued movement from generating repeated commits.
- Mode switches reset pending letter agreement, motion/smoothing history,
  probability vectors, word recording, and previous word candidates. Committed
  text stays intact, and results from old mode generations cannot append.

The original dimensions, camera area, rail, colors, bottom transcript area, and
Reference / Filters controls remain. Mode buttons and a last-recorded overlay
are added over the preview. In WORD mode the old top-three area reports automatic
recording information instead. Filters are enabled in LETTER mode; Reference
continues to work in either mode. Word tracking quality/frame counts appear in
the existing debug rail, and commit/rejection status appears in the bottom bar.

## Shared transcript and controls

Automatic letters form an uppercase fingerspelled run. Words use the model's
display label, including multiword labels such as `thank you`, with one separating
space. Letters after a recorded word begin a new run. Switching modes alone
inserts no space, so `CA`, switching to WORD and back without committing a word,
then signing T still gives `CAT`.

| Control | Shared transcript action |
|---|---|
| Space | Add one space; do not accumulate duplicate spaces |
| Backspace | Delete one character |
| C | Clear text |
| R | Repeat the last character, preserving the existing repeat behavior |
| U | Undo the last transcript operation, including an automatic word append |
| M | Toggle Letter/Word mode; keep transcript |
| Q | Quit and release the camera |

The last-recorded overlay follows transcript history and is restored by undo.
The transcript is in memory; this phase does not add file persistence or TTS.

## Tests

```bash
.venv/bin/python -B -m unittest discover -v
```

The combined tests exercise `realtime_translator.main()` with a fake camera,
tracking graphs, model outputs, and GUI callbacks. They verify a single capture
instance, keyboard AND button switches, selected-pipeline execution, correct
input mirroring, original letter commitment, and automatic word recording in
the same UI state. Other cases cover transcript edits/spacing/undo, mode resets,
stale completions, duplicate prevention, timeout rearming, original geometry,
and hiding word candidate lists. Real ONNX replay tests verify the word model's
separate recording threshold; all existing Phase 1/2 and J/Z tests remain.

## Manual webcam verification

1. Confirm LETTER is highlighted and the original preview, reference, filters,
   prediction smoothing, and ten-frame progress still work. Test static letters
   and both J/Z with `--debug-motion` if needed.
2. Spell CA, switch modes with M and return without signing a word, then sign T.
   Check that the transcript is CAT. Repeat switching using both visible buttons.
3. In WORD, move back so face, shoulders, and both hands are visible. Sign a
   supported isolated word such as hello or police, then pause or lower the hands.
   Check that a confident result appends once and the last-recorded overlay updates.
   No top-three candidates or confirmation window should appear.
4. Hold the endpoint and lower the hands: no duplicate append should occur.
   Sign the same word again as a separate gesture; that repeat should be allowed.
5. Observe a rejection (e.g. incomplete/poorly framed input or lower-confidence
   prediction): status should explain that it was not recorded and text stays
   unchanged. The classifier can still confidently misrecognize arbitrary motion;
   semantic correctness is not guaranteed by its confidence threshold.
6. Switch modes mid-gesture: its unfinished recording must not commit later.
   Exercise space, backspace, R, C, and U across mixed letter/word text.
7. Check camera orientation and hand overlays in WORD. For a long gesture that
   reaches the timeout, pause/lower hands before starting the next sign.

Live accuracy and tracking FPS must be checked on the actual webcam. Python
legacy Holistic tracks one prominent person, cannot reliably detect crowds,
and may differ from the browser's tracking quality. This remains isolated-sign
classification over the existing 250-word vocabulary, not sentence translation.
Known model confusions such as `quiet` -> `shh` remain. The required GestureSign runtime subset is bundled under vendor/GestureSign/.

Upstream code/model/replay attribution and the MIT notice remain in
[WORD_RECOGNITION.md](WORD_RECOGNITION.md). The extracted `word_session.py`
retains GestureSign's quality-vote attribution. Neither pretrained model,
`motion.py`, `smoothing.py`, nor either model's preprocessing was changed.
