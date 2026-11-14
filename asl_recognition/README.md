# ASL letter and word recognition

Self-contained Python 3.11 desktop application with an OpenCV webcam UI.
Both pretrained ONNX models, labels, reference image, GestureSign runtime
preprocessing, and regression fixtures are included. No external clone,
training packages, or dataset download is required. Install dependencies first;
MediaPipe supplies its own tracking models in its Python distribution.

```bash
cd ~/Signify/asl_recognition
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B realtime_translator.py --camera 0 --word-commit-confidence 0.75
```

Requires a desktop display and camera access. Grant camera permission to the
terminal/Python application when prompted by the OS. Use `--camera 1` if needed.
Start in word mode with `--mode word`; model defaults resolve relative to the
script, so launching by absolute path also works. Explicit overrides remain
`--model`, `--classes`, `--word-repository` (bundled subset layout), and
`--word-model-dir` (word ONNX plus both JSON files).

Press M or click LETTER/WORD to switch; Q quits. Space, Backspace, C, R,
and U edit the shared transcript. Words append automatically only after a
completed segment passes model acceptance and the 0.75 commit threshold.
This classifies isolated signs; it does not translate continuous ASL sentences.
See COMBINED_APP.md for controls, limitations, and manual camera checks.

```bash
.venv/bin/python -B -m unittest discover -v
.venv/bin/python -B realtime_translator.py --help
.venv/bin/python -B word_recognizer.py
```

The tests use unittest (no extra test dependencies). They include real ONNX
word inference over six attributed replay clips, eleven preprocessing parity
fixtures, letter motion, transcript behavior, and the application main loop
with a simulated camera and GUI. `test_word_webcam.py` is also a manual webcam
harness and supplies test helpers; it has no unittest test cases itself.

## Packaging and licenses

Only runtime modules/assets and tests were extracted from GestureSign into
`vendor/GestureSign/`; its original internal paths are retained for compatible
asset lookup. Its preprocessing modules and model bytes are unchanged.
The adapter default now targets that subset. Letter asset defaults resolve
relative to this application. Both projects' MIT notices are retained.
Historical phase documents are preserved with packaging notes; the original
letter README is archived in docs/ORIGINAL_README.md (its training instructions
and relative links refer to the original project, not this runtime package).

Read THIRD_PARTY_NOTICES.md before redistribution: the alphabet model's weight
license is unresolved. No publication authorization is implied by this copy.
ADDED_FILES.txt lists every packaged file. MODEL_MANIFEST.json records model
hashes, source revisions, and letter model input/output metadata.

## Verification of this copy

74 automated tests passed from this directory using Python 3.11 and the source
project's existing interpreter, with bytecode writes disabled. No environment
was copied. Requirements pin the installed versions used for those checks;
a fresh dependency installation has not been tested. CLI help, real loading of
both models, and six-clip replay inference passed. Main-loop launch behavior
was tested with simulated camera/GUI; actual webcam accuracy, OS permissions,
and display interaction require the manual checks in COMBINED_APP.md.
