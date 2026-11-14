> Packaging note: this application bundles the required GestureSign runtime subset; no external clone is needed. See README.md for current installation instructions. Historical phase instructions below describe development.

# GestureSign word recognition — Phase 1

This standalone adapter classifies **already segmented** ASL landmark clips.
It does not open a webcam, modify the OpenCV UI, or participate in letter/J/Z
recognition. No training dependencies or new packages are required in the
existing Python 3.11 inference environment.

```bash
.venv/bin/python -B word_recognizer.py
.venv/bin/python -B -m unittest -v test_word_recognizer test_motion
```

The CLI runs the six bundled replay clips and prints their expected source
sign, calibrated top-three probabilities, and ACCEPT/REJECT status. The
`quiet` clip is expected to produce `shh`; this is an upstream model confusion,
not an assertion that its source label is wrong. These probabilities describe
model confidence, not measured accuracy on a new signer.

## Local assets and source

The default source is the existing clone at `vendor/GestureSign/`, inspected
at commit `ffc0578`. It remains unchanged. Required files:

- `pipeline/src/gesturesign/{__init__,features,landmarks,resample,types}.py`
- `web/public/models/{model.onnx,model-meta.json,labels.json}`
- `web/public/replay/clips.json` for the replay CLI
- `shared/fixtures/features.json` for preprocessing parity tests

Override paths with `--repository PATH`, `--model-dir PATH`, and `--replay PATH`.
The model directory must contain all three matching model assets. The adapter
checks feature version, dimensions, label indices, calibration, and ONNX I/O.
Only one upstream source checkout may be imported per Python process.

The clone is currently ignored by the parent project's `.gitignore`; copying
only this adapter to another machine is insufficient. Retain the clone and
its license/credits, or explicitly package these source/assets in a later
deployment phase. Do not install `vendor/GestureSign/pipeline` wholesale:
its training dependencies require a conflicting NumPy version and Python >=3.12.
The reused lightweight source modules work in the existing Python 3.11/NumPy
1.26.4 environment and are checked against all 11 upstream parity cases.

## Python interface

```python
from word_recognizer import WordRecognizer

recognizer = WordRecognizer()
result = recognizer.recognize(frames, src_fps=30.0)
if result.accepted:
    print(result.top[0].label, result.top[0].confidence)
```

`frames` is a float-compatible array `[T, 543, 3]` in normalized MediaPipe
coordinates: face 0:468, left hand 468:489, pose 489:522, right hand 522:543.
Use NaN for absent landmarks. Do not pass pixel coordinates or hand-centered
CNN landmarks. The model uses x/y; z is retained for the common frame layout.

Preparation preserves upstream order: resample to metadata FPS, squeeze the
whole clip to at most 64 frames, then compute sequence features. It does not
pad short clips to 64 frames. After resampling, fewer than eight frames are
rejected. Completely handless clips are also rejected before inference.

The default shipped model recognizes 250 isolated signs. Decisions use
shipped-label-only softmax with the metadata temperature (about 0.88265) and
reject below the metadata threshold (about 0.29028). Rejected low-confidence
results retain the top three candidates. Nonfinite logits produce no guesses.
`partial=True` rejects an inadequately framed clip while retaining candidates;
the caller must supply that framing assessment. Live segmentation, automatic
framing checks, mode switching, transcript integration, and TTS are deferred.
There is no general out-of-vocabulary/random-motion detector: a forced segment
can still receive a confident incorrect label.

## Attribution and license

Upstream: [GestureSign](https://github.com/shashank11yadav/GestureSign),
Copyright (c) 2026 Shashank Yadav. The adapter imports the original NumPy
preprocessing without editing it. Calibrated decision behavior follows
`web/src/ml/decision.ts`; replay decoding follows `web/src/replay/clips.ts`.
The upstream MIT notice remains at `vendor/GestureSign/LICENSE` and is
reproduced below for the adapted behavior.

Model and replay provenance: Google – Isolated Sign Language Recognition
(Kaggle `asl-signs`), from the PopSign project, CC BY 4.0, as credited by the
upstream README and replay file. Keep those credits when redistributing assets.
The replay CLI prints the bundled dataset credit. The parent project's
alphabet-model licensing remains separate.

MIT License

Copyright (c) 2026 Shashank Yadav

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
