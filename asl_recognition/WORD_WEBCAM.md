> Packaging note: this application bundles the required GestureSign runtime subset; no external clone is needed. See README.md for current installation instructions. Historical phase instructions below describe development.

# Standalone ASL words — Phase 2

Launch from this project's root using the existing inference environment:

```bash
.venv/bin/python -B test_word_webcam.py --camera 0 --debug
```

Allow camera access for the terminal/application if macOS asks. If camera 0 is
not your intended webcam, try `--camera 1`. Requested capture size defaults to
960x720; the driver may choose a different size. Use `--width` and `--height`
to request alternatives. `--repository PATH` selects the GestureSign clone.

This is a separate OpenCV window. The alphabet recognizer, its UI, J/Z rules,
and Phase 1 adapter are unchanged. No webcam is opened when importing the
modules or running automated tests. There is no transcript or TTS integration.

## Manual verification

1. Keep one person, the face, shoulders, and signing hands in the image. Confirm
   `ok`, hand count, face presence, and a reasonable measured FPS in the panel.
   The two hand colors/labels identify anatomical L/R, not screen-left/right.
2. Sign a supported isolated word such as hello, police, blue, or giraffe. The
   state should change from `IDLE` to `SIGNING` and the frame count should grow.
3. Hold still or lower the hands for at least 300 ms. A completed segment is
   passed to the unchanged `WordRecognizer`; its top three guesses and
   ACCEPT/REJECT status appear under **Last completed sign**. Predictions remain
   visible while waiting for the next sign; they are not new repeated results.
4. A still hand should not repeatedly start recordings. A quick lowering motion
   after a held sign should produce `discard: hands-leaving`, not another word.
5. Press `r` mid-sign: the recording, previous result, tracking history, and
   segmentation state should clear. Press `q` or close the window to release
   the camera. An unfinished recording is discarded on exit, not classified.
6. Move a shoulder outside the frame and sign again. Majority-partial framing
   should produce REJECT with reason `partial`, while showing its candidate
   guesses. Missing-hand frames do not count against this framing vote.

The default preview is mirrored. Tracking and inference ALWAYS receive the
original unmirrored camera frame. `--no-mirror` changes only the preview;
landmarks are mapped to the preview without changing their model coordinates.

Console output includes quality changes, segment start/end/discard reasons,
captured frame count, estimated source FPS/duration, calibrated probabilities,
and rejection reasons. `--debug` additionally prints live state once per second.
FPS is measured from monotonic capture timestamps, not the camera's nominal FPS.

## Tracking and segmentation

`word_tracking.py` uses the installed MediaPipe 0.10 Python Holistic API with
the standard packaged model (`model_complexity=1`), no image segmentation mask,
and a 468-point face mesh. It packs normalized xyz into float32 `[543,3]`:
face 0:468, anatomical left hand 468:489, pose 489:522, right hand 522:543.
Missing parts remain NaN. No pixel-space normalization or letter smoothing is
applied to these landmarks. Holistic's own landmark smoothing is enabled.

`word_segmenter.py` ports GestureSign's `web/src/ml/segmenter.ts` defaults:

| Rule | Default |
|---|---:|
| Hand-point speed, normalized by shoulder width | 0.6 shoulder widths/sec |
| Speed smoothing | 0.5 previous + 0.5 current |
| Preroll | 200 ms |
| Still/absent gap ending a sign | 300 ms |
| Minimum captured frames | 8 |
| Maximum recording duration | 4000 ms |
| Short exit after a held sign rejected | under 500 ms |

The `idle/signing/resting` transitions, gap trimming, dropout bridging, timeout,
and flush behavior follow upstream. Each completed clip estimates its source
FPS as `(frames - 1) / duration`. Phase 1 then resamples and checks the model's
minimum frame count again. Python additionally copies buffered arrays and
rejects nonfinite or non-increasing timestamps. Reset before changing sources.

The demo's quality vote follows `web/src/ml/recognizer.ts`: if partial frames
outnumber ok frames during recording, pass `partial=True` to the adapter. Its
temperature and confidence threshold are unchanged; no thresholds were lowered.

## Automated checks

```bash
.venv/bin/python -B -m unittest -v test_word_live test_word_recognizer test_motion
```

The new camera-free tests cover the upstream segmenter scenarios, resets,
preroll, timestamp validation, short/incomplete recordings, landmark packing,
preview mirroring, RGB conversion/lifecycle with a fake tracker, framing
rejection, and all six real replay clips through segmentation and ONNX.
The preexisting 30 tests remain unchanged. No real webcam or MediaPipe graph
is initialized by these tests; live capture accuracy and FPS need manual checks.

## Limitations and attribution

- Python legacy Holistic and the browser Tasks landmarker may differ in quality
  and performance. This port preserves the feature contract, not guaranteed
  webcam prediction parity. CPU tracking, rather than ONNX inference, may limit FPS.
- Holistic tracks one prominent person. It cannot reliably count additional
  people; although the segmenter supports a crowded flag, the demo cannot
  automatically provide reliable crowd detection. Use a single-person scene.
- Missing lips are reported but not an automatic rejection, matching upstream.
  Keep the face visible for the model's expected features and aspect correction.
- Very still signs may not trigger movement onset. Eight frames is a recording
  quality threshold, not proof that a semantic sign is complete. Arbitrary or
  incomplete motion can still receive a confident word prediction.
- Recognition is for isolated signs from the 250-label vocabulary, not continuous
  ASL sentences. The known `quiet` to displayed `shh` confusion remains.

Segmentation, assembly, landmark layout, and quality-vote behavior derive from
[GestureSign](https://github.com/shashank11yadav/GestureSign), inspected at commit
`ffc0578`, Copyright (c) 2026 Shashank Yadav, MIT License. Full MIT terms and
model/replay dataset credits are preserved in [WORD_RECOGNITION.md](WORD_RECOGNITION.md)
and the untouched `vendor/GestureSign/LICENSE` and README. Replay landmarks
are from Google/Kaggle `asl-signs` (PopSign), CC BY 4.0. No training dependencies
were installed and neither pretrained model was changed.
