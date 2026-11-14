# SignBridge – ASL recognition backend

FastAPI service that takes a short video of **one isolated ASL sign** and returns the
recognised English word with calibrated confidence and the top-3 guesses.

- **Model:** [GestureSign](https://github.com/shashank11yadav/GestureSign) (MIT), a pretrained
  1.4 M-parameter conv + transformer over MediaPipe landmarks, exported to ONNX (5.6 MB).
  It was trained on the Kaggle *asl-signs* dataset (PopSign, CC BY 4.0) and covers
  **250 signs**. `GET /labels` lists them.
- **Upstream accuracy** (3 held-out signers, 14,396 clips): 72.1 % top-1, 85.3 % top-3.
- **Pipeline:** OpenCV decode → MediaPipe Holistic landmarks → upstream feature code
  (vendored in `app/gesturesign/`, checked against upstream's parity fixture) → ONNX Runtime (CPU).

## Setup (≈5 min)

```bash
cd signbridge-backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py      # fetches + SHA-256-verifies models/ (already committed)
```

The checkpoint files (`models/model.onnx`, `model-meta.json`, `labels.json`) are pinned
to upstream commit `ffc0578e`. `download_model.py` re-fetches them if they're missing or
modified.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test the pipeline without the server:

```bash
python scripts/test_inference.py --video examples/hello.mp4      # add --json for JSON
```

```bash
curl -X POST http://localhost:8000/predict -F "file=@examples/hello.mp4"
pytest                                                           # 13 tests, real model
```

Interactive docs: http://localhost:8000/docs

## API

### `GET /health`
```json
{"status":"ok","model_loaded":true,"model_name":"GestureSign (asl-signs, 250 signs)",
 "model_version":"default-2fc9038cc4","device":"cpu","error":null}
```
If the model fails to load, the server still starts and returns `"status":"degraded"`,
`"model_loaded":false` and the error. `/predict` then returns 503.

### `GET /labels`
`{"count":250,"labels":[{"index":0,"label":"TV","sign":"TV","held_out_top1":0.83}, ...]}`.
`label` is the display name (`"thank you"`, `"call (on phone)"`). `sign` is the raw class
name (`"thankyou"`, `"callonphone"`). `held_out_top1` is upstream's per-sign accuracy.

### `POST /predict`
Multipart field `file`: `.mp4 .mov .webm .m4v .avi .mkv`, ≤ 50 MB, ≤ 15 s.

Real response for `examples/hello.mp4`:
```json
{
  "status": "success",
  "prediction": "hello",
  "confidence": 0.9665,
  "confident": true,
  "not_sure_threshold": 0.2903,
  "top_predictions": [
    {"label": "hello", "sign": "hello", "confidence": 0.9665},
    {"label": "all",   "sign": "all",   "confidence": 0.0018},
    {"label": "open",  "sign": "open",  "confidence": 0.001}
  ],
  "inference_time_ms": 4023.0,
  "timing_ms": {"preprocessing": 4019.8, "model": 1.7, "total": 4023.0},
  "model_name": "GestureSign (asl-signs, 250 signs)",
  "model_version": "default-2fc9038cc4",
  "video": {"frames_processed": 72, "frames_with_hands": 28, "sign_segment_seconds": 0.936,
            "hand_coverage_in_segment": 1.0, "model_frames": 28, "source_fps": 29.92,
            "fps_source": "timestamps", "video_seconds": 2.406, "width": 1080, "height": 1920},
  "warnings": []
}
```

- `confidence` is `softmax(logits / T)` with the temperature that upstream fitted on held-out
  validation signers (T = 0.883), so it is calibrated rather than a raw softmax.
- `confident` is `false` when the top confidence is below the calibrated "not sure" threshold
  (0.29) or when the signer's shoulders are out of frame in most frames. **The frontend should
  show "Not sure" when `confident` is false.** Upstream measured that above the threshold,
  answers are 77 % correct on held-out signers.
- `warnings` flags clips that are only partly framed or are longer than 4 s of signing.

Errors always look like `{"status":"error","error":{"code":"...","message":"..."}}`:

| HTTP | code | when |
|---|---|---|
| 400 | `empty_file` | zero-byte upload |
| 413 | `file_too_large`, `video_too_long` | > 50 MB, > 15 s |
| 415 | `unsupported_file_type` | wrong extension |
| 422 | `unreadable_video`, `empty_or_corrupt_video` | can't decode |
| 422 | `no_hands_detected`, `sign_too_short` | no hands, or < 8 frames (0.27 s) with hands |
| 422 | `invalid_request` | missing `file` field |
| 503 | `model_unavailable` | model failed to load |

## How the preprocessing matches training

1. Every decoded frame goes through **MediaPipe Holistic (legacy graph, `mediapipe==0.10.21`)**,
   the same graph Google used to produce the asl-signs landmarks. Results are stored in the
   Kaggle 543-point layout (face 468, left hand 21, pose 33, right hand 21), with NaN for misses.
2. The clip is trimmed to the first-to-last frame that contains a hand. The browser demo
   segments signs by hand presence in the same way.
3. The fps comes from frame timestamps, falling back to the container value. Sources above
   45 fps are frame-skipped toward 30 fps.
4. `resample → 30 fps`, `fit_length → ≤ 64 frames`, then `sequence_features` (460 dims):
   upstream functions, called in upstream's order. Handedness and mirroring are normalised
   by upstream's features, so left-handed signers and mirrored selfie video are handled.
5. ONNX forward, then calibrated softmax.

`tests/test_pipeline.py` checks the vendored features against upstream's parity fixture
(11 cases, tolerance 1e-4). It also runs 6 real asl-signs landmark clips from held-out
signers through the model.

## Verification results (Apple M2, CPU)

**Held-out landmark clips** (upstream replay set, signers never seen in training):
5/6 top-1, 5/6 top-3. *quiet* → "shh" is a known near-synonym confusion.

**Real videos, full video pipeline**: 6 labeled PopSign v1.0 clips (CC BY 4.0,
`game/test` split):

| true sign | prediction (confidence) | top-3 hit |
|---|---|---|
| hello | hello (0.63) | ✓ |
| hello | pretend (0.79) | ✗ |
| hello | hello (0.97) | ✓ |
| thank you | thank you (0.42) | ✓ |
| thank you | napkin (0.35) | ✓ (2nd) |
| thank you | thank you (0.86) | ✓ |

4/6 top-1, 5/6 top-3. This is a smoke test, not an accuracy benchmark: six clips, two signs,
and these PopSign signers may overlap the asl-signs training signers. Rely on upstream's
72 % held-out figure, and test with your own team's signing before the demo.

**Latency:** the ONNX model takes about 2–4 ms. Almost all the time is MediaPipe Holistic,
about 60 ms per frame on 1080p input. Measured end to end on the server: 3.4–5.0 s for a
2–2.4 s clip. The CLI also pays a one-off ~2 s MediaPipe import.
To go faster, set `SIGNBRIDGE_HOLISTIC_COMPLEXITY=0` (~30 % faster per frame; not
accuracy-tested), or record shorter clips.

## Configuration (env vars)

| var | default | |
|---|---|---|
| `SIGNBRIDGE_MAX_UPLOAD_MB` | 50 | |
| `SIGNBRIDGE_MAX_VIDEO_SECONDS` | 15 | |
| `SIGNBRIDGE_HOLISTIC_COMPLEXITY` | 1 | 0 = faster, lite pose model (downloaded by MediaPipe on first use) |
| `SIGNBRIDGE_MAX_FRAME_SIDE` | 960 | frames downscaled before MediaPipe (landmarks are resolution-independent) |
| `SIGNBRIDGE_MAX_CONCURRENT` | 2 | simultaneous MediaPipe jobs |
| `SIGNBRIDGE_LOG_LEVEL` | INFO | |

## Known limitations

- **Isolated signs only, 250-word vocabulary** (children's-vocabulary list from PopSign: *hello,
  thank you, yes, no, please, water, …*). There is no fingerspelling, grammar or sentences, and
  no *help*. A clip with several signs gets squeezed into one prediction, with a warning.
- **The signer must be framed from the chest up**, with hands visible and decent lighting.
  If MediaPipe can't see the hands, the model can't either.
- **Accuracy:** 72 % top-1 for unseen signers (upstream). Look-alike signs get confused
  (awake/wake, pen/pencil, quiet/shh, sleep/nap). Always show the top-3.
- **Latency is seconds, not milliseconds**, because MediaPipe runs per frame on the CPU. This
  is fine for "record, then upload", not for live streaming.
- Browser `MediaRecorder` WebM files often have no fps metadata. The pipeline estimates fps
  from frame timestamps instead. WebM was tested with an OpenCV-written file, not a real
  browser recording.
- `pip check` reports "mediapipe 0.10.21 is not supported on this platform" on macOS 14 arm64.
  This is a wheel-tag quirk: the package imports and runs correctly, as the tests show.
- Not a medical-grade or interpreter-grade system. See upstream's model card.

## Layout

```
app/main.py            FastAPI app, endpoints, error handling, upload handling
app/model.py           ONNX Runtime wrapper, calibration, top-k
app/preprocessing.py   video decode, MediaPipe Holistic, trimming, features
app/gesturesign/       upstream feature code (vendored verbatim, MIT)
app/schemas.py         Pydantic response models
app/config.py          settings
models/                pretrained ONNX + metadata + labels
scripts/               test_inference.py (CLI), download_model.py
tests/                 test_api.py, test_pipeline.py, data/ (upstream fixtures)
examples/              hello.mp4, thank_you.mp4 (PopSign, CC BY 4.0)
```

## Credits

Model and feature code: GestureSign by Shashank Yadav (MIT). Data: Google – Isolated Sign
Language Recognition (Kaggle asl-signs) / PopSign ASL v1.0, Georgia Tech (CC BY 4.0).
Landmarks: Google MediaPipe (Apache 2.0).
