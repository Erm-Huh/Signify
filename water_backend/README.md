# SignBridge – ASL recognition backend

ASL recognition with two pretrained models: **250 isolated ASL words** (GestureSign) and
**fingerspelled letters A–Z** (MediaPipe-ASL alphabet CNN). It includes a FastAPI service
for a frontend and one unified desktop webcam app.

**Launch the unified app:**

```bash
cd ~/signbridge-backend && source .venv/bin/activate && python scripts/signbridge_app.py
```

- **Model:** [GestureSign](https://github.com/shashank11yadav/GestureSign) (MIT), a pretrained
  1.4 M-parameter conv + transformer over MediaPipe landmarks, exported to ONNX (5.6 MB).
  It was trained on the Kaggle *asl-signs* dataset (PopSign, CC BY 4.0) and covers
  **250 signs**. `GET /labels` lists them.
- **Upstream accuracy** (3 held-out signers, 14,396 clips): 72.1 % top-1, 85.3 % top-3.
- **Pipeline:** OpenCV decode → MediaPipe Holistic landmarks → upstream feature code
  (vendored in `app/gesturesign/`, checked against upstream's parity fixture) → ONNX Runtime (CPU).
- **Alphabet model:** [MediaPipe-ASL-sign-language-recognition](https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition)
  (code MIT; **weights derive from GPL-2.0/CC0 datasets**, see
  `models/alphabet/ATTRIBUTION.md`). MediaPipe Hands landmarks are rendered as a skeleton
  image and classified by a CNN (ONNX, 26 classes). J and Z are recognised by upstream's
  rule-based motion detector. Upstream reports 93.7 % on a held-out photo dataset and 96.7 %
  precision at its 0.75 commit threshold.

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
pytest                                                           # 75 tests, real models
```

Interactive docs: http://localhost:8000/docs

## Unified live app: words + letters, one result

```bash
python scripts/signbridge_app.py                  # AUTO mode, camera 0
python scripts/signbridge_app.py --mode SPELL     # start in fingerspelling mode
python scripts/signbridge_app.py --mode SPELL --pause-word-in-spell   # letters at full rate
python scripts/signbridge_app.py --source examples/hello.mp4 --duration 15   # no camera
```

There is one window and one camera owner. Every frame goes to both recognisers, the word
model (GestureSign sliding windows, as in live mode) and the alphabet model (per frame),
through bounded drop-oldest queues on their own worker threads. A conservative **router**
then picks a single primary result.

| Shown | Meaning |
|---|---|
| **WORD** / **LETTER**, *Confident (stable)* | that model passed **its own** threshold and stability rule |
| **WORD** / **LETTER**, *Tentative* | above its threshold but not yet stable; never auto-committed |
| **AMBIGUOUS** | both models are confident; there is no validated way to choose, so both are shown |
| **NOT SURE** | hands visible, but nothing passed its model's threshold |
| **NO HANDS** | neither model saw a hand recently |

**Modes** (TAB cycles): **AUTO** (default) uses the router above. **WORD** takes the primary
result from GestureSign only. **SPELL** takes it from the alphabet model only and builds words
from letters. The other model keeps running for the details panel; use
`--pause-word-in-spell` / `--pause-alphabet-in-word` to save CPU.

**Keys** (also shown in the window):

| Key | Action |
|---|---|
| Q / Esc | quit (releases camera, workers, MediaPipe, TTS) |
| TAB | AUTO → WORD → SPELL |
| SPACE | AUTO/WORD: capture the next 2 s for the word model (as before) · SPELL: insert a space |
| ENTER | confirm the primary WORD/LETTER into the text |
| W / L | confirm the current top word / top letter manually (the fallback for AMBIGUOUS) |
| R | repeat the last letter (intentional double letters, upstream's key) |
| D | toggle **Other Possibilities**: top 3 words and top 3 letters with their age, greyed + STALE when old |
| BACKSPACE | delete the last committed item |
| C | clear the text |
| S | speak the committed text (macOS `say`, in a child process, never blocks the preview) |

Changes from the earlier demos: SPACE still captures a word, except in SPELL mode where it
inserts a space, as in upstream's alphabet app. Confirming moved to ENTER. The earlier
scripts (`webcam_demo.py`, `live_demo.py`) are unchanged.

### Decision rules (`app/services/recognition_router.py`, all configurable)

- The two models' scores are **never compared** and never combined. Word "confident" means
  the calibrated smoothed probability is ≥ 0.29 (GestureSign's held-out threshold) and the
  signer is in frame. Letter "confident" means the 8-frame mean softmax is ≥ 0.75 (upstream's
  `COMMIT_CONFIDENCE`; uncalibrated).
- **Freshness:** a word result older than 1.5 s (`--word-max-age`) or a letter result older
  than 0.5 s (`--letter-max-age`) is stale. It is ignored for the decision and marked STALE
  in the details.
- **AUTO:** if both are confident the result is AMBIGUOUS, with no winner picked. If one is
  confident, that one is shown. Otherwise it's NOT SURE or NO HANDS.
- **Automatic text** only comes from stable results that the router is currently showing:
  - a word from the live model's emission rule (2 agreeing windows plus cooldown); the
    same word is never auto-appended twice in a row
  - a letter from upstream's commit rule (10 still frames ≥ 0.75; no immediate repeat)

  AMBIGUOUS, NOT SURE and tentative results are never appended automatically. ENTER, W, L
  and R are deliberate manual confirmations.
- A SPACE capture result stays the primary output for 5 s, shown as WORD only if it passed
  the calibrated threshold.
- **Not validated:** whether AUTO picks the right kind of sign. No labeled data mixes words
  and fingerspelling. A flat "hello" hand, for example, *is* the B handshape. That's why AUTO
  refuses to guess when both models are confident, and why WORD and SPELL exist.

### Measured performance (Apple M2, CPU, real webcam 1280×720)

| | AUTO (both models), 20 s | SPELL with word paused, 15 s |
|---|---|---|
| preview frame interval | median 33 ms, p95 46 ms | median 33 ms, p95 36 ms |
| alphabet frames processed | 443 / 571 (21.6 per s) | 432 / 445 (30.0 per s) |
| Hands + CNN per frame | median 41 ms, p95 84 ms | median 24 ms |
| alphabet capture → result | median 60 ms, p95 98 ms | median 24 ms |
| Holistic (word) per frame | median 74 ms (288 / 571 frames) | paused |
| word window end → decision | median 239 ms, p95 304 ms | – |
| word windows skipped (busy) | 0 of 39 | – |

How I got there: the first version ran both models at once, and ONNX Runtime's spin-waiting
threads starved MediaPipe. Alphabet throughput was 13.5 per s, with 112 ms per Holistic frame
and a 64 ms p95 preview. Giving the alphabet session 2 threads with spinning disabled, and
running Hands on a 640 px copy, fixed most of it. Landmarks are converted back to
full-frame pixel units, so upstream's pixel-based smoothing and motion thresholds are
unchanged. The SPELL+pause column matches the 30 fps that upstream's frame-count constants
assume. In AUTO, letters run at about 22 per s, so upstream's 10-frame commit takes about
0.45 s instead of 0.33 s. That trade-off is documented rather than hidden.

## Webcam demo (no frontend needed)

```bash
source .venv/bin/activate
python scripts/webcam_demo.py            # camera index 0; --source 1 for another camera
```

1. A preview window opens (mirrored like a selfie camera). Frame yourself **chest-up**
   with both hands visible.
2. Press **SPACE**, then sign **one** word during the 2-second recording (red bar).
3. The camera preview stays live while one background worker runs the backend pipeline on
   the clip. SPACE is ignored until that prediction finishes, so predictions never overlap.
4. The panel shows the predicted sign, its calibrated confidence and the top 3. When
   `confident` is false (same rule as `/predict`), it shows **"Not sure"** with the best guess.
   Results are also printed to the terminal.
5. **Q** or **Esc** quits. This releases the camera, waits for any running prediction and
   deletes the temp clip folder.

**This is capture-then-predict, not continuous real-time recognition.** Each result arrives
about 3–7 s after the recording ends (MediaPipe Holistic on CPU).

The demo reuses `preprocess_video` and `SignRecognizer.predict` from `app/` unchanged. Each
capture is written to a temp MP4 at the frame rate measured during capture, then processed
exactly like an upload.

**macOS camera permission:** the first run asks for camera access for the app that launched
Python (Terminal, iTerm or VS Code). If you see `not authorized to capture video` or
`could not open camera`, allow that app in *System Settings › Privacy & Security › Camera*,
restart it and try again.

Without a camera or a person (e.g. CI), use a video file as a looping fake camera:
```bash
python scripts/webcam_demo.py --source examples/hello.mp4 --auto-capture 2
```
`--auto-capture N` presses SPACE automatically N times and quits after the last result.

## Continuous live mode (no start/stop)

```bash
python scripts/live_demo.py                       # camera 0
python scripts/live_demo.py --interval 0.5 --window 2.0 --smoothing 3 --stable 2 --cooldown 1.5
python scripts/live_demo.py --source examples/hello.mp4 --duration 15   # fake camera, auto-quit
```

Sign one word, chest-up, then lower your hands. The large label is the current smoothed
guess, or **"Not sure"** when it's below the calibrated threshold or you're partly out of
frame. The **Accepted** strip lists signs that passed smoothing and cooldown. **Q**/Esc
quits, **C** clears the strip. The script prints measured latency stats when it exits.

**What it is:** sliding-window classification of *isolated* signs. Every `--interval`
seconds, the last `--window` seconds of the stream are classified as if they were one
uploaded clip. **What it is not:** sign segmentation or sentence recognition. A window that
covers half a sign, or the end of one and the start of the next, still gets classified.
Smoothing and the "not sure" threshold suppress most of those, but not all.

How it fits the model's input contract. These limits come from inspecting the ONNX model and
`model-meta.json`:

- The model takes `[1, T, 460]` features at **30 fps with 8 ≤ T ≤ 64**. The ONNX graph errors
  for T > 64, so `--window` is validated to 0.23–2.1 s. The default 2.0 s gives 61 frames.
- MediaPipe Holistic is a *tracking* graph and takes 33–60 ms per frame on the M2 CPU. Each
  camera frame therefore goes through Holistic **once**, in order, on a landmark worker thread.
  The resulting landmarks, 6.5 KB per frame, go into a RAM ring buffer of 4 s. No raw frames
  are buffered and no files are written.
- Holistic can't keep up with 30 fps, so the camera→Holistic queue holds 2 frames and drops the
  oldest. Each window is then resampled to an even 30 fps **on the real capture timestamps**,
  using upstream's `interpolate`. For evenly spaced input this is identical to upstream
  `resample`, and a test checks that.
- After resampling, the window goes through the **same** `landmarks_to_features` (hand-span
  trim, `fit_length`, upstream features) and `SignRecognizer.predict` (calibrated softmax) as
  `POST /predict`. A window without hands, or with fewer than 8 hand frames, is "idle".

Threads: camera/preview (main), landmark worker, and one inference worker. A tick that comes
due while the previous window is still running is **skipped, not queued**. A window with no
new landmarks since the last one is skipped too. Shutdown stops both workers, cancels pending
jobs, closes Holistic and clears the buffers.

**Smoothing and cooldown** (`app/live.py: SignSmoother`):
- The calibrated probabilities of the last `--smoothing` windows are averaged. The display is
  "Not sure" unless the averaged top-1 is ≥ 0.29 (upstream's held-out threshold) and you're
  in frame.
- A sign is *accepted* after `--stable` consecutive confident windows with the same label, at
  least `--cooldown` seconds after the previous accepted sign.
- The same sign isn't accepted twice in a row until something interrupts it: hands down, a
  "Not sure" window, or a different label.
- Lowering your hands clears the smoothing history.

**Measured latency** (Apple M2, CPU, defaults, `live_demo.py` exit stats):

| | real webcam (1280×720, 20 s) | fake camera `examples/hello.mp4` (1080×1920, 15 s) |
|---|---|---|
| preview frame interval | median 33 ms, p95 35 ms | median 33 ms, p95 34 ms |
| Holistic per frame | median 33 ms, p95 56 ms | median 57 ms, p95 77 ms |
| frames reaching Holistic | 436 / 589 (74 %) | 275 / 449 (61 %) |
| features + ONNX per window | median 4.7 ms, p95 10.6 ms | median 3.9 ms, p95 6.9 ms |
| last frame of window → decision | median 140 ms, p95 182 ms | median 139 ms, p95 146 ms |
| windows skipped because busy | 0 of 38 | 0 of 30 |

Not measured (no ground-truth timing): the delay from finishing a sign to it being
*accepted*. By construction it is at least one more interval (0.5 s) plus about 0.15 s,
because acceptance needs `--stable 2` consecutive windows.

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

### `GET /labels/alphabet`
`{"count": 26, "labels": ["A", ..., "Z"]}`

### `POST /predict/alphabet`
Multipart field `file`, either:
- **an image** (`.jpg .jpeg .png .bmp .webp`): one frame, extracted the way upstream extracted
  its training photos (static mode, 25 % black border). There is no smoothing, stability or
  motion (`motion_supported: false`), so J and Z come with a warning.
- **a video** (same formats and limits as `/predict`): upstream's live pipeline per frame
  (mirroring, OneEuro smoothing, 8-frame average, J/Z motion, commit rule). Returns the final
  state and the letters committed along the way.

Real response for frame 30 of `examples/hello.mp4` as a PNG (a flat hand, which *is* the B handshape):
```json
{"status":"success","input_type":"image","hand_detected":true,"prediction":"B",
 "confidence":0.996,"passes_threshold":true,"uncertain":false,"stable":null,"threshold":0.75,
 "top_predictions":[{"label":"B","confidence":0.996}, ...],"motion_supported":false, ...}
```
`confidence` is a plain softmax score (**not calibrated**, unlike `/predict`). `/health` also
reports `alphabet_model_loaded`. The alphabet model loads independently, so if it fails,
`/predict` keeps working.

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

### Alphabet model checks

- **Parity:** upstream files are byte-identical (hash-checked in tests). Our wrapper
  reproduces the logits upstream's own code produced, within 2e-3, on 23 landmark sets.
- **Labeled photos:** upstream's 26 reference photos, run through upstream's photo pipeline.
  MediaPipe found a hand in 23/26 (missed N, O, P on these small 128 px tiles). Top-1 was
  21/23 (H→J, W→U) and top-3 23/23. *These photos look like the single-subject training
  set, so this checks wiring, not accuracy.*
- **Live path:** the same photos played as a 45 s video (each held 1.3 s, blank between)
  through the live pipeline. 15 letters were committed, all while that letter's photo was
  showing, apart from an I during the static J photo (upstream rule: a motionless J is I).
  12 letters were never committed (no stable detection on these upscaled tiles). No letter
  was committed while a different photo was showing.
- **Our accuracy on real fingerspelling, and AUTO-mode routing accuracy:** not measured.
  There is no labeled fingerspelling video. A short webcam session showed the app working,
  but there was no ground truth.

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
- **Live mode:**
  - It doesn't segment signs or recognise sentences; it classifies overlapping 2 s windows
    as isolated signs. Transitions between signs can briefly show wrong labels, usually as
    "Not sure".
  - To repeat the **same** sign, lower your hands for about one window (~2 s) first. Otherwise
    the duplicate guard treats both as one sign. A looping 2.4 s clip is accepted only once
    for this reason.
  - Signs faster than about 8 frames (0.27 s) of visible hands are ignored. Signs longer than
    2 s get cut off.
  - Holistic drops about 25–40 % of camera frames before landmarking, compensated by
    timestamp resampling. Fast motion may be undersampled compared with the 30 fps training
    data.
  - Accuracy in live mode **has not been measured**. The 20 s webcam test accepted "hello"
    and "please", but I didn't record what was actually signed, so treat that only as proof
    the system runs. Upstream's 72 % figure is for pre-segmented clips; sliding windows will
    likely do worse.
- **Alphabet / unified app:**
  - AUTO's word-vs-letter routing is **not validated**. It shows AMBIGUOUS whenever both
    models are confident, which happens often with static handshapes such as hello vs B.
    Use WORD or SPELL mode when you know which you're signing.
  - Alphabet scores are uncalibrated softmax; only the 0.75 commit threshold comes with
    upstream precision figures. Upstream lists M, N, S and T as easily confused, and N as
    the weakest letter (54 % recall).
  - J and Z use upstream's hand-tuned motion rules. Sign them deliberately; fast motion
    makes MediaPipe lose the hand. A still image can't show J or Z.
  - With both models running, letters arrive at about 22 per s instead of 30, and the word
    model gets about 13 landmark frames per s instead of about 22. `--pause-word-in-spell`
    restores 30 per s for letters.
  - The weights' licence: see `models/alphabet/ATTRIBUTION.md` (GPL-2.0 training data;
    unresolved).
  - TTS uses macOS `say`. On other platforms S reports "TTS not available".
- **Webcam demo:** it's a fixed 2 s window, so starting late or signing slowly cuts the sign
  off. Keep your hands down until the bar starts and sign promptly. Each result arrives 3–7 s
  after recording. Writing the temp MP4 adds a little time on top of `/predict`'s cost. Its
  accuracy on live webcam signing has not been measured; one untested live capture came back
  "Not sure". Pressing Q while a prediction runs waits for it to finish before exiting.

## Layout

```
app/main.py            FastAPI app, endpoints, error handling, upload handling
app/model.py           ONNX Runtime wrapper, calibration, top-k
app/preprocessing.py   video decode, MediaPipe Holistic, trimming, features
app/live.py            continuous mode: ring buffer, timestamp resampling, scheduler, smoothing
app/gesturesign/       upstream feature code (vendored verbatim, MIT)
app/asl_alphabet/      alphabet upstream code (vendored verbatim, MIT): preprocessing,
                       smoothing, motion, spelling (SpellingBuffer + thresholds)
app/models/alphabet_recognizer.py   AlphabetModel, HandDetector, AlphabetStream, image/video helpers
app/services/          recognition_router.py (decision layer), unified_session.py (one camera,
                       both pipelines), text_buffer.py, speech.py
app/schemas.py         Pydantic response models
app/config.py          settings
models/                pretrained ONNX + metadata + labels; models/alphabet/ (CNN + ATTRIBUTION.md)
scripts/               signbridge_app.py (UNIFIED APP), test_inference.py (CLI),
                       webcam_demo.py (SPACE capture), live_demo.py (continuous), download_model.py
tests/                 test_api.py, test_pipeline.py, test_live.py, test_alphabet.py, test_router.py,
                       test_unified.py, test_api_alphabet.py, data/ (upstream fixtures)
examples/              hello.mp4, thank_you.mp4 (PopSign, CC BY 4.0)
```

## Credits

Model and feature code: GestureSign by Shashank Yadav (MIT). Data: Google – Isolated Sign
Language Recognition (Kaggle asl-signs) / PopSign ASL v1.0, Georgia Tech (CC BY 4.0).
Landmarks: Google MediaPipe (Apache 2.0). Alphabet model and code: MediaPipe-ASL-sign-language-
recognition by Nicolás Florentín (code MIT; weights trained on ASL Alphabet by Akash Nagaraj
(GPL-2.0), ASL Dataset by Ayush Thakur (CC0-1.0), ASL Alphabet Test by Dan Rasband (CC0-1.0)).
