# Signify website (wireframe V4)

Plain HTML/CSS/JS, no build step. Fonts come from Google Fonts (DM Sans). GitHub Pages can serve this folder as is.

| File | Page |
|---|---|
| `index.html` | Home |
| `visit.html` | Translate: camera, tap to say, speaker box (speech to text + name catching), text to speech, visit summary |
| `why.html` | Why it matters |
| `how.html` | How it works |
| `team.html` | Team (headshots in `assets/`) |
| `assets/view.js` | Small template runtime the pages use (documented at the top of the file) |

Each page has a `<template id="view">` with the design's markup and a `class Component` with its state and behavior.

## How the Translate page gets signs (hybrid: the website + the laptop)

The website itself never touches the camera. Blake's bridge (`Blake_frontend/server/ws_bridge.py`) runs on the
laptop, owns the webcam, runs the word + letter models from `water_backend/` and decides what gets added to the
sentence. The Translate page connects to it over a WebSocket and shows its camera preview, the live sign and the
sentence, and sends commands back (capture, confirm, mode, word/letter pick, undo, clear).

```
webcam -> ws_bridge.py (laptop) --ws://localhost:8765--> visit.html (localhost or GitHub Pages)
```

The page works the same whether it is opened from `localhost` or from GitHub Pages: it always connects to the
bridge address in **Settings** (default `ws://localhost:8765`).

## Run the demo on the laptop

One-time setup (Python 3.11 is required by MediaPipe 0.10.21; `uv` downloads it):

```bash
cd water_backend
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt     # macOS/Linux: .venv/bin/python
```

Every time:

```bash
# terminal 1: the bridge (camera 0)
cd water_backend
.venv/Scripts/python.exe ../Blake_frontend/server/ws_bridge.py           # macOS/Linux: .venv/bin/python
#   no camera? replay a clip:  ... ws_bridge.py --source examples/hello.mp4

# terminal 2: the website (or just open the GitHub Pages URL)
cd joa_frontend
python -m http.server 5173
```

Open http://localhost:5173/visit.html. The badge above the camera should say **Sign bridge connected**.

- **GitHub Pages + laptop:** open the Pages URL on the laptop that runs the bridge. Chrome may ask to allow
  access to devices on your local network the first time; allow it.
- **The sentence** builds up word by word in the box under the conversation ("Sign, tap or type to start").
  It is editable: click in and type over a wrong sign. **Say it** (or Enter) speaks it and posts it to the
  conversation; **Undo** drops the last word; **Start over** clears it.
- Keyboard: P pause camera · Space end the word (or capture a word sign when there are no letters) · Enter confirm ·
  M mode · W / L pick word or letter · R repeat letter · Backspace undo · C clear.
  Tab is left for keyboard navigation (Blake's page uses Tab for mode; here it is M).

## Pause camera (privacy)

**Pause camera** (or **P**) next to the camera: the page hides the picture at once, and the bridge
**releases the webcam**, so the camera light goes off and nothing is read. Use it while changing, or while
waiting alone. **Turn camera back on** reopens it. The pause is sticky: reloading the page or
reconnecting does not turn the camera back on by itself. Typing in the box, Undo / Start over, Tap to speak
and text to speech all keep working while paused.

## Spaces between fingerspelled words

ASL has no standard "space" sign. Fluent signers mark the end of a fingerspelled word with a short pause
or by moving or lowering the hand, so that is what Signify uses, plus a dictionary:

1. **Lower your hand** (out of the camera) for about 1 second after a spelled word: a space is added.
   Change the delay with `SIGNBRIDGE_AUTO_SPACE_S` (seconds, `0` turns it off).
2. **Intelligent spacing.** When a word ends (hand lowered, **Space ␣** button, Space key, or a word sign
   right after letters), the letter run is split into known words: `HELLOMYNAMEIS` → `HELLO MY NAME IS`,
   `IHAVEAHEADACHE` → `I HAVE A HEADACHE`. It is conservative: a run is only split when every piece is a
   known word, so names stay whole (`OLVERA`, `MAYA`), except a name right after "is / am / called / Dr /
   nurse" (`MYNAMEISJON` → `MY NAME IS JON`). The vocabulary is ~700 common and clinic words plus the
   word model's 250 sign labels: `water_backend/app/services/smart_spacing.py` (add words to `_COMMON`).
3. **Space ␣** button (any mode) or the Space key for an explicit break.

Backspace still removes one letter or one inserted space at a time.

## Only confident signs are shown

The page never shows a guess. The overlay on the camera names a letter or word **only after the
bridge has committed it**, and that is the same moment it is added to the sentence box:

| What the overlay shows | When |
|---|---|
| `—` "Show your hands, chest up" | no hand / signer in frame |
| `…` "Reading your sign…" + bar | a hand is there but no model is confident yet. The bar is how close the reading is to that model's own threshold, not the guess |
| `…` "Almost. Hold steady" | a model is over its threshold but the reading has not held steady long enough yet |
| **`B`** "Added ✓ letter" / **`HELLO`** "Added ✓ word" | committed: shown for 1.6 s and appended to the sentence box at the same time |
| "Either" + Word / Letter buttons | both models are confident at once (Auto mode only). Nothing is added until you pick |

"Confident" is decided in `water_backend`, never in the page, and each model is judged against
its own bar (the two scores are never compared):

| Model | Confident when | Then committed when | Where |
|---|---|---|---|
| Letters (A–Z CNN) | 8-frame average softmax ≥ **0.75** | the same letter for **10** consecutive still frames (no immediate repeat; R repeats) | `water_backend/app/asl_alphabet/spelling.py` (`COMMIT_CONFIDENCE`, `COMMIT_FRAMES`) |
| Words (GestureSign, 250 signs) | calibrated probability ≥ **0.29** with the signer in frame | **2** agreeing windows in a row (`min_stable_windows`) | threshold: `notSureBelow` in `water_backend/models/model-meta.json`; stability: `water_backend/app/live.py` |

To make it stricter or looser, change those constants in `water_backend`. `WORD_BAR` / `LETTER_BAR`
at the top of `visit.html`'s script only draw the progress bar; keep them in step with the
backend values. Note that `tests/test_alphabet.py` pins the letter values to upstream's
(0.50 / 0.75 / 10), so update that test if you change them on purpose.

## The letter model: MediaPipe-ASL-sign-language-recognition

The A–Z fingerspelling model comes from
[punpuniacitizen/MediaPipe-ASL-sign-language-recognition](https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition)
(Nicolás Florentín; code MIT, weights trained on Kaggle data, see licensing below). **It is already
incorporated.** The download `MediaPipe-ASL-sign-language-recognition-main.zip` was checked file by
file against this repo: the model, the labels and the pipeline code are identical (upstream commit
`45ff2c0e`), so there is nothing to copy in.

### Where each piece of the zip lives here

| In the zip | In this repo | Notes |
|---|---|---|
| `asl_cnn_model.onnx`, `class_names.txt` | `water_backend/models/alphabet/` | unmodified; 26 classes a–z, 96×96 skeleton input, ONNX Runtime on CPU |
| `preprocessing.py`, `smoothing.py`, `motion.py` | `water_backend/app/asl_alphabet/` | byte-for-byte copies (landmark normalisation + skeleton renderer, 1-euro filter, J/Z motion rules) |
| `realtime_translator.py` | `water_backend/app/asl_alphabet/spelling.py` (constants + `SpellingBuffer`) and `water_backend/app/models/alphabet_recognizer.py` (the per-frame loop) | the rest of that script is its OpenCV window; the website replaces it |
| `ui.py`, `check_render.py`, `iniciar_visualizador.bat` | not used | desktop UI; the Translate page is our UI |
| `prepare_dataset.py`, `train_model.py`, `evaluate.py`, `export_onnx.py`, `requirements-train.txt` | not used | training only (needs TensorFlow in a separate venv, see the zip's README) |
| `LICENSE` | `water_backend/app/asl_alphabet/LICENSE` | weights' dataset credits: `water_backend/models/alphabet/ATTRIBUTION.md` |

### How it runs, end to end

```
webcam -> ws_bridge.py -> MediaPipe Hands (21 landmarks) -> 1-euro smoothing -> normalise + render skeleton
       -> asl_cnn_model.onnx -> softmax, averaged over 8 frames -> J/Z motion check
       -> commit rule (>= 0.75 for 10 still frames) -> text buffer -> "text" + "commits" -> visit.html
```

The word model runs on the same frames in parallel; `recognition_router.py` decides what the
page sees (Auto / Words / Spell, see the table above).

### Updating to a newer version of the upstream model

1. Unzip the new download into its own folder (not inside the repo).
2. Copy `asl_cnn_model.onnx` and `class_names.txt` into `water_backend/models/alphabet/`.
3. If `preprocessing.py`, `smoothing.py` or `motion.py` changed, copy them into
   `water_backend/app/asl_alphabet/`. If `realtime_translator.py` changed its constants or
   `SpellingBuffer`, copy those into `spelling.py`. **Training and live must render the skeleton
   with the same `preprocessing.py`**, which is upstream's main rule.
4. Update the commit hash in `app/asl_alphabet/__init__.py` and `models/alphabet/ATTRIBUTION.md`,
   and the SHA-256 values in `tests/test_alphabet.py`.
5. Run the tests, then the site:
   ```bash
   cd water_backend
   .venv/Scripts/python.exe -m pytest tests/test_alphabet.py -q      # macOS/Linux: .venv/bin/python
   ```
   On Windows with `core.autocrlf=true`, `test_vendored_files_match_upstream` fails only because Git
   checked the `.py` files out with CRLF line endings, so their SHA-256 changes; the other 9 tests
   should pass. It passes on macOS/Linux.

**Licensing:** the code is MIT. The weights are trained partly on a GPL-2.0 dataset, which
upstream flags as an open question. Fine for this non-commercial hackathon demo with credit;
check it before any commercial release (details in `ATTRIBUTION.md`).

**Known limits (from upstream):** 93.7% on an unseen signer. Digits are not supported (0/O, 2/V,
6/W and 9/F are the same handshape). J and Z are motions, detected by rules, so sign them slowly.
Double letters need **R** (or move your hand away and back).

## Speech

### Your voice: ElevenLabs

When `ELEVENLABS_API_KEY` is set in `water_backend/.env` (git-ignored), the bridge speaks with ElevenLabs and
the voice panel shows **ElevenLabs voice**. The page sends the sentence to the bridge over the same WebSocket,
the bridge calls ElevenLabs and sends back the MP3. **The key never reaches the browser or GitHub Pages.**

| Setting | Value | Change with (in `water_backend/.env`) |
|---|---|---|
| Model | `eleven_v4` (most natural; ~1 s per sentence). Falls back to `eleven_multilingual_v2` if refused | `ELEVENLABS_MODEL=eleven_multilingual_v2` (~0.5 s) or `eleven_flash_v2_5` (~0.4 s) |
| Warm | Sarah `EXAVITQu4vr4xnSDxMaL` | `ELEVENLABS_VOICE_WARM=<voice id>` |
| Bright | Jessica `cgSgspJ2msm6clMCkdW9` | `ELEVENLABS_VOICE_BRIGHT=` |
| Deep | Brian `nPczCjzI2devNBz1zQrb` | `ELEVENLABS_VOICE_DEEP=` |

Speed maps to ElevenLabs' 0.8–1.2; volume is applied in the browser. If ElevenLabs fails or takes longer
than 7 s (no key, quota used up, offline), the page says so under the voice panel and uses the most natural
browser voice instead (Edge "Natural" voices first, then Google, then the system voice).
The account is on the free plan: only ElevenLabs' current premade voices work over the API (not library
voices) and the monthly character quota is small. The key also lacks the `voices_read` permission, which
only matters for listing voices. Check it any time with
`.venv/Scripts/python.exe scripts/tts_check.py <folder>` (writes sample MP3s per voice and model).
Code: `water_backend/app/services/elevenlabs.py`, tests: `tests/test_elevenlabs.py`.

### The staff member's voice: Tap to speak (speech to text)

Uses the browser's `SpeechRecognition` (Chrome / Edge). Each finished line is checked for **who is talking**:
a self-introduction with a role keyword switches "You're talking with" and the conversation to that person
and fills in their name.

| They say | Card shows |
|---|---|
| "Hi, I'm Dr. Patel" · "I'm Doctor Patel" | **Dr. Patel · Doctor** |
| "I'm Sarah, I'll be your nurse" · "Nurse Kelly here" | **Nurse Sarah · Nurse** |
| "This is the front desk" · "I'm the receptionist" | Front desk |
| "I'm the pharmacist" | Checkout / pharmacy |
| "I'll go get the doctor" · "Dr. Patel will see you soon" · "I'm not a doctor" | no change (not about the speaker) |

Keywords: doctor, Dr, doc, physician, surgeon, MD, resident, attending, specialist; nurse, RN, LPN, NP, CNA,
medical assistant; receptionist, front desk, registration; pharmacist, checkout, billing (`ROLE_WORDS` at the
top of `visit.html`'s script). **Fix name** still corrects anything it gets wrong.

## Still simulated

- Waiting room "Demo: nurse calls you" and "Request an interpreter".
- Bracketed placeholders in the copy (`[NAME]`, `[DEVPOST LINK]`, `[CITE]` and similar).
