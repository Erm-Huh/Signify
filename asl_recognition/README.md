# ASL Letter and Word Recognition

A Python 3.11 desktop application for real-time American Sign Language (ASL) letter and isolated-word recognition using OpenCV, MediaPipe, and ONNX.

This application is part of **Signify**, a HackUMass 2026 project focused on making sign language recognition more accessible.

The application supports two recognition modes:

- **Letter Mode:** Recognizes individual ASL alphabet signs, including motion-based letters J and Z.
- **Word Mode:** Recognizes isolated ASL word signs using a pretrained GestureSign model.

Recognized letters and words are displayed in a shared transcript.

**Note:** This application recognizes individual letters and isolated word signs. It does not translate continuous ASL sentences or understand full ASL grammar.

## Features

- Real-time webcam recognition using OpenCV and MediaPipe
- Letter and Word recognition modes
- Mode switching through the interface or keyboard
- Shared transcript across recognition modes
- Automatic word commitment when recognition confidence meets the configured threshold
- Pretrained ONNX models for inference
- Keyboard controls for editing recognized text
- Automated regression tests

## Requirements

- Python 3.11
- Webcam
- Desktop environment with graphical display
- Camera permissions
- Python dependencies listed in `requirements.txt`

No model training or dataset downloads are required to run the application.

The GestureSign word model and its required runtime assets are included in this repository. The alphabet recognition model must be downloaded separately from its original repository.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Erm-Huh/Signify.git
cd Signify/asl_recognition
```

**Important:** Until this project's Pull Request is merged into `main`, clone the feature branch instead:

```bash
git clone -b feature/combined-asl-recognition https://github.com/Erm-Huh/Signify.git
cd Signify/asl_recognition
```

### 2. Create a Python virtual environment

```bash
python3.11 -m venv .venv
```

### 3. Install dependencies

```bash
.venv/bin/python -m pip install -r requirements.txt
```

MediaPipe provides its tracking models through its Python distribution.

### 4. Download the alphabet recognition model

The pretrained alphabet model is not included in this repository because its redistribution permissions have not been fully clarified.

To enable Letter Mode:

1. Visit the [original MediaPipe-ASL-sign-language-recognition repository](https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition).
2. Download `asl_cnn_model.onnx`.
3. Place the file directly inside the `asl_recognition/` directory, alongside `realtime_translator.py`.

Your directory should look like this:

```text
Signify/
└── asl_recognition/
    ├── realtime_translator.py
    ├── asl_cnn_model.onnx
    ├── class_names.txt
    ├── requirements.txt
    ├── vendor/
    └── ...
```

The model is used for local inference and is not redistributed through this repository.

### 5. Run the application

```bash
.venv/bin/python -B realtime_translator.py --camera 0 --word-commit-confidence 0.75
```

Grant camera access when prompted by your operating system.

If your webcam is not detected, try:

```bash
.venv/bin/python -B realtime_translator.py --camera 1
```

To start directly in Word Mode:

```bash
.venv/bin/python -B realtime_translator.py --mode word
```

## Using the Application

### Recognition Modes

**Letter Mode**

Uses the pretrained alphabet recognition model to identify individual ASL letters. Motion tracking supports the dynamic letters J and Z.

**Word Mode**

Uses MediaPipe Holistic landmarks and the GestureSign ONNX model to classify isolated word signs.

Words are automatically appended to the transcript only after a completed gesture segment passes the model's acceptance checks and the configured confidence threshold (default: `0.75`).

Recognition accuracy depends on camera conditions, gesture execution, and model limitations.

### Keyboard Controls

| Key | Action |
|---|---|
| `M` | Switch between Letter and Word Mode |
| `Q` | Quit the application |
| `Space` | Insert a space |
| `Backspace` | Delete the previous character |
| `C` | Clear the transcript |
| `R` | Reset recognition state |
| `U` | Undo the previous transcript action |

You can also switch recognition modes using the LETTER and WORD buttons in the application interface.

For additional controls and limitations, see [COMBINED_APP.md](COMBINED_APP.md).

## Advanced Configuration

The application supports additional command-line arguments:

| Argument | Description |
|---|---|
| `--camera` | Select webcam index |
| `--mode` | Choose the starting recognition mode |
| `--word-commit-confidence` | Set the confidence threshold for automatically committing recognized words |
| `--model` | Override the alphabet ONNX model path |
| `--classes` | Override the alphabet class labels path |
| `--word-repository` | Override the GestureSign runtime repository path |
| `--word-model-dir` | Override the word model directory containing the ONNX model and JSON metadata |

Model paths default to locations relative to the application, so launching the script using an absolute path is also supported.

For the full list of options:

```bash
.venv/bin/python -B realtime_translator.py --help
```

## Testing

Run the automated test suite:

```bash
.venv/bin/python -B -m unittest discover -v
```

Additional verification commands:

```bash
.venv/bin/python -B realtime_translator.py --help
.venv/bin/python -B word_recognizer.py
```

The automated tests use Python's built-in `unittest` framework and require no additional testing packages.

The test suite covers:

- Letter motion recognition
- Word recognition preprocessing
- ONNX word inference using six attributed replay clips
- Eleven preprocessing parity fixtures
- Transcript behavior
- Recognition mode integration
- Application main-loop behavior using simulated camera and GUI components

**Verification results:**

- 74 automated tests passed in the prepared Python 3.11 environment.
- Both ONNX models loaded successfully during local verification.
- Replay inference tests passed.
- CLI help and simulated application behavior were verified.

These results were obtained using the existing development environment. A completely fresh dependency installation has not yet been independently verified.

Actual webcam recognition accuracy, camera permissions, and graphical interaction still require manual testing.

The `test_word_webcam.py` module also provides a manual webcam testing harness.

## Project Structure

```text
asl_recognition/
├── realtime_translator.py      # Main application
├── ui.py                       # OpenCV user interface
├── preprocessing.py            # Alphabet preprocessing
├── motion.py                   # Dynamic letter recognition
├── smoothing.py                # Prediction smoothing
├── recognition_session.py      # Combined recognition session
├── transcript.py               # Shared transcript
├── word_recognizer.py          # ONNX word recognition
├── word_tracking.py            # Word landmark tracking
├── word_segmenter.py           # Gesture segmentation
├── word_session.py             # Word recognition session
├── class_names.txt             # Alphabet labels
├── requirements.txt            # Python dependencies
├── vendor/GestureSign/         # Bundled word model and runtime assets
├── docs/                       # Original documentation and notices
├── test_*.py                   # Automated tests
├── COMBINED_APP.md             # Application documentation
├── WORD_RECOGNITION.md         # Word recognition documentation
├── WORD_WEBCAM.md              # Webcam testing documentation
├── MODEL_MANIFEST.json         # Model metadata and provenance
├── THIRD_PARTY_NOTICES.md      # Third-party licensing information
└── README.md
```

After manual installation, `asl_cnn_model.onnx` should also be present in this directory.

## Packaging and Licenses

This application integrates components from two open-source projects.

### Alphabet Recognition

Original project: [MediaPipe-ASL-sign-language-recognition](https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition)

The original project's source code is MIT-licensed. However, the pretrained alphabet model weights are not automatically covered by that license, and their redistribution permissions remain unresolved.

For this reason, the alphabet ONNX model is not distributed in this repository. Users should obtain it directly from the original project and review its licensing information.

The original project's documentation is preserved in `docs/ORIGINAL_README.md`.

### Word Recognition

Original project: [GestureSign](https://github.com/shashank11yadav/GestureSign)

The application uses GestureSign's pretrained ONNX model and selected runtime preprocessing components.

Only the required runtime modules, model assets, labels, and regression fixtures are included under `vendor/GestureSign/`. The original internal paths are retained for compatibility.

GestureSign's original MIT license and relevant attribution information are preserved.

### Third-Party Notices

Please review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing any third-party assets.

Additional information is available in:

- [MODEL_MANIFEST.json](MODEL_MANIFEST.json) — Model hashes, provenance, source revisions, and metadata
- [ADDED_FILES.txt](ADDED_FILES.txt) — Packaged file inventory
- [COMBINED_APP.md](COMBINED_APP.md) — Combined application implementation and limitations

## Limitations

- Recognition is limited to individual ASL letters and isolated word signs.
- Continuous ASL sentence translation is not supported.
- Recognition quality may vary with lighting, camera positioning, background, and signing style.
- The word recognition model supports a fixed vocabulary.
- Automatic recognition can produce incorrect predictions.
- Live webcam behavior has not been fully validated across operating systems and devices.
- The alphabet model must be downloaded separately.

## Credits

This application builds upon the work of the developers of:

- [MediaPipe-ASL-sign-language-recognition](https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition)
- [GestureSign](https://github.com/shashank11yadav/GestureSign)

We acknowledge and appreciate the original authors and contributors whose open-source work made this project possible.

Developed as part of **Signify — HackUMass 2026**.
