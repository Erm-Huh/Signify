# ASL alphabet model — attribution and licensing

`asl_cnn_model.onnx` and `class_names.txt` come unmodified from
https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition
at commit `45ff2c0e26925c50f4546073bd573fe3ca7a5e6e` (author: Nicolás Florentín).

- **Code** from that repository, vendored in `app/asl_alphabet/`, is MIT-licensed
  (`app/asl_alphabet/LICENSE`).
- **The weights are not covered by that MIT licence.** Upstream's README says this
  explicitly. The model was trained on three Kaggle datasets:

| Dataset | Role | Licence |
|---|---|---|
| ASL Alphabet (Akash Nagaraj) | main training set, one subject | **GPL-2.0** |
| ASL Dataset (Ayush Thakur) | extra signers in training | CC0-1.0 |
| ASL Alphabet Test (Dan Rasband) | held-out test set | CC0-1.0 |

**Unresolved:** whether trained weights are a derivative work of GPL-2.0 training data is an
open legal question. Upstream asks redistributors to credit the datasets above and to check
those terms against their use. This project redistributes the weights for a non-commercial
hackathon demo with that credit. Review this before any commercial or public release.

`tests/data/alphabet_reference_landmarks.json` contains hand-landmark coordinates and logits
derived from upstream's `docs/asl-alphabet-reference.png`. The photos themselves are not
included.
