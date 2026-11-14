"""ASL alphabet (A-Z fingerspelling) code vendored from upstream.

Source: https://github.com/punpuniacitizen/MediaPipe-ASL-sign-language-recognition
commit 45ff2c0e26925c50f4546073bd573fe3ca7a5e6e, by Nicolás Florentín. Code is MIT
(``LICENSE`` in this folder).

* ``preprocessing.py``, ``smoothing.py``, ``motion.py``: byte-for-byte copies.
* ``spelling.py``: the threshold constants and ``SpellingBuffer`` class copied verbatim out
  of upstream ``realtime_translator.py`` (the rest of that script is GUI code).

The model weights (``models/alphabet/asl_cnn_model.onnx``) are NOT covered by the MIT
licence: upstream states they derive from Kaggle datasets under GPL-2.0 (ASL Alphabet,
Akash Nagaraj) and CC0-1.0 (ASL Dataset, Ayush Thakur; ASL Alphabet Test, Dan Rasband).
See models/alphabet/ATTRIBUTION.md.
"""
