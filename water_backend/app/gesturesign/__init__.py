"""Preprocessing vendored from GestureSign (MIT, see LICENSE in this folder).

Source: https://github.com/shashank11yadav/GestureSign, commit
ffc0578e68449676b6e9545da82092679914e8c8, ``pipeline/src/gesturesign/``.

``landmarks.py``, ``resample.py``, ``features.py`` and ``types.py`` are byte-for-byte
copies except that the package-absolute imports in ``features.py`` were made relative.
Do not edit them: the ONNX model was trained on exactly these features, and
``tests/test_pipeline.py`` checks them against the upstream parity fixture.
"""
