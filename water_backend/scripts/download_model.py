"""Download the pretrained GestureSign checkpoint into models/ and verify its SHA-256.

    python scripts/download_model.py [--force]

The files are the ONNX export the upstream author trained on the Kaggle asl-signs
dataset (MIT licence; data CC BY 4.0), pinned to one commit.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "ffc0578e68449676b6e9545da82092679914e8c8"
BASE = f"https://raw.githubusercontent.com/shashank11yadav/GestureSign/{COMMIT}/web/public/models"
FILES = {
    "model.onnx": "2fc9038cc409a5117a3eb2bb350b005663957a03e19f24c24b1f209c703c753d",
    "model-meta.json": "1a3f2378c270838f0f315f3151aadd669edf2336a515b8d57be96ce29b8d914d",
    "labels.json": "96244c54104bc5544f57672fdcc1de48ef46f27663ce23c2957c9035209b6439",
}
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# ASL alphabet (A-Z) model. Code MIT; weights derive from GPL-2.0 / CC0 datasets
# (see models/alphabet/ATTRIBUTION.md).
ALPHA_COMMIT = "45ff2c0e26925c50f4546073bd573fe3ca7a5e6e"
ALPHA_BASE = (
    "https://raw.githubusercontent.com/punpuniacitizen/"
    f"MediaPipe-ASL-sign-language-recognition/{ALPHA_COMMIT}"
)
ALPHA_FILES = {
    "asl_cnn_model.onnx": "c3843cabc57032080a81e5995b578139c90631080af1e6d070df1c6dfdc1172b",
    "class_names.txt": "9bcf92a12d8469d0b9b9a8a493e639b51fe3a8b46e05f776ef34cddf5b945847",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "alphabet").mkdir(exist_ok=True)
    jobs = [(f"{BASE}/{n}", MODEL_DIR / n, d) for n, d in FILES.items()]
    jobs += [(f"{ALPHA_BASE}/{n}", MODEL_DIR / "alphabet" / n, d) for n, d in ALPHA_FILES.items()]
    for url, dest, digest in jobs:
        name = str(dest.relative_to(MODEL_DIR))
        if dest.exists() and not args.force and sha256(dest) == digest:
            print(f"ok       {name}")
            continue
        print(f"download {name}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        if sha256(tmp) != digest:
            tmp.unlink()
            print(f"error: checksum mismatch for {name}", file=sys.stderr)
            return 1
        tmp.replace(dest)
        print(f"ok       {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
