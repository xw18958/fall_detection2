#!/usr/bin/env python3
import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

EXPECTED_SIZE = 704896
EXPECTED_SHA256 = "914736434cafefac9d49d9573d3a537a9a9e18b87c2d5cfc01b5ee26515d9a89"
MEMBER = "model_convert/tf_saved_model/model_float32_integer_quant.tflite"

parser = argparse.ArgumentParser()
parser.add_argument("zip_path", help="Path to model_convert(1).zip")
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
out = root / "main" / "model.tflite"

with zipfile.ZipFile(args.zip_path, "r") as zf:
    names = zf.namelist()
    member = MEMBER if MEMBER in names else next(
        (n for n in names if n.endswith("tf_saved_model/model_float32_integer_quant.tflite")),
        None,
    )
    if member is None:
        raise SystemExit("Could not find model_float32_integer_quant.tflite in the zip")
    data = zf.read(member)

if len(data) != EXPECTED_SIZE:
    raise SystemExit(f"Unexpected model size: {len(data)} (expected {EXPECTED_SIZE})")
sha = hashlib.sha256(data).hexdigest()
if sha != EXPECTED_SHA256:
    raise SystemExit(f"Unexpected model SHA256: {sha}")

out.write_bytes(data)
print(f"Prepared {out} ({len(data)} bytes, sha256={sha})")
