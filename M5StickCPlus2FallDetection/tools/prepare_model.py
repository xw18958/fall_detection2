#!/usr/bin/env python3
import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

EXPECTED_SIZE = 2692976
EXPECTED_SHA256 = "afdf57892d64c5422220d9132b811c1f0ac356b8aedbf219ae7cc3749faa4979"
MEMBER = "model_convert/tf_saved_model/model_float32_float32.tflite"

parser = argparse.ArgumentParser()
parser.add_argument("zip_path", help="Path to model_convert(1).zip")
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
out = root / "main" / "model.tflite"

with zipfile.ZipFile(args.zip_path, "r") as zf:
    names = zf.namelist()
    member = MEMBER if MEMBER in names else next(
        (n for n in names if n.endswith("tf_saved_model/model_float32_float32.tflite")),
        None,
    )
    if member is None:
        raise SystemExit("Could not find model_float32_float32.tflite in the zip")
    data = zf.read(member)

if len(data) != EXPECTED_SIZE:
    raise SystemExit(f"Unexpected model size: {len(data)} (expected {EXPECTED_SIZE})")
sha = hashlib.sha256(data).hexdigest()
if sha != EXPECTED_SHA256:
    raise SystemExit(f"Unexpected model SHA256: {sha}")

out.write_bytes(data)
print(f"Prepared {out} ({len(data)} bytes, sha256={sha})")
