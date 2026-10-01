#!/usr/bin/env python3
"""Check the locked model, inference functions and original detector files."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def digest(data):
    return hashlib.sha256(data).hexdigest()

def function(text, name):
    match = re.search(r"^(?:inline )?(?:bool|void|float) " + name + r"\([^\n]*\)\s*\{", text, re.M)
    if not match:
        raise AssertionError(f"Missing detector function: {name}")
    index = text.index("{", match.start()) + 1
    depth = 1
    while depth:
        depth += (text[index] == "{") - (text[index] == "}")
        index += 1
    return text[match.start():index].encode()

def verify(original=None, firmware_binary=None):
    baseline = json.loads((ROOT / "detector_baseline.json").read_text())
    target = ROOT / "firmware"
    text = (target / "main/main.cc").read_text()
    for name, expected in baseline["detector_functions_sha256"].items():
        assert digest(function(text, name)) == expected, f"Detector function changed: {name}"
    for name in ["main/model_v2_config.h", "partitions.csv", "main/fast_conv1d.cc", "main/gelu_kernel.cc", "main/fall_op_resolver.cc"]:
        assert digest((target/name).read_bytes()) == baseline["source_sha256"][name], f"Locked detector input changed: {name}"
    if original:
        for name, expected in baseline["source_sha256"].items():
            assert digest((original/name).read_bytes()) == expected, f"Original detector changed during implementation: {name}"
    replay = target / "main/model_v2_replay.h"
    if replay.exists():
        assert digest(replay.read_bytes()) == baseline["replay_sha256"], "Model replay vectors changed"
    model = target / "main/model.tflite"
    if model.exists():
        data = model.read_bytes()
        assert digest(data) == baseline["model_sha256"], "Model bytes changed"
        if firmware_binary:
            binary = firmware_binary.read_bytes()
            assert binary.count(data) == 1, "Built application must embed the exact model once"
            assert len(binary) <= 0x3D0000, "Application exceeds the existing OTA slot"
    elif firmware_binary:
        raise AssertionError("Prepare private model inputs first")
    print("PASS: locked inference functions, model/configuration, kernels and partitions preserved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path)
    parser.add_argument("--firmware-binary", type=Path)
    args = parser.parse_args()
    verify(args.original, args.firmware_binary)
