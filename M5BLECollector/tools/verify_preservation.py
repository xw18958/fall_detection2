#!/usr/bin/env python3
"""Check the locked model, inference functions and original detector files."""
import argparse
import hashlib
import json
import re
import subprocess
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

def verify(original=None, firmware_binary=None, firmware_elf=None, nm="xtensa-esp32-elf-nm"):
    baseline = json.loads((ROOT / "detector_baseline.json").read_text())
    target = ROOT / "firmware"
    text = (target / "main/main.cc").read_text()
    for name, expected in baseline["detector_functions_sha256"].items():
        assert digest(function(text, name)) == expected, f"Detector function changed: {name}"
    for name in ["main/model_v2_config.h", "main/input_pipeline.h", "main/CMakeLists.txt", "main/model_data.S.in", "partitions.csv", "main/fast_conv1d.cc", "main/gelu_kernel.cc", "main/fall_op_resolver.cc"]:
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
        identity = (target / "main/model_identity.h").read_text()
        assert f'kModelSha256[] = "{baseline["model_sha256"]}"' in identity, "Advertised identity differs"
        if firmware_binary:
            binary = firmware_binary.read_bytes()
            assert binary.count(data) == 1, "Built application must embed the exact model once"
            assert len(binary) <= 0x3D0000, "Application exceeds the existing OTA slot"
    elif firmware_binary:
        raise AssertionError("Prepare private model inputs first")
    if firmware_elf:
        symbols = subprocess.check_output([nm, str(firmware_elf)], text=True)
        addresses = {name: int(address, 16) for address, name in
                     re.findall(r"^([0-9a-fA-F]+) \w (_binary_model_tflite_(?:start|end))$", symbols, re.M)}
        start = addresses["_binary_model_tflite_start"]
        assert start % 16 == 0, "Embedded FlatBuffer must be 16-byte aligned"
        assert addresses["_binary_model_tflite_end"] - start == len(model.read_bytes()), "Embedded model extent differs"
    print("PASS: locked inference functions, model/configuration, kernels and partitions preserved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path)
    parser.add_argument("--firmware-binary", type=Path)
    parser.add_argument("--firmware-elf", type=Path)
    parser.add_argument("--nm", default="xtensa-esp32-elf-nm")
    args = parser.parse_args()
    verify(args.original, args.firmware_binary, args.firmware_elf, args.nm)
