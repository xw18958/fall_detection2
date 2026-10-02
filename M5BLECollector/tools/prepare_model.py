#!/usr/bin/env python3
"""Prepare only the exact verified V4 package pinned by detector_baseline.json."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def prepare(source: Path) -> None:
    baseline = json.loads((ROOT / "detector_baseline.json").read_text())
    names = ["model.tflite", "model_v2_replay.h", "model_v2_config.h"]
    inputs = {name: (source / name).read_bytes() for name in names}
    digest = hashlib.sha256(inputs["model.tflite"]).hexdigest()
    if digest != baseline["model_sha256"]:
        raise ValueError("Source model does not match the locked detector model. Nothing was copied.")
    if hashlib.sha256(inputs["model_v2_replay.h"]).hexdigest() != baseline["replay_sha256"]:
        raise ValueError("Replay vectors differ from the locked baseline. Nothing was copied.")
    config_digest = hashlib.sha256(inputs["model_v2_config.h"]).hexdigest()
    if config_digest != baseline["source_sha256"]["main/model_v2_config.h"]:
        raise ValueError("Normalization/threshold configuration differs. Nothing was copied.")
    target = ROOT / "firmware/main"
    for name in names:
        (target / name).write_bytes(inputs[name])
    (target / "model_identity.h").write_text(
        '#pragma once\nnamespace m5ble {\n'
        f'constexpr char kModelSha256[] = "{digest}";\n'
        f'constexpr char kFirmwareVersion[] = "{baseline.get("firmware_version", "1.3.0-v4-ptq")}";\n}}\n'
    )
    report = {"model_sha256": digest, "bytes": len(inputs["model.tflite"]),
              "replay_sha256": hashlib.sha256(inputs["model_v2_replay.h"]).hexdigest()}
    (ROOT / "firmware/local_model_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared identical model: {digest}; {report['bytes']} bytes. No device access.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT.parent / "M5StickCPlus2FallDetection/main")
    args = parser.parse_args()
    prepare(args.source.resolve())
