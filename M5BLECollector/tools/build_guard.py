"""Verify locked local inputs and prohibit USB flashing from this project."""
Import("env")
from pathlib import Path
import hashlib
import json

if any(t in {"upload", "uploadfs", "erase", "erase_flash"} for t in COMMAND_LINE_TARGETS):
    raise RuntimeError("USB flashing is disabled. Review an application-only OTA image after device backup.")

project = Path(env.subst("$PROJECT_DIR"))
baseline = json.loads((project.parent / "detector_baseline.json").read_text())
for name in ["model.tflite", "model_v2_replay.h", "model_identity.h"]:
    if not (project / "main" / name).is_file():
        raise RuntimeError("Missing private model input; run tools/prepare_model.py first.")
digest = hashlib.sha256((project / "main/model.tflite").read_bytes()).hexdigest()
if digest != baseline["model_sha256"]:
    raise RuntimeError("Model differs from the locked detector baseline; refusing to build.")

import sys
sys.path.insert(0, str(project.parent / "tools"))
from verify_preservation import verify
verify()
