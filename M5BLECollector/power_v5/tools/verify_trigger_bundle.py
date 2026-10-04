"""Bind a private shadow/cascade bundle to the exact frozen V5 contract."""
import hashlib, json, re
from pathlib import Path

def verify(root, gated=False):
    root = Path(root)
    manifest = json.loads((root / 'trigger_bundle.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for field, filename in [('full_model_sha256', 'model.tflite'),
                            ('firmware_config_sha256', 'model_v2_config.h'),
                            ('header_sha256', 'trigger_bundle.h')]:
        if manifest.get(field) != sha(root / filename):
            raise ValueError('Trigger contract mismatch: ' + field)
    if manifest.get('sample_hz') != 30 or manifest.get('input_shape') != [1, 90, 6]:
        raise ValueError('Trigger must use the frozen 30 Hz, 90x6 window')
    header = (root / 'trigger_bundle.h').read_text()
    match = re.search(r'kQualified=(true|false)', header)
    eligible = bool(manifest.get('deployment_eligible'))
    if not match or (match[1] == 'true') != eligible:
        raise ValueError('Header qualification mismatch')
    if gated and not eligible:
        raise ValueError('Cascade requires locked-data and hardware qualification; use shadow mode')
    print('PASS: exact V5 trigger bundle contract; gated=' + str(gated))
