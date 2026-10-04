import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class UIGuards(unittest.TestCase):
    def test_ui_changes_preserve_backend_and_physical_buttons(self):
        baseline = json.loads((ROOT / 'power_contract.json').read_text())
        spec = importlib.util.spec_from_file_location('verify_ui', ROOT / 'tools/verify_preservation.py')
        verify = importlib.util.module_from_spec(spec); spec.loader.exec_module(verify)
        digest = lambda data: hashlib.sha256(data).hexdigest()
        main = (ROOT / 'firmware/main/main.cc').read_text()
        for name, expected in baseline['functions'].items():
            self.assertEqual(digest(verify.function(main, name)), expected, name)
        for path in ('ui_presentation.h', 'battery_monitor.cc', 'battery_monitor.h', 'ble_collector.cc', 'recording_buffer.h'):
            self.assertEqual(digest((ROOT / 'firmware/main' / path).read_bytes()), baseline['files'][path], path)
        # Power scheduling and screen expiry are authorized changes; collector
        # presentation/backend remain frozen and executable C++ tests cover UI.
        self.assertIn('m5ble::CanLeave()', main)
        self.assertIn('m5ble::Keep', main)
        self.assertIn('m5ble::Discard', main)
