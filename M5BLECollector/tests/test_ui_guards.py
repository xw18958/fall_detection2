import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class UIGuards(unittest.TestCase):
    def test_ui_changes_preserve_backend_and_physical_buttons(self):
        baseline = json.loads((ROOT / 'tests/ui_preservation_baseline.json').read_text())
        spec = importlib.util.spec_from_file_location('verify_ui', ROOT / 'tools/verify_preservation.py')
        verify = importlib.util.module_from_spec(spec); spec.loader.exec_module(verify)
        digest = lambda data: hashlib.sha256(data).hexdigest()
        main = (ROOT / 'firmware/main/main.cc').read_text()
        for name, expected in baseline['functions'].items():
            self.assertEqual(digest(verify.function(main, name)), expected, name)
        for path, expected in baseline['files'].items():
            self.assertEqual(digest((ROOT.parent / path).read_bytes()), expected, path)
        regions = {
            'buttons': main[main.index('struct Button {'):main.index('}  // namespace', main.index('struct Button {'))],
            'commands_and_wake': main[main.index('    int a = button_a.Poll'):main.index('      if (screen_awake && now - last_ui >= 250000)')],
            'sleep': main[main.index('      const bool may_sleep'):main.index('    portENTER_CRITICAL(&g_ring_lock);', main.index('      const bool may_sleep'))]
        }
        for name, value in regions.items():
            self.assertEqual(digest(value.strip().encode()), baseline['regions'][name], name)
