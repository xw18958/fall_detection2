import hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('trigger_contract',ROOT/'tools/verify_trigger_bundle.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)

class TriggerContractTests(unittest.TestCase):
    def fixture(self, root):
        files={'model.tflite':b'full model', 'model_v2_config.h':b'frozen config',
               'trigger_bundle.h':b'constexpr bool kQualified=false;'}
        for name,value in files.items(): (root/name).write_bytes(value)
        sha=lambda value:hashlib.sha256(value).hexdigest()
        manifest=dict(full_model_sha256=sha(files['model.tflite']),
            firmware_config_sha256=sha(files['model_v2_config.h']),
            header_sha256=sha(files['trigger_bundle.h']),sample_hz=30,
            input_shape=[1,90,6],deployment_eligible=False)
        (root/'trigger_bundle.json').write_text(json.dumps(manifest))
        return manifest

    def test_unqualified_bundle_only_builds_shadow(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root);guard.verify(root)
            with self.assertRaisesRegex(ValueError,'qualification'):guard.verify(root,gated=True)

    def test_changed_full_model_normalization_or_header_rejected(self):
        for name in ('model.tflite','model_v2_config.h','trigger_bundle.h'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);self.fixture(root);(root/name).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'contract mismatch'):guard.verify(root)

    def test_wrong_sampling_or_shape_rejected(self):
        for field,value in (('sample_hz',20),('input_shape',[1,60,6])):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);manifest=self.fixture(root);manifest[field]=value
                (root/'trigger_bundle.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'30 Hz'):guard.verify(root)

if __name__=='__main__':unittest.main()
