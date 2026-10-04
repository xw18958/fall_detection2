import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class V4Preservation(unittest.TestCase):
    def test_only_authorized_detector_pins_changed(self):
        old=json.loads((ROOT/'detector_baseline_v2.json').read_text())
        new=json.loads((ROOT/'detector_baseline.json').read_text())
        changed={'Normalize','PushSample','FillModelInput','ModelReplaySelfTest','InitModel','RunInference'}
        for name,digest in old['detector_functions_sha256'].items():
            if name not in changed:self.assertEqual(new['detector_functions_sha256'][name],digest,name)
        for name,digest in old['source_sha256'].items():
            if name not in {'main/model_v2_config.h','main/fall_op_resolver.cc','main/fall_op_resolver.h'}:self.assertEqual(new['source_sha256'][name],digest,name)
        # Resolver changes are explicitly limited to the verified normalization op.
        resolver=(ROOT/'firmware/main/fall_op_resolver.cc').read_text()
        self.assertIn('AddCustom("LayerNormV4", &layer_norm_registration_)',resolver)
    def test_no_inference_or_conversion_added_to_collection(self):
        main=(ROOT/'firmware/main/main.cc').read_text()
        sensor=main.split('void SensorTask(void*)',1)[1].split('bool ReadLogits',1)[0]
        self.assertIn('m5ble::Capture(acquisition_us, counts, flags)',sensor)
        self.assertIn('ReadImuCounts(counts)',sensor)
        startup=main.split('// Release only RAM allocations after replay;',1)[1].split('} else if (!fall_ota::Start())',1)[0]
        self.assertLess(startup.index('heap_caps_free(g_tensor_arena)'),startup.index('m5ble::Init()'))
    def test_package_keeps_hash_check_and_no_generic_upload(self):
        build=(ROOT/'tools/build_guard.py').read_text()
        self.assertIn("'erase_flash'",build)
        self.assertIn('from verify_power_contract import verify',build)
        main=(ROOT/'firmware/main/main.cc').read_text()
        self.assertIn('std::strcmp(hex, m5ble::kModelSha256)',main)
        self.assertIn('g_input->params.scale != fall_v2::kInputScale',main)
