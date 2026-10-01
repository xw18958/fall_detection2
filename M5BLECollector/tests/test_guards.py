import importlib.util
import json
import runpy
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class Guards(unittest.TestCase):
    def test_usb_upload_and_erase_are_rejected_before_any_io(self):
        for target in ["upload", "uploadfs", "erase", "erase_flash"]:
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError,"USB flashing is disabled"):
                runpy.run_path(str(ROOT/"tools/build_guard.py"),init_globals={
                    "Import": lambda _: None, "env": None, "COMMAND_LINE_TARGETS": [target]})

    def test_wrong_model_is_rejected_without_changing_target(self):
        spec = importlib.util.spec_from_file_location("prepare", ROOT/"tools/prepare_model.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        target = ROOT/"firmware/main/model.tflite"
        old = target.read_bytes() if target.exists() else None
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            for name in ["model.tflite","model_v2_replay.h","model_v2_config.h"]:
                (source/name).write_bytes(b"wrong input")
            with self.assertRaisesRegex(ValueError,"Source model does not match"):
                module.prepare(source)
        self.assertEqual(target.read_bytes() if target.exists() else None,old)

    def test_locked_inference_and_partition_configuration(self):
        spec = importlib.util.spec_from_file_location("verify", ROOT/"tools/verify_preservation.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.verify()

    def test_collection_transfer_is_keep_gated(self):
        ble = (ROOT/"firmware/main/ble_collector.cc").read_text().replace(" ", "")
        buffer = (ROOT/"firmware/main/recording_buffer.h").read_text()
        self.assertIn("if(status.state!=RecordState::Saving)", ble)
        self.assertIn("if (state_ != RecordState::Saving) return 0;", buffer)
        self.assertIn("RecordState::Review", buffer)

    def test_collection_csv_is_sensor_only(self):
        recorder = (ROOT/"macos/Sources/M5BLECollectorCore/Recorder.swift").read_text()
        export = recorder.split("public func exportCSV() throws {",1)[1].split("public func finish",1)[0]
        self.assertIn("seq,device_timestamp_us,ax,ay,az,gx,gy,gz", export)
        self.assertNotIn("host_received_timestamp_utc", export)
        self.assertNotIn("events.csv", export)
        self.assertNotIn("_raw", export)
        self.assertNotIn("_dps", export)

    def test_collection_reads_counts_without_physical_conversion(self):
        main = (ROOT/"firmware/main/main.cc").read_text()
        sensor = main.split("void SensorTask(void*)",1)[1].split("bool ReadLogits",1)[0]
        self.assertIn("ReadImuCounts(counts)", sensor)
        self.assertIn("m5ble::Capture(acquisition_us, counts, flags)", sensor)

if __name__ == "__main__":
    unittest.main()
