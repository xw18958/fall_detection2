from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "firmware/main/main.cc").read_text()
BLE = (ROOT / "firmware/main/ble_collector.cc").read_text()
RECORDER = (ROOT / "macos/Sources/M5BLECollectorCore/Recorder.swift").read_text()


def test_detector_path_remains_present():
    assert "void RunInference()" in MAIN
    assert "fall_display::ShowResult" in MAIN
    assert "kPrototypeFallThreshold" in MAIN


def test_collection_transfer_is_keep_gated():
    assert "RecordState::Saving" in BLE
    assert "if(status.state!=RecordState::Saving)" in BLE.replace(" ", "")
    assert "KeepRecording" in BLE
    assert "DiscardRecording" in BLE


def test_training_csv_is_sensor_only():
    assert "host_received_timestamp_utc" not in RECORDER.split("public func exportCSV() throws {", 1)[1].split("public func finish", 1)[0]
    header = "seq,device_timestamp_us,ax,ay,az,gx,gy,gz\\n"
    assert header in RECORDER
    assert "marker" not in header
