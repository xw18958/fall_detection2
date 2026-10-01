#pragma once
#include <cstdint>
#include "recording_buffer.h"
namespace m5ble {
struct Status {
  RecordState state; uint64_t session; uint32_t samples, pending;
  bool connected, ready, overflow; uint8_t last_error;
};
bool Init();
void Tick();
void Capture(uint64_t time_us, const int16_t raw[6], uint16_t flags);
bool ToggleRecording();
void AddMarker(uint16_t marker = 1);
Status GetStatus();
bool CanLeave();
bool DetectRequested();
bool ResetPairing();
}  // namespace m5ble
