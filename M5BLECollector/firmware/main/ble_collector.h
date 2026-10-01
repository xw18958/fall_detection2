#pragma once
#include <cstdint>
#include "recording_buffer.h"

namespace m5ble {
struct Status {
  RecordState state = RecordState::Ready;
  uint64_t session = 0;
  uint32_t samples = 0;
  uint32_t pending = 0;
  bool connected = false;
  bool ready = false;
  bool overflow = false;
  uint8_t last_error = 0;
};

bool Init();
void Tick();
void Capture(uint64_t time_us, const int16_t raw[6], uint16_t flags);
Status GetStatus();
bool StartRecording();
bool StopRecording();
bool KeepRecording();
bool DiscardRecording();
bool CanLeave();
bool DetectRequested();
bool ResetPairing();
}  // namespace m5ble
