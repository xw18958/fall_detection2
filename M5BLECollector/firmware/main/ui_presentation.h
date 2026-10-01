#pragma once
#include <cstdint>
#include <cstdio>
#include "ble_collector.h"
#include "model_v2_config.h"

namespace device_ui {
constexpr int64_t kConfirmationUs = 1800000;
struct CollectionView {
  m5ble::RecordState screen;
  bool error;
};
// Presentation-only timers: never call commands or mutate the collector Status.
class CollectionPresentation {
 public:
  CollectionView Observe(const m5ble::Status& status, int64_t now) {
    if (!initialized_ || state_ != status.state || session_ != status.session) {
      if (status.state == m5ble::RecordState::Complete) completed_at_ = now;
      state_ = status.state; session_ = status.session; initialized_ = true;
    }
    if (status.last_error && status.last_error != error_) error_at_ = now;
    error_ = status.last_error;
    auto screen = status.state;
    if (screen == m5ble::RecordState::Complete && now - completed_at_ >= kConfirmationUs)
      screen = m5ble::RecordState::Ready;
    return {screen, error_ != 0 && now - error_at_ < kConfirmationUs};
  }
 private:
  bool initialized_ = false;
  m5ble::RecordState state_ = m5ble::RecordState::Ready;
  uint64_t session_ = 0;
  uint8_t error_ = 0;
  int64_t completed_at_ = 0, error_at_ = 0;
};
inline void FormatDuration(uint32_t samples, char out[16]) {
  const uint64_t tenths = uint64_t(samples) * 10 / fall_v2::kSampleHz;
  std::snprintf(out, 16, "%02lu:%02lu.%lu", static_cast<unsigned long>(tenths / 600),
                static_cast<unsigned long>((tenths / 10) % 60), static_cast<unsigned long>(tenths % 10));
}
inline uint32_t SavedCount(const m5ble::Status& s) { return s.samples >= s.pending ? s.samples - s.pending : 0; }
inline int ProgressPercent(const m5ble::Status& s) { return s.samples ? int(uint64_t(SavedCount(s)) * 100 / s.samples) : 100; }
}  // namespace device_ui
