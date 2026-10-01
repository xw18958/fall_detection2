#pragma once

#include <cstdint>
#include <cstring>
#include "esp_log.h"
#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"

// Bounded profiling storage; disabled after the first live prediction.
class InferenceProfiler final : public tflite::MicroProfilerInterface {
 public:
  uint32_t BeginEvent(const char* tag) override {
    if (!enabled_) return kDisabled;
    uint32_t i = 0;
    for (; i < used_; ++i) if (std::strcmp(rows_[i].tag, tag) == 0) break;
    if (i == used_) {
      if (used_ == kRows) return kDisabled;
      rows_[used_++].tag = tag;
    }
    rows_[i].started = esp_timer_get_time();
    return i;
  }
  void EndEvent(uint32_t handle) override {
    if (handle == kDisabled || handle >= used_) return;
    rows_[handle].us += esp_timer_get_time() - rows_[handle].started;
    ++rows_[handle].calls;
  }
  void Reset() { used_ = 0; for (auto& r : rows_) r = {}; }
  void Dump() const {
    for (uint32_t i = 0; i < used_; ++i)
      ESP_LOGI("fall_profile", "%s calls=%u total_us=%lld", rows_[i].tag,
               static_cast<unsigned>(rows_[i].calls),
               static_cast<long long>(rows_[i].us));
  }
  void Disable() { enabled_ = false; }
 private:
  static constexpr uint32_t kRows = 32, kDisabled = UINT32_MAX;
  struct Row { const char* tag = nullptr; int64_t started = 0, us = 0; uint32_t calls = 0; };
  Row rows_[kRows]{};
  uint32_t used_ = 0;
  bool enabled_ = true;
};
