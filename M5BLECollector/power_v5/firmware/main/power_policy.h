#pragma once
#include <cstdint>

namespace fall_power {
constexpr int kSamples = 90, kAxes = 6;
constexpr int64_t kStrideUs = 250000;
constexpr int64_t kQuietUs = 1000000;
constexpr int64_t kBurstUs = 8000000;
// Integer deadlines preserve the fractional 30 Hz period; never add 33333.
inline int64_t SampleDeadline(int64_t origin, uint64_t sample) {
  return origin + static_cast<int64_t>((sample / 30) * 1000000 +
                                      (sample % 30) * 1000000 / 30);
}
inline int64_t StepDeadline(int64_t origin, uint64_t step) {
  return origin + static_cast<int64_t>(step) * kStrideUs;
}
struct Window {
  float x[kSamples][kAxes];
  int64_t deadline_us = 0;
  uint64_t last_sample = 0;
  uint32_t generation = 0;
};
struct Decision { bool run_full = false; bool woke = false; bool capped = false; };
class FallbackCadence {
 public:
  explicit FallbackCadence(int64_t origin) : due_(origin) {}
  bool Step(int64_t nominal_deadline) {
    if(nominal_deadline<due_) return false;
    due_=nominal_deadline+750000;
    return true;
  }
 private:
  int64_t due_;
};
class Policy {
 public:
  Decision Step(int64_t now, bool suspicious) {
    Decision d;
    if (active_ && !suspicious && now - last_ >= kQuietUs) active_ = false;
    if (active_ && now - start_ >= kBurstUs) {
      d.capped = true;
      if(suspicious) start_=now;
      else active_=false;
    }
    // Persistent suspicion is one episode; bounded maintenance rearms are
    // counted separately. Never insert a blind cooldown between real falls.
    if (suspicious) {
      if (!active_) { active_ = true; start_ = now; d.woke = true; }
      last_ = now;
    }
    d.run_full = active_;
    return d;
  }
  bool active() const { return active_; }
  void Reset() { active_ = false; start_ = last_ = 0; }
 private:
  bool active_ = false;
  int64_t start_ = 0, last_ = 0;
};
}  // namespace fall_power
