#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "model_v2_config.h"

namespace fall_v2 {
// DETECT only. COLLECT keeps original signed sensor counts.
inline float PhysicalValue(int channel, int16_t count) {
  return channel < 3
      ? (static_cast<float>(count) * (8.0f / 32768.0f)) * kGravity
      : (static_cast<float>(count) * (2000.0f / 32768.0f)) * kDegreesToRadians;
}
inline float NormalizeM5(int channel, float physical) {
  return (physical - kMean[channel]) / kSigma[channel];
}
inline int8_t QuantizeInput(float normalized, float scale, int zero) {
  const int32_t q = static_cast<int32_t>(std::lround(normalized / scale)) + zero;
  return static_cast<int8_t>(std::max<int32_t>(-128, std::min<int32_t>(127, q)));
}
}  // namespace fall_v2
