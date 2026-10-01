#pragma once
#include <algorithm>
namespace m5_battery {
constexpr int kUnknown = -1;
// M5Unified Power_Class::getBatteryLevel, pmic_adc, pinned source documented
// in UI_BATTERY.md. A voltage-based estimate, not a coulomb-counted fuel gauge.
inline int PercentageFromMillivolts(int mv) {
  if (mv <= 0) return kUnknown;
  if (mv <= 3300) return 0;
  if (mv >= 4100) return 100;
  return std::clamp((mv - 3300) * 100 / (4150 - 3350), 0, 100);
}
}  // namespace m5_battery
