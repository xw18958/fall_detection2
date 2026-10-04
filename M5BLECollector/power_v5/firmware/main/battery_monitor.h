#pragma once
namespace m5_battery {
// Failure is nonfatal; Percentage() remains -1 when no calibrated reading exists.
void Start();
int Percentage();
}  // namespace m5_battery
