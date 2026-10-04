#pragma once
#include "esp_pm.h"
#include "esp_err.h"
#include "sdkconfig.h"
namespace fall_power {
bool InitPowerManagement();
class FastCpu {
 public:
  FastCpu();
  ~FastCpu();
  FastCpu(const FastCpu&) = delete;
  FastCpu& operator=(const FastCpu&) = delete;
};
}  // namespace fall_power
