#pragma once
#include "esp_err.h"

namespace fall_buzzer {
// GPIO2 stays low and the dedicated PWM timer stays paused between beeps.
esp_err_t Init();
esp_err_t Beep(int milliseconds);
}
