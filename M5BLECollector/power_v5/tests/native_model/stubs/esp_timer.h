#pragma once
#include <cstdint>
// Only used by ESP-NN diagnostic counters; native timing is not device timing.
inline int64_t esp_timer_get_time() { return 0; }
