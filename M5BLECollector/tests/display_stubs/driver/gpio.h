#pragma once
#include <cstdint>
using gpio_num_t = int;
constexpr int GPIO_NUM_15=15, GPIO_NUM_13=13, GPIO_NUM_5=5, GPIO_NUM_14=14, GPIO_NUM_12=12, GPIO_NUM_27=27;
constexpr int GPIO_MODE_OUTPUT=1, GPIO_PULLUP_DISABLE=0, GPIO_PULLDOWN_DISABLE=0, GPIO_INTR_DISABLE=0, ESP_OK=0;
struct gpio_config_t { uint64_t pin_bit_mask; int mode, pull_up_en, pull_down_en, intr_type; };
extern int test_data_mode;
inline int gpio_config(const gpio_config_t*) { return ESP_OK; }
inline int gpio_set_level(int pin, int value) { if (pin==14) test_data_mode=value; return ESP_OK; }
