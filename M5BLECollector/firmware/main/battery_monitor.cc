#include "battery_monitor.h"
#include "battery_level.h"
#include <atomic>
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace m5_battery {
namespace {
constexpr char kTag[] = "plus2_battery";
// Official M5Unified board_M5StickCPlus2: GPIO38, ADC1 channel 2, 2:1 divider.
constexpr adc_channel_t kChannel = ADC_CHANNEL_2;
std::atomic<int> g_percentage{kUnknown};
adc_oneshot_unit_handle_t g_adc = nullptr;
adc_cali_handle_t g_calibration = nullptr;

bool Initialize() {
  adc_oneshot_unit_init_cfg_t unit{};
  unit.unit_id = ADC_UNIT_1;
  if (adc_oneshot_new_unit(&unit, &g_adc) != ESP_OK) return false;
  adc_oneshot_chan_cfg_t channel{};
  channel.atten = ADC_ATTEN_DB_12;
  channel.bitwidth = ADC_BITWIDTH_12;
  if (adc_oneshot_config_channel(g_adc, kChannel, &channel) == ESP_OK) {
    adc_cali_line_fitting_config_t calibration{};
    calibration.unit_id = ADC_UNIT_1;
    calibration.atten = ADC_ATTEN_DB_12;
    calibration.bitwidth = ADC_BITWIDTH_12;
    // Use factory eFuse calibration. If absent, fail rather than invent Vref.
    if (adc_cali_create_scheme_line_fitting(&calibration, &g_calibration) == ESP_OK)
      return true;
  }
  adc_oneshot_del_unit(g_adc); g_adc = nullptr;
  return false;
}

void PollTask(void*) {
  if (!Initialize()) {
    ESP_LOGW(kTag, "Calibrated battery ADC unavailable; showing --%%");
    vTaskDelete(nullptr); return;
  }
  int64_t maximum_us = 0;
  bool first = true;
  TickType_t last = xTaskGetTickCount();
  while (true) {
    const int64_t started = esp_timer_get_time();
    int total_mv = 0;
    bool valid = true;
    // Eight calibrated conversions reduce noise, only once every three seconds.
    for (int i = 0; i < 8; ++i) {
      int raw = 0, mv = 0;
      if (adc_oneshot_read(g_adc, kChannel, &raw) != ESP_OK ||
          adc_cali_raw_to_voltage(g_calibration, raw, &mv) != ESP_OK || mv <= 0) {
        valid = false; break;
      }
      total_mv += mv;
    }
    const int battery_mv = valid ? (total_mv * 2 + 4) / 8 : 0;
    g_percentage.store(PercentageFromMillivolts(battery_mv), std::memory_order_relaxed);
    const int64_t elapsed = esp_timer_get_time() - started;
    if (elapsed > maximum_us) maximum_us = elapsed;
    if (first) {
      ESP_LOGI(kTag, "ADC1/GPIO38 calibrated, battery=%d mV, read=%lld us; poll=3 s", battery_mv, static_cast<long long>(elapsed));
      first = false;
    }
    if (elapsed > 5000)
      ESP_LOGW(kTag, "Battery read=%lld us (maximum=%lld us)", static_cast<long long>(elapsed), static_cast<long long>(maximum_us));
    vTaskDelayUntil(&last, pdMS_TO_TICKS(3000));
  }
}
}  // namespace

void Start() {
  // Sampling remains on core 1 at priority 8. This ADC-only task uses core 0
  // at priority 1, shares neither I2C nor SPI, and never reads in the UI loop.
  if (xTaskCreatePinnedToCore(PollTask, "battery_3s", 3072, nullptr, 1, nullptr, 0) != pdPASS)
    ESP_LOGW(kTag, "Battery task unavailable; showing --%%");
}
int Percentage() { return g_percentage.load(std::memory_order_relaxed); }
}  // namespace m5_battery
