#include "power_management.h"
#include "power_config.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "esp_log.h"
namespace fall_power {
namespace {
esp_pm_lock_handle_t fast=nullptr;
SemaphoreHandle_t clock_mutex=nullptr;
unsigned depth=0;
int normal_mhz=80;
esp_err_t Configure(int maximum) {
  esp_pm_config_t config{};
  config.max_freq_mhz=maximum;
  config.min_freq_mhz=CONFIG_FALL_LEGACY_POWER_BASELINE?240:80;
  config.light_sleep_enable=!CONFIG_FALL_LEGACY_POWER_BASELINE && CONFIG_FREERTOS_USE_TICKLESS_IDLE;
  return esp_pm_configure(&config);
}
}
bool InitPowerManagement() {
#if CONFIG_PM_ENABLE
  // Configure() runs around each full inference; IDF's INFO clock messages
  // would otherwise add serial traffic and delay to every scheduled window.
  esp_log_level_set("pm",CONFIG_FALL_DEBUG_LOGS?ESP_LOG_DEBUG:ESP_LOG_WARN);
  normal_mhz=CONFIG_FALL_LEGACY_POWER_BASELINE?240:80;
  clock_mutex=xSemaphoreCreateRecursiveMutex();
  if(!clock_mutex || Configure(normal_mhz)!=ESP_OK ||
     esp_pm_lock_create(ESP_PM_CPU_FREQ_MAX,0,"full_tcn",&fast)!=ESP_OK) return false;
  ESP_LOGI("fall_power","Active CPU: normal=%d MHz, TCN=240 MHz; automatic sleep=%d",normal_mhz,int(!CONFIG_FALL_LEGACY_POWER_BASELINE && CONFIG_FREERTOS_USE_TICKLESS_IDLE));
  return true;
#else
  ESP_LOGW("fall_power","PM disabled; no frequency/sleep savings claimed");return true;
#endif
}
FastCpu::FastCpu() {
  if(!fast) return;
  // IDF's per-core RTOS locks select configured MAX while tasks are runnable.
  // Merely setting MIN=80,MAX=240 does not execute an active trigger at 80 MHz.
  xSemaphoreTakeRecursive(clock_mutex,portMAX_DELAY);
  if(depth++==0) ESP_ERROR_CHECK(Configure(240));
  ESP_ERROR_CHECK(esp_pm_lock_acquire(fast));
}
FastCpu::~FastCpu() {
  if(!fast) return;
  ESP_ERROR_CHECK(esp_pm_lock_release(fast));
  if(--depth==0) ESP_ERROR_CHECK(Configure(normal_mhz));
  xSemaphoreGiveRecursive(clock_mutex);
}
} // namespace fall_power
