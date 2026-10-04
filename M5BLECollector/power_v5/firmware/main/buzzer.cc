#include "buzzer.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_pm.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

namespace fall_buzzer {
namespace {
constexpr gpio_num_t kPin = GPIO_NUM_2;
constexpr auto kMode = LEDC_LOW_SPEED_MODE;
constexpr auto kTimer = LEDC_TIMER_0;
constexpr auto kChannel = LEDC_CHANNEL_0;
// REF_TICK is 1 MHz: 2 kHz * 256 counts fits; 2 kHz * 1024 did not.
constexpr int kFrequencyHz = 2000;
constexpr int kDuty = 128;  // 50% of the 8-bit period.
SemaphoreHandle_t g_mutex = nullptr;
esp_pm_lock_handle_t g_sleep_lock = nullptr;
bool g_initialized = false;

esp_err_t Silence() {
  const esp_err_t stop = ledc_stop(kMode, kChannel, 0);
  const esp_err_t pause = ledc_timer_pause(kMode, kTimer);
  return stop != ESP_OK ? stop : pause;
}
}

esp_err_t Init() {
  if (g_initialized) return ESP_OK;
  gpio_config_t gpio{};
  gpio.pin_bit_mask = 1ULL << kPin;
  gpio.mode = GPIO_MODE_OUTPUT;
  esp_err_t err = gpio_config(&gpio);
  if (err != ESP_OK) return err;
  err = gpio_set_level(kPin, 0);
  if (err != ESP_OK) return err;

  g_mutex = xSemaphoreCreateMutex();
  if (!g_mutex) return ESP_ERR_NO_MEM;
  err = esp_pm_lock_create(ESP_PM_NO_LIGHT_SLEEP, 0, "buzzer", &g_sleep_lock);
  if (err == ESP_OK) {
    ledc_timer_config_t timer{};
    timer.speed_mode = kMode;
    timer.timer_num = kTimer;
    timer.duty_resolution = LEDC_TIMER_8_BIT;
    timer.freq_hz = kFrequencyHz;
    timer.clk_cfg = LEDC_USE_REF_TICK;
    err = ledc_timer_config(&timer);
    if (err == ESP_OK) {
      ledc_channel_config_t channel{};
      channel.gpio_num = kPin;
      channel.speed_mode = kMode;
      channel.channel = kChannel;
      channel.timer_sel = kTimer;
      channel.duty = 0;
      err = ledc_channel_config(&channel);
      const esp_err_t idle = Silence();
      if (err == ESP_OK) err = idle;
    }
  }
  if (err != ESP_OK) {
    if (g_sleep_lock) esp_pm_lock_delete(g_sleep_lock);
    g_sleep_lock = nullptr;
    vSemaphoreDelete(g_mutex);
    g_mutex = nullptr;
    return err;
  }
  g_initialized = true;
  return ESP_OK;
}

esp_err_t Beep(int milliseconds) {
  if (milliseconds <= 0) return ESP_OK;
  if (!g_initialized) return ESP_ERR_INVALID_STATE;
  if (xSemaphoreTake(g_mutex, portMAX_DELAY) != pdTRUE) return ESP_ERR_TIMEOUT;
  esp_err_t err = esp_pm_lock_acquire(g_sleep_lock);
  const bool locked = err == ESP_OK;
  if (err == ESP_OK) err = ledc_timer_resume(kMode, kTimer);
  if (err == ESP_OK) err = ledc_set_duty(kMode, kChannel, kDuty);
  if (err == ESP_OK) err = ledc_update_duty(kMode, kChannel);
  if (err == ESP_OK) {
    const TickType_t ticks = pdMS_TO_TICKS(milliseconds);
    vTaskDelay(ticks > 0 ? ticks : 1);
  }
  // Also silence on errors. No persistent PWM or sleep lock after an alert.
  const esp_err_t idle = Silence();
  if (err == ESP_OK) err = idle;
  if (locked) {
    const esp_err_t release = esp_pm_lock_release(g_sleep_lock);
    if (err == ESP_OK) err = release;
  }
  xSemaphoreGive(g_mutex);
  return err;
}
}
