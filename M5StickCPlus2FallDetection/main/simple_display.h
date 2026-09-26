#pragma once

#include <algorithm>
#include <cstdint>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Tiny ST7789V2 driver for the M5StickC PLUS2.
// No graphics library is required: the screen is used only as a fast status UI.
//   cyan  = boot / waiting for first 3 s window
//   green = normal
//   red   = fall trigger
// Bottom bar length = current fall probability.
namespace fall_display {

constexpr gpio_num_t kMosi = GPIO_NUM_15;
constexpr gpio_num_t kClk = GPIO_NUM_13;
constexpr gpio_num_t kCs = GPIO_NUM_5;
constexpr gpio_num_t kDc = GPIO_NUM_14;
constexpr gpio_num_t kRst = GPIO_NUM_12;
constexpr gpio_num_t kBl = GPIO_NUM_27;

constexpr int kWidth = 135;
constexpr int kHeight = 240;
constexpr int kXOffset = 52;
constexpr int kYOffset = 40;
constexpr int kBarY = 220;
constexpr int kBarHeight = 20;

constexpr uint16_t kBlack = 0x0000;
constexpr uint16_t kGreen = 0x07E0;
constexpr uint16_t kRed = 0xF800;
constexpr uint16_t kCyan = 0x07FF;
constexpr uint16_t kWhite = 0xFFFF;

static spi_device_handle_t g_spi = nullptr;
static bool g_ready = false;
static int g_last_state = -1;

inline bool Tx(bool data_mode, const uint8_t* bytes, size_t len) {
  if (g_spi == nullptr || bytes == nullptr || len == 0) return false;
  gpio_set_level(kDc, data_mode ? 1 : 0);
  spi_transaction_t t{};
  t.length = len * 8;
  t.tx_buffer = bytes;
  return spi_device_polling_transmit(g_spi, &t) == ESP_OK;
}

inline bool Command(uint8_t cmd, const uint8_t* data = nullptr, size_t len = 0) {
  if (!Tx(false, &cmd, 1)) return false;
  return len == 0 || Tx(true, data, len);
}

inline void SetWindow(int x, int y, int w, int h) {
  const uint16_t x0 = static_cast<uint16_t>(x + kXOffset);
  const uint16_t x1 = static_cast<uint16_t>(x + w - 1 + kXOffset);
  const uint16_t y0 = static_cast<uint16_t>(y + kYOffset);
  const uint16_t y1 = static_cast<uint16_t>(y + h - 1 + kYOffset);
  const uint8_t cols[4] = {
      static_cast<uint8_t>(x0 >> 8), static_cast<uint8_t>(x0),
      static_cast<uint8_t>(x1 >> 8), static_cast<uint8_t>(x1)};
  const uint8_t rows[4] = {
      static_cast<uint8_t>(y0 >> 8), static_cast<uint8_t>(y0),
      static_cast<uint8_t>(y1 >> 8), static_cast<uint8_t>(y1)};
  Command(0x2A, cols, sizeof(cols));
  Command(0x2B, rows, sizeof(rows));
  Command(0x2C);
}

inline void FillRect(int x, int y, int w, int h, uint16_t color) {
  if (!g_ready || w <= 0 || h <= 0) return;
  x = std::max(0, x);
  y = std::max(0, y);
  w = std::min(w, kWidth - x);
  h = std::min(h, kHeight - y);
  if (w <= 0 || h <= 0) return;

  constexpr int kRowsPerChunk = 8;
  uint8_t pixels[kWidth * kRowsPerChunk * 2];
  for (int i = 0; i < w * kRowsPerChunk; ++i) {
    pixels[2 * i] = static_cast<uint8_t>(color >> 8);
    pixels[2 * i + 1] = static_cast<uint8_t>(color);
  }

  SetWindow(x, y, w, h);
  int remaining = h;
  while (remaining > 0) {
    const int rows = std::min(remaining, kRowsPerChunk);
    Tx(true, pixels, static_cast<size_t>(w * rows * 2));
    remaining -= rows;
  }
}

inline bool Init() {
  gpio_config_t io{};
  io.pin_bit_mask = (1ULL << kDc) | (1ULL << kRst) | (1ULL << kBl);
  io.mode = GPIO_MODE_OUTPUT;
  io.pull_up_en = GPIO_PULLUP_DISABLE;
  io.pull_down_en = GPIO_PULLDOWN_DISABLE;
  io.intr_type = GPIO_INTR_DISABLE;
  if (gpio_config(&io) != ESP_OK) return false;

  gpio_set_level(kBl, 0);
  gpio_set_level(kRst, 0);
  vTaskDelay(pdMS_TO_TICKS(20));
  gpio_set_level(kRst, 1);
  vTaskDelay(pdMS_TO_TICKS(120));

  spi_bus_config_t bus{};
  bus.mosi_io_num = kMosi;
  bus.miso_io_num = -1;
  bus.sclk_io_num = kClk;
  bus.quadwp_io_num = -1;
  bus.quadhd_io_num = -1;
  bus.max_transfer_sz = kWidth * 8 * 2;
  if (spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO) != ESP_OK) return false;

  spi_device_interface_config_t dev{};
  dev.clock_speed_hz = 20 * 1000 * 1000;
  dev.mode = 0;
  dev.spics_io_num = kCs;
  dev.queue_size = 1;
  if (spi_bus_add_device(SPI2_HOST, &dev, &g_spi) != ESP_OK) return false;

  Command(0x01);  // software reset
  vTaskDelay(pdMS_TO_TICKS(120));
  Command(0x11);  // sleep out
  vTaskDelay(pdMS_TO_TICKS(120));
  const uint8_t pixel_format = 0x55;  // RGB565
  Command(0x3A, &pixel_format, 1);
  const uint8_t madctl = 0x00;
  Command(0x36, &madctl, 1);
  Command(0x21);  // display inversion on (ST7789 panels)
  Command(0x29);  // display on
  vTaskDelay(pdMS_TO_TICKS(20));

  g_ready = true;
  gpio_set_level(kBl, 1);
  FillRect(0, 0, kWidth, kHeight, kCyan);
  g_last_state = -1;
  return true;
}

inline void ShowResult(float fall_probability, bool fall_triggered) {
  if (!g_ready) return;
  fall_probability = std::max(0.0f, std::min(1.0f, fall_probability));
  const int state = fall_triggered ? 1 : 0;

  // The expensive large fill happens only when the NORMAL/FALL state changes.
  if (state != g_last_state) {
    FillRect(0, 0, kWidth, kBarY, fall_triggered ? kRed : kGreen);
    g_last_state = state;
  }

  // Small probability bar; only ~5 KB of SPI traffic per inference.
  FillRect(0, kBarY, kWidth, kBarHeight, kBlack);
  const int bar_width = static_cast<int>(fall_probability * kWidth + 0.5f);
  if (bar_width > 0) {
    FillRect(0, kBarY, bar_width, kBarHeight,
             fall_triggered ? kWhite : kCyan);
  }
}

}  // namespace fall_display
