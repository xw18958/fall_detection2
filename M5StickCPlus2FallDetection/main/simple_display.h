#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Tiny ST7789V2 driver for the M5StickC PLUS2.
// Deliberately avoids a graphics dependency: only a small built-in 5x7 font
// and simple rectangles are used so display updates stay cheap.
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
constexpr int kBarY = 218;
constexpr int kBarHeight = 16;

constexpr uint16_t kBlack = 0x0000;
constexpr uint16_t kGreen = 0x07E0;
constexpr uint16_t kRed = 0xF800;
constexpr uint16_t kCyan = 0x07FF;
constexpr uint16_t kWhite = 0xFFFF;
constexpr uint16_t kYellow = 0xFFE0;

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

inline void Glyph(char c, uint8_t out[5]) {
  out[0] = out[1] = out[2] = out[3] = out[4] = 0;
  switch (c) {
    case 'A': { uint8_t v[5]={0x7E,0x11,0x11,0x11,0x7E}; std::copy(v,v+5,out); break; }
    case 'B': { uint8_t v[5]={0x7F,0x49,0x49,0x49,0x36}; std::copy(v,v+5,out); break; }
    case 'C': { uint8_t v[5]={0x3E,0x41,0x41,0x41,0x22}; std::copy(v,v+5,out); break; }
    case 'D': { uint8_t v[5]={0x7F,0x41,0x41,0x22,0x1C}; std::copy(v,v+5,out); break; }
    case 'E': { uint8_t v[5]={0x7F,0x49,0x49,0x49,0x41}; std::copy(v,v+5,out); break; }
    case 'F': { uint8_t v[5]={0x7F,0x09,0x09,0x09,0x01}; std::copy(v,v+5,out); break; }
    case 'G': { uint8_t v[5]={0x3E,0x41,0x49,0x49,0x7A}; std::copy(v,v+5,out); break; }
    case 'H': { uint8_t v[5]={0x7F,0x08,0x08,0x08,0x7F}; std::copy(v,v+5,out); break; }
    case 'I': { uint8_t v[5]={0x00,0x41,0x7F,0x41,0x00}; std::copy(v,v+5,out); break; }
    case 'J': { uint8_t v[5]={0x20,0x40,0x41,0x3F,0x01}; std::copy(v,v+5,out); break; }
    case 'K': { uint8_t v[5]={0x7F,0x08,0x14,0x22,0x41}; std::copy(v,v+5,out); break; }
    case 'L': { uint8_t v[5]={0x7F,0x40,0x40,0x40,0x40}; std::copy(v,v+5,out); break; }
    case 'M': { uint8_t v[5]={0x7F,0x02,0x0C,0x02,0x7F}; std::copy(v,v+5,out); break; }
    case 'N': { uint8_t v[5]={0x7F,0x04,0x08,0x10,0x7F}; std::copy(v,v+5,out); break; }
    case 'O': { uint8_t v[5]={0x3E,0x41,0x41,0x41,0x3E}; std::copy(v,v+5,out); break; }
    case 'P': { uint8_t v[5]={0x7F,0x09,0x09,0x09,0x06}; std::copy(v,v+5,out); break; }
    case 'Q': { uint8_t v[5]={0x3E,0x41,0x51,0x21,0x5E}; std::copy(v,v+5,out); break; }
    case 'R': { uint8_t v[5]={0x7F,0x09,0x19,0x29,0x46}; std::copy(v,v+5,out); break; }
    case 'S': { uint8_t v[5]={0x46,0x49,0x49,0x49,0x31}; std::copy(v,v+5,out); break; }
    case 'T': { uint8_t v[5]={0x01,0x01,0x7F,0x01,0x01}; std::copy(v,v+5,out); break; }
    case 'U': { uint8_t v[5]={0x3F,0x40,0x40,0x40,0x3F}; std::copy(v,v+5,out); break; }
    case 'V': { uint8_t v[5]={0x1F,0x20,0x40,0x20,0x1F}; std::copy(v,v+5,out); break; }
    case 'W': { uint8_t v[5]={0x7F,0x20,0x18,0x20,0x7F}; std::copy(v,v+5,out); break; }
    case 'X': { uint8_t v[5]={0x63,0x14,0x08,0x14,0x63}; std::copy(v,v+5,out); break; }
    case 'Y': { uint8_t v[5]={0x03,0x04,0x78,0x04,0x03}; std::copy(v,v+5,out); break; }
    case 'Z': { uint8_t v[5]={0x61,0x51,0x49,0x45,0x43}; std::copy(v,v+5,out); break; }
    case '0': { uint8_t v[5]={0x3E,0x51,0x49,0x45,0x3E}; std::copy(v,v+5,out); break; }
    case '1': { uint8_t v[5]={0x00,0x42,0x7F,0x40,0x00}; std::copy(v,v+5,out); break; }
    case '2': { uint8_t v[5]={0x42,0x61,0x51,0x49,0x46}; std::copy(v,v+5,out); break; }
    case '3': { uint8_t v[5]={0x21,0x41,0x45,0x4B,0x31}; std::copy(v,v+5,out); break; }
    case '4': { uint8_t v[5]={0x18,0x14,0x12,0x7F,0x10}; std::copy(v,v+5,out); break; }
    case '5': { uint8_t v[5]={0x27,0x45,0x45,0x45,0x39}; std::copy(v,v+5,out); break; }
    case '6': { uint8_t v[5]={0x3C,0x4A,0x49,0x49,0x30}; std::copy(v,v+5,out); break; }
    case '7': { uint8_t v[5]={0x01,0x71,0x09,0x05,0x03}; std::copy(v,v+5,out); break; }
    case '8': { uint8_t v[5]={0x36,0x49,0x49,0x49,0x36}; std::copy(v,v+5,out); break; }
    case '9': { uint8_t v[5]={0x06,0x49,0x49,0x29,0x1E}; std::copy(v,v+5,out); break; }
    case '%': { uint8_t v[5]={0x63,0x13,0x08,0x64,0x63}; std::copy(v,v+5,out); break; }
    case ':': { uint8_t v[5]={0x00,0x36,0x36,0x00,0x00}; std::copy(v,v+5,out); break; }
    case '.': { uint8_t v[5]={0x00,0x60,0x60,0x00,0x00}; std::copy(v,v+5,out); break; }
    case '-': { uint8_t v[5]={0x08,0x08,0x08,0x08,0x08}; std::copy(v,v+5,out); break; }
    default: break;
  }
}

inline void DrawChar(int x, int y, char c, int scale, uint16_t fg, uint16_t bg) {
  if (!g_ready || scale < 1 || scale > 3) return;
  uint8_t glyph[5];
  Glyph(c, glyph);
  const int w = 6 * scale;
  const int h = 8 * scale;
  uint8_t pixels[6 * 3 * 8 * 3 * 2];
  int p = 0;
  for (int py = 0; py < h; ++py) {
    const int row = py / scale;
    for (int px = 0; px < w; ++px) {
      const int col = px / scale;
      const bool on = (col < 5 && row < 7 && (glyph[col] & (1U << row)) != 0);
      const uint16_t color = on ? fg : bg;
      pixels[p++] = static_cast<uint8_t>(color >> 8);
      pixels[p++] = static_cast<uint8_t>(color);
    }
  }
  SetWindow(x, y, w, h);
  Tx(true, pixels, static_cast<size_t>(p));
}

inline void DrawText(int x, int y, const char* text, int scale,
                     uint16_t fg, uint16_t bg = kBlack) {
  if (text == nullptr) return;
  while (*text && x + 6 * scale <= kWidth) {
    DrawChar(x, y, *text++, scale, fg, bg);
    x += 6 * scale;
  }
}

inline int CenterX(const char* text, int scale) {
  int n = 0;
  while (text[n]) ++n;
  return std::max(0, (kWidth - n * 6 * scale) / 2);
}

inline void DrawHeader() {
  DrawText(CenterX("FALL", 2), 18, "FALL", 2, kWhite);
  DrawText(CenterX("DETECTOR", 2), 42, "DETECTOR", 2, kWhite);
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
  Command(0x21);  // inversion on
  Command(0x29);  // display on
  vTaskDelay(pdMS_TO_TICKS(20));

  g_ready = true;
  gpio_set_level(kBl, 1);
  FillRect(0, 0, kWidth, kHeight, kBlack);
  DrawHeader();
  DrawText(CenterX("BOOTING", 2), 105, "BOOTING", 2, kCyan);
  DrawText(CenterX("WAIT 3S", 1), 142, "WAIT 3S", 1, kYellow);
  g_last_state = -1;
  return true;
}

inline void ShowResult(float fall_probability, bool fall_triggered, int inference_ms) {
  if (!g_ready) return;
  fall_probability = std::max(0.0f, std::min(1.0f, fall_probability));
  const int state = fall_triggered ? 1 : 0;

  if (state != g_last_state) {
    FillRect(0, 72, kWidth, 55, kBlack);
    const char* status = fall_triggered ? "FALL" : "NORMAL";
    const uint16_t status_color = fall_triggered ? kRed : kGreen;
    DrawText(CenterX(status, 3), 88, status, 3, status_color);
    g_last_state = state;
  }

  char line[20];
  FillRect(0, 135, kWidth, 70, kBlack);
  const int pct = static_cast<int>(fall_probability * 100.0f + 0.5f);
  snprintf(line, sizeof(line), "FALL %d%%", pct);
  DrawText(CenterX(line, 2), 140, line, 2, kWhite);
  snprintf(line, sizeof(line), "%d MS", inference_ms);
  DrawText(CenterX(line, 2), 172, line, 2, kCyan);

  FillRect(0, kBarY, kWidth, kBarHeight, kBlack);
  const int bar_width = static_cast<int>(fall_probability * kWidth + 0.5f);
  if (bar_width > 0) {
    FillRect(0, kBarY, bar_width, kBarHeight,
             fall_triggered ? kRed : kCyan);
  }
}

}  // namespace fall_display
