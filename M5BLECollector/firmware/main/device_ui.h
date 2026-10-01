#pragma once
#include <algorithm>
#include <cstdio>
#include <cstring>
#include "simple_display.h"
#include "ui_presentation.h"

// UI only: fixed-size buffers, no heap allocation, no commands or hardware reads.
namespace device_ui {
using namespace fall_display;
constexpr uint16_t kDivider = 0x4208;
struct TextCache {
  char text[24]{};
  int x = 0, y = 0, scale = 2;
  uint16_t color = kWhite;
  bool valid = false;
};
static TextCache g_text[16];
static int g_screen = -1, g_battery = -999, g_progress = -1;
static bool g_body_changed = false;
static CollectionPresentation g_presentation;

inline int TextWidth(const char* text, int size) { return int(std::strlen(text)) * 6 * size; }
inline void Text(int slot, int x, int y, const char* text, int size, uint16_t color = kWhite) {
  auto& old = g_text[slot];
  const size_t length = std::strlen(text);
  if (length >= sizeof(old.text) || size < 2 || size > 4 || x < 0 || y < 0 ||
      x + TextWidth(text, size) > kWidth || y + 8 * size > kHeight) return;
  const bool same_geometry = old.valid && old.x == x && old.y == y && old.scale == size;
  if (old.valid && !same_geometry)
    FillRect(old.x, old.y, TextWidth(old.text, old.scale), 8 * old.scale, kBlack);
  const size_t previous_length = old.valid ? std::strlen(old.text) : 0;
  for (size_t i = 0; i < length; ++i) {
    if (!same_geometry || i >= previous_length || old.text[i] != text[i] || old.color != color)
      DrawChar(x + int(i) * 6 * size, y, text[i], size, color, kBlack);
  }
  if (same_geometry && previous_length > length)
    FillRect(x + int(length) * 6 * size, y, int(previous_length - length) * 6 * size, 8 * size, kBlack);
  std::memcpy(old.text, text, length + 1);
  old.x = x; old.y = y; old.scale = size; old.color = color; old.valid = true;
}
inline void Center(int slot, const char* text, int y, int size = 2, uint16_t color = kWhite) {
  Text(slot, (kWidth - TextWidth(text, size)) / 2, y, text, size, color);
}
inline void Outline(int x, int y, int w, int h, uint16_t color) {
  FillRect(x, y, w, 1, color); FillRect(x, y + h - 1, w, 1, color);
  FillRect(x, y, 1, h, color); FillRect(x + w - 1, y, 1, h, color);
}
inline void BeginScreen(int screen) {
  g_body_changed = screen != g_screen;
  if (!g_body_changed) return;
  FillRect(0, 62, kWidth, kHeight - 62, kBlack);
  for (int i = 2; i < 16; ++i) g_text[i].valid = false;
  g_progress = -1; g_screen = screen;
}
inline void DrawBattery(int percentage) {
  percentage = percentage < 0 ? -1 : std::min(percentage, 100);
  const uint16_t color = percentage >= 0 && percentage <= 10 ? kRed :
                         percentage >= 0 && percentage <= 20 ? kYellow : kWhite;
  if (percentage != g_battery) {
    FillRect(101, 10, 31, 16, kBlack);
    Outline(102, 10, 25, 14, color);
    FillRect(127, 14, 3, 6, color);
    if (percentage > 0) FillRect(105, 13, 19 * percentage / 100, 8, color);
    g_battery = percentage;
  }
  char label[8];
  if (percentage < 0) std::snprintf(label, sizeof(label), "--%%");
  else std::snprintf(label, sizeof(label), "%d%%", percentage);
  Text(1, 129 - TextWidth(label, 2), 34, label, 2, color);
}
inline void DrawTopBar(const char* mode, int battery) {
  Text(0, 6, 13, mode, 2);
  DrawBattery(battery);
  if (g_body_changed) FillRect(8, 60, kWidth - 16, 1, kDivider);
}
inline void DrawPrimaryAction(int slot, const char* key, const char* label, int y, uint16_t color = kWhite) {
  const int x = (kWidth - (22 + 10 + TextWidth(label, 2))) / 2;
  if (g_body_changed) Outline(x, y, 22, 22, color);
  Text(slot, x + 5, y + 3, key, 2, color);
  Text(slot + 1, x + 32, y + 3, label, 2, color);
}
inline void DrawCheck(int y) {
  if (!g_body_changed) return;
  for (int i = 0; i < 8; ++i) FillRect(46 + 2 * i, y + 14 + 2 * i, 4, 4, kGreen);
  for (int i = 0; i < 15; ++i) FillRect(60 + 2 * i, y + 28 - 2 * i, 4, 4, kGreen);
}
inline void DrawProgressBar(int percentage) {
  if (percentage == g_progress) return;
  if (g_body_changed) Outline(12, 161, 111, 12, kWhite);
  const int width = 107 * std::clamp(percentage, 0, 100) / 100;
  FillRect(14, 163, 107, 8, kBlack);
  if (width) FillRect(14, 163, width, 8, kWhite);
  g_progress = percentage;
}
inline void SampleCount(uint32_t samples, int y, int size = 2) {
  char count[16]; std::snprintf(count, sizeof(count), "%lu", static_cast<unsigned long>(samples));
  Center(5, count, y, size);
  Center(6, "SAMPLES", y + 8 * size + 6);
}
inline void DrawStarting(bool collection) {
  if (!g_ready) return;
  BeginScreen(collection ? 200 : 201); DrawTopBar(collection ? "COLLECT" : "DETECT", -1);
  Center(2, "STARTING", 98);
  Center(3, "PLEASE", 143);
  Center(4, "WAIT", 165);
}
inline void DrawDetectScreen(int battery) {
  if (!g_ready) return;
  const bool alert = g_has_result && g_fall_triggered;
  BeginScreen(g_has_result ? (alert ? 2 : 1) : 3);
  DrawTopBar("DETECT", battery);
  if (!g_has_result) {
    Center(2, "STARTING", 98);
    Center(3, "MONITORING", 137);
  } else if (alert) {
    Center(2, "FALL", 68, 3, kRed);
    Center(3, "DETECTED", 98, 2, kRed);
    if (g_body_changed) {
      FillRect(63, 123, 8, 22, kRed); FillRect(63, 151, 8, 6, kRed);
    }
    Center(4, "ALERT", 158, 3, kRed);
  } else {
    Center(2, "MONITORING", 73);
    DrawCheck(100);
    Center(4, "ACTIVE", 144, 3);
  }
  if (g_has_result) {
    const int tenths = int(g_fall_probability * 1000.0f + 0.5f);
    char probability[24]; std::snprintf(probability, sizeof(probability), "FALL %d.%d%%", tenths / 10, tenths % 10);
    Center(5, probability, 183);
  }
  Center(8, "HOLD B", 202);
  Center(9, "COLLECT", 223);
}
inline void DrawCollectionScreen(const m5ble::Status& status, int battery, int64_t now) {
  if (!g_ready) return;
  const auto view = g_presentation.Observe(status, now);
  int screen = 100 + int(view.screen);
  if (view.screen == m5ble::RecordState::Saving && !status.ready) screen += 10;
  if (view.error) screen = 150;
  BeginScreen(screen); DrawTopBar("COLLECT", battery);
  if (view.error) {
    Center(2, "ACTION", 91, 3, kRed);
    Center(3, "FAILED", 130, 3, kRed);
    if (status.pending) {
      Center(4, "SAMPLES", 187); Center(5, "SAFE", 209);
    } else Center(4, "TRY AGAIN", 191);
    return;
  }
  char duration[16]; FormatDuration(status.samples, duration);
  switch (view.screen) {
    case m5ble::RecordState::Ready:
      Center(2, "READY", 76, 3);
      Center(3, "MAC", 119);
      Center(4, status.ready ? "READY" : status.connected ? "CONNECTING" : "OFFLINE", 141, 2, status.ready ? kGreen : kWhite);
      DrawPrimaryAction(8, "A", "START", 179);
      DrawPrimaryAction(10, "C", "DETECT", 213);
      break;
    case m5ble::RecordState::Recording:
      Center(2, "RECORDING", 77);
      if (g_body_changed) {
        FillRect(36, 112, 8, 8, kRed); FillRect(34, 114, 12, 4, kRed);
      }
      Text(3, 55, 109, "REC", 2, kRed);
      Center(4, duration, 148, 3);
      DrawPrimaryAction(8, "A", "STOP", 207);
      break;
    case m5ble::RecordState::Review:
    case m5ble::RecordState::Full:
      if (view.screen == m5ble::RecordState::Full) {
        Center(2, "BUFFER", 67, 2, kYellow); Center(3, "FULL", 88, 2, kYellow);
      } else Center(2, "FINISHED", 78);
      Center(4, duration, 112, 3);
      SampleCount(status.samples, 148);
      DrawPrimaryAction(8, "A", "KEEP", 187, kGreen);
      DrawPrimaryAction(10, "B", "DISCARD", 215, kYellow);
      break;
    case m5ble::RecordState::Saving:
      Center(2, "SAVING", 75, 3);
      if (status.ready) {
        const int percentage = ProgressPercent(status);
        char progress[16]; std::snprintf(progress, sizeof(progress), "%d%%", percentage);
        Center(3, progress, 116, 4);
        DrawProgressBar(percentage);
        char counts[24]; std::snprintf(counts, sizeof(counts), "%lu/%lu", static_cast<unsigned long>(SavedCount(status)), static_cast<unsigned long>(status.samples));
        Center(4, counts, 196);
        Center(5, "SAMPLES", 218);
      } else {
        Center(3, "WAITING", 112); Center(4, "FOR MAC", 134);
        SampleCount(status.samples, 158, 3);
        Center(7, "SAFE", 210, 2, kGreen);
      }
      break;
    case m5ble::RecordState::Complete:
      DrawCheck(86);
      Center(2, "SAVED", 137, 3, kGreen);
      SampleCount(status.samples, 183);
      break;
  }
}
}  // namespace device_ui
