#include "internet_ota.h"

#include <cctype>
#include <cstdint>
#include <cstring>

#include "esp_crt_bundle.h"
#include "esp_event.h"
#include "esp_http_client.h"
#include "esp_https_ota.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "esp_sntp.h"
#include <ctime>
#if __has_include("wifi_secrets.h")
#include "wifi_secrets.h"
#else
#error "Missing main/wifi_secrets.h. Copy main/wifi_secrets.example.h to main/wifi_secrets.h and set your Wi-Fi credentials."
#endif

namespace fall_internet_ota {
namespace {

constexpr char kTag[] = "internet_ota";
constexpr char kFirmwareVersion[] = "1.0.0";
constexpr char kLatestVersionUrl[] =
    "https://raw.githubusercontent.com/xw18958/fall_detection2/main/M5StickCPlus2FallDetection/ota/latest.txt";
constexpr char kFirmwareUrl[] =
    "https://github.com/xw18958/fall_detection2/releases/latest/download/firmware.bin";
constexpr int kWifiTimeoutMs = 12000;
constexpr int kHttpTimeoutMs = 10000;
constexpr int kSntpTimeoutMs = 8000;
constexpr EventBits_t kConnectedBit = BIT0;
constexpr EventBits_t kFailedBit = BIT1;

EventGroupHandle_t g_wifi_events = nullptr;

struct Version {
  uint32_t part[3]{};
};

bool ParseVersion(const char* text, Version* version) {
  if (text == nullptr || version == nullptr) return false;
  const char* p = text;
  for (int i = 0; i < 3; ++i) {
    if (!std::isdigit(static_cast<unsigned char>(*p))) return false;
    const char* component_start = p;
    uint32_t value = 0;
    do {
      const uint32_t digit = static_cast<uint32_t>(*p - '0');
      if (p == component_start && digit == 0 &&
          std::isdigit(static_cast<unsigned char>(p[1])))
        return false;
      if (value > (UINT32_MAX - digit) / 10) return false;
      value = value * 10 + digit;
      ++p;
    } while (std::isdigit(static_cast<unsigned char>(*p)));
    version->part[i] = value;
    if (i < 2) {
      if (*p++ != '.') return false;
    } else if (*p != '\0') {
      return false;
    }
  }
  return true;
}

int Compare(const Version& a, const Version& b) {
  for (int i = 0; i < 3; ++i) {
    if (a.part[i] < b.part[i]) return -1;
    if (a.part[i] > b.part[i]) return 1;
  }
  return 0;
}

void WifiEventHandler(void*, esp_event_base_t base, int32_t id, void* data) {
  if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
    xEventGroupSetBits(g_wifi_events, kConnectedBit);
  } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
    const auto* event = static_cast<wifi_event_sta_disconnected_t*>(data);
    ESP_LOGW(kTag, "Wi-Fi disconnected (reason=%u)", event->reason);
    xEventGroupSetBits(g_wifi_events, kFailedBit);
  }
}

bool ConnectWifi() {
  if (std::strcmp(FALL_WIFI_SSID, "YOUR_WIFI_SSID") == 0 ||
      std::strcmp(FALL_WIFI_PASSWORD, "YOUR_WIFI_PASSWORD") == 0) {
    ESP_LOGW(kTag, "Set credentials in main/wifi_secrets.h to enable Internet OTA");
    return false;
  }
  g_wifi_events = xEventGroupCreate();
  if (g_wifi_events == nullptr) {
    ESP_LOGE(kTag, "Could not allocate Wi-Fi event group");
    return false;
  }
  esp_event_handler_instance_t wifi_handler;
  esp_event_handler_instance_t ip_handler;
  if (esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
          WifiEventHandler, nullptr, &wifi_handler) != ESP_OK ||
      esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
          WifiEventHandler, nullptr, &ip_handler) != ESP_OK) {
    ESP_LOGE(kTag, "Could not register Wi-Fi event handlers");
    return false;
  }
  wifi_config_t config{};
  std::strncpy(reinterpret_cast<char*>(config.sta.ssid), FALL_WIFI_SSID,
               sizeof(config.sta.ssid) - 1);
  std::strncpy(reinterpret_cast<char*>(config.sta.password), FALL_WIFI_PASSWORD,
               sizeof(config.sta.password) - 1);
  config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
  if (esp_wifi_set_config(WIFI_IF_STA, &config) != ESP_OK ||
      esp_wifi_connect() != ESP_OK) {
    ESP_LOGW(kTag, "Could not start Wi-Fi station connection");
    return false;
  }
  const EventBits_t bits = xEventGroupWaitBits(g_wifi_events,
      kConnectedBit | kFailedBit, pdTRUE, pdFALSE, pdMS_TO_TICKS(kWifiTimeoutMs));
  if ((bits & kConnectedBit) == 0) {
    ESP_LOGW(kTag, "Internet Wi-Fi connection timed out or failed");
    esp_wifi_disconnect();
    return false;
  }
  ESP_LOGI(kTag, "Internet Wi-Fi connected");
  return true;
}

bool FetchLatestVersion(char* out, size_t out_size) {
  esp_http_client_config_t config{};
  config.url = kLatestVersionUrl;
  config.timeout_ms = kHttpTimeoutMs;
  config.crt_bundle_attach = esp_crt_bundle_attach;
  esp_http_client_handle_t client = esp_http_client_init(&config);
  if (client == nullptr) return false;
  esp_err_t err = esp_http_client_open(client, 0);
  const int length = err == ESP_OK ? esp_http_client_fetch_headers(client) : -1;
  const int status = esp_http_client_get_status_code(client);
  if (err != ESP_OK || status != 200 || length <= 0 ||
      length >= static_cast<int>(out_size)) {
    ESP_LOGW(kTag, "Version request failed (err=%s HTTP=%d length=%d)",
             esp_err_to_name(err), status, length);
    esp_http_client_cleanup(client);
    return false;
  }
  int received = 0;
  while (received < length) {
    const int count = esp_http_client_read(client, out + received,
        static_cast<int>(out_size - 1) - received);
    if (count <= 0) break;
    received += count;
  }
  esp_http_client_cleanup(client);
  if (received <= 0 || received >= static_cast<int>(out_size)) return false;
  out[received] = '\0';
  while (received > 0 && std::isspace(static_cast<unsigned char>(out[received - 1])))
    out[--received] = '\0';
  char* begin = out;
  while (std::isspace(static_cast<unsigned char>(*begin))) ++begin;
  if (begin != out) std::memmove(out, begin, std::strlen(begin) + 1);
  return out[0] != '\0';
}

bool SynchronizeClock() {
  // TLS certificate validity checks need a plausible wall clock after power-on.
  esp_sntp_setoperatingmode(SNTP_OPMODE_POLL);
  esp_sntp_setservername(0, "pool.ntp.org");
  esp_sntp_init();
  const time_t minimum_time = 1704067200;  // 2024-01-01 UTC
  const int attempts = kSntpTimeoutMs / 250;
  for (int i = 0; i < attempts; ++i) {
    if (std::time(nullptr) >= minimum_time) {
      esp_sntp_stop();
      ESP_LOGI(kTag, "System clock synchronized for HTTPS validation");
      return true;
    }
    vTaskDelay(pdMS_TO_TICKS(250));
  }
  esp_sntp_stop();
  ESP_LOGW(kTag, "Could not synchronize clock; skipping HTTPS update check");
  return false;
}

}  // namespace

void CheckAndUpdate() {
  ESP_LOGI(kTag, "Installed firmware version: %s", kFirmwareVersion);
  if (!ConnectWifi()) return;
  if (!SynchronizeClock()) return;
  char latest_text[32]{};
  Version current, latest;
  if (!FetchLatestVersion(latest_text, sizeof(latest_text)) ||
      !ParseVersion(latest_text, &latest)) {
    ESP_LOGW(kTag, "Could not read a valid latest.txt version; continuing offline");
    return;
  }
  if (!ParseVersion(kFirmwareVersion, &current)) {
    ESP_LOGE(kTag, "Internal firmware version is malformed");
    return;
  }
  const int comparison = Compare(latest, current);
  ESP_LOGI(kTag, "Latest version: %s", latest_text);
  if (comparison <= 0) {
    ESP_LOGI(kTag, "No newer firmware available");
    return;
  }
  ESP_LOGI(kTag, "New firmware available; downloading over verified HTTPS");
  esp_http_client_config_t http_config{};
  http_config.url = kFirmwareUrl;
  http_config.timeout_ms = kHttpTimeoutMs;
  http_config.crt_bundle_attach = esp_crt_bundle_attach;
  esp_https_ota_config_t ota_config{};
  ota_config.http_config = &http_config;
  const esp_err_t err = esp_https_ota(&ota_config);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "HTTPS OTA failed: %s; continuing with current firmware",
             esp_err_to_name(err));
    return;
  }
  ESP_LOGI(kTag, "HTTPS OTA image validated; rebooting into version %s", latest_text);
  esp_restart();
}

}  // namespace fall_internet_ota
