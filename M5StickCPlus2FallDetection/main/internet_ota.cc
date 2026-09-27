#include "internet_ota.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "cJSON.h"
#include "esp_crt_bundle.h"
#include "esp_event.h"
#include "esp_http_client.h"
#include "esp_https_ota.h"
#include "esp_log.h"
#include "esp_sntp.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "mbedtls/md.h"
#include "version.h"

#if FALL_ENCRYPTED_INTERNET_OTA
#include "esp_encrypted_img.h"
#endif

#if __has_include("wifi_secrets.h")
#include "wifi_secrets.h"
#else
#error "Missing main/wifi_secrets.h. Copy main/wifi_secrets.example.h to main/wifi_secrets.h and set your Wi-Fi credentials."
#endif

namespace fall_internet_ota {
namespace {

constexpr char kTag[] = "internet_ota";
constexpr char kManifestUrl[] =
    "https://raw.githubusercontent.com/xw18958/fall_detection2/main/M5StickCPlus2FallDetection/ota/stable.json";
constexpr char kReleaseUrlFormat[] =
    "https://github.com/xw18958/fall_detection2/releases/download/v%s/firmware.enc";
constexpr int kWifiTimeoutMs = 12000;
constexpr int kHttpTimeoutMs = 15000;
constexpr int kSntpTimeoutMs = 8000;
constexpr size_t kManifestBufferSize = 768;
constexpr EventBits_t kConnectedBit = BIT0;
constexpr EventBits_t kFailedBit = BIT1;

EventGroupHandle_t g_wifi_events = nullptr;

struct Version {
  uint32_t part[3]{};
};

struct UpdateManifest {
  bool enabled = false;
  char version_text[32]{};
  Version version{};
  uint8_t sha256[32]{};
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
          std::isdigit(static_cast<unsigned char>(p[1]))) {
        return false;
      }
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

int HexNibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

bool ParseSha256(const char* text, uint8_t out[32]) {
  if (text == nullptr || std::strlen(text) != 64) return false;
  for (int i = 0; i < 32; ++i) {
    const int hi = HexNibble(text[i * 2]);
    const int lo = HexNibble(text[i * 2 + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = static_cast<uint8_t>((hi << 4) | lo);
  }
  return true;
}

void WifiEventHandler(void*, esp_event_base_t base, int32_t id, void* data) {
  if (g_wifi_events == nullptr) return;
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

  esp_event_handler_instance_t wifi_handler{};
  esp_event_handler_instance_t ip_handler{};
  const bool wifi_registered =
      esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                          WifiEventHandler, nullptr,
                                          &wifi_handler) == ESP_OK;
  const bool ip_registered =
      esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                          WifiEventHandler, nullptr,
                                          &ip_handler) == ESP_OK;
  if (!wifi_registered || !ip_registered) {
    ESP_LOGE(kTag, "Could not register Wi-Fi event handlers");
    if (wifi_registered) {
      esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                            wifi_handler);
    }
    if (ip_registered) {
      esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                            ip_handler);
    }
    vEventGroupDelete(g_wifi_events);
    g_wifi_events = nullptr;
    return false;
  }

  wifi_config_t config{};
  std::strncpy(reinterpret_cast<char*>(config.sta.ssid), FALL_WIFI_SSID,
               sizeof(config.sta.ssid) - 1);
  std::strncpy(reinterpret_cast<char*>(config.sta.password), FALL_WIFI_PASSWORD,
               sizeof(config.sta.password) - 1);
  config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

  bool connected = false;
  if (esp_wifi_set_config(WIFI_IF_STA, &config) == ESP_OK &&
      esp_wifi_connect() == ESP_OK) {
    const EventBits_t bits = xEventGroupWaitBits(
        g_wifi_events, kConnectedBit | kFailedBit, pdTRUE, pdFALSE,
        pdMS_TO_TICKS(kWifiTimeoutMs));
    connected = (bits & kConnectedBit) != 0;
  }

  esp_event_handler_instance_unregister(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                        wifi_handler);
  esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                        ip_handler);
  vEventGroupDelete(g_wifi_events);
  g_wifi_events = nullptr;

  if (!connected) {
    ESP_LOGW(kTag, "Internet Wi-Fi connection timed out or failed");
    esp_wifi_disconnect();
    return false;
  }
  ESP_LOGI(kTag, "Internet Wi-Fi connected");
  return true;
}

bool SynchronizeClock() {
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

bool HttpGetSmallText(const char* url, char* out, size_t out_size) {
  if (url == nullptr || out == nullptr || out_size < 2) return false;

  esp_http_client_config_t config{};
  config.url = url;
  config.timeout_ms = kHttpTimeoutMs;
  config.crt_bundle_attach = esp_crt_bundle_attach;
  config.keep_alive_enable = true;

  esp_http_client_handle_t client = esp_http_client_init(&config);
  if (client == nullptr) return false;

  esp_err_t err = esp_http_client_open(client, 0);
  if (err == ESP_OK) {
    esp_http_client_fetch_headers(client);
  }
  const int status = esp_http_client_get_status_code(client);
  if (err != ESP_OK || status != 200) {
    ESP_LOGW(kTag, "Manifest request failed (err=%s HTTP=%d)",
             esp_err_to_name(err), status);
    esp_http_client_cleanup(client);
    return false;
  }

  size_t received = 0;
  while (received + 1 < out_size) {
    const int count = esp_http_client_read(
        client, out + received,
        static_cast<int>(out_size - received - 1));
    if (count < 0) {
      esp_http_client_cleanup(client);
      return false;
    }
    if (count == 0) break;
    received += static_cast<size_t>(count);
  }

  // Reject oversized metadata instead of parsing a truncated document.
  if (received + 1 == out_size) {
    char extra = 0;
    if (esp_http_client_read(client, &extra, 1) > 0) {
      ESP_LOGW(kTag, "OTA manifest is larger than %u bytes",
               static_cast<unsigned>(out_size - 1));
      esp_http_client_cleanup(client);
      return false;
    }
  }

  out[received] = '\0';
  esp_http_client_cleanup(client);
  return received > 0;
}

bool ParseManifest(const char* json, UpdateManifest* manifest) {
  if (json == nullptr || manifest == nullptr) return false;
  cJSON* root = cJSON_Parse(json);
  if (root == nullptr) return false;

  const cJSON* enabled = cJSON_GetObjectItemCaseSensitive(root, "enabled");
  const cJSON* version = cJSON_GetObjectItemCaseSensitive(root, "version");
  const cJSON* sha256 = cJSON_GetObjectItemCaseSensitive(root, "sha256");

  bool ok = cJSON_IsBool(enabled) && cJSON_IsString(version) &&
            version->valuestring != nullptr;
  if (ok) {
    manifest->enabled = cJSON_IsTrue(enabled);
    const size_t version_len = std::strlen(version->valuestring);
    ok = version_len > 0 && version_len < sizeof(manifest->version_text) &&
         ParseVersion(version->valuestring, &manifest->version);
    if (ok) {
      std::memcpy(manifest->version_text, version->valuestring, version_len + 1);
    }
  }

  if (ok && manifest->enabled) {
    ok = cJSON_IsString(sha256) && sha256->valuestring != nullptr &&
         ParseSha256(sha256->valuestring, manifest->sha256);
  }

  cJSON_Delete(root);
  return ok;
}

bool FetchManifest(UpdateManifest* manifest) {
  char buffer[kManifestBufferSize]{};
  if (!HttpGetSmallText(kManifestUrl, buffer, sizeof(buffer))) return false;
  if (!ParseManifest(buffer, manifest)) {
    ESP_LOGW(kTag, "OTA manifest is malformed; continuing with installed firmware");
    return false;
  }
  return true;
}

#if FALL_ENCRYPTED_INTERNET_OTA

extern const char ota_private_pem_start[] asm("_binary_ota_private_pem_start");
extern const char ota_private_pem_end[] asm("_binary_ota_private_pem_end");

struct DecryptContext {
  esp_decrypt_handle_t decrypt_handle = nullptr;
  mbedtls_md_context_t sha_ctx{};
  bool sha_ready = false;
};

esp_err_t DecryptCallback(decrypt_cb_arg_t* args, void* user_ctx) {
  if (args == nullptr || user_ctx == nullptr) return ESP_ERR_INVALID_ARG;
  auto* ctx = static_cast<DecryptContext*>(user_ctx);
  if (ctx->decrypt_handle == nullptr || !ctx->sha_ready) return ESP_FAIL;

  if (mbedtls_md_update(&ctx->sha_ctx,
                        reinterpret_cast<const unsigned char*>(args->data_in),
                        args->data_in_len) != 0) {
    ESP_LOGE(kTag, "SHA-256 update failed during OTA download");
    return ESP_FAIL;
  }

  pre_enc_decrypt_arg_t decrypt_args{};
  decrypt_args.data_in = args->data_in;
  decrypt_args.data_in_len = args->data_in_len;
  const esp_err_t err =
      esp_encrypted_img_decrypt_data(ctx->decrypt_handle, &decrypt_args);
  if (err != ESP_OK && err != ESP_ERR_NOT_FINISHED) {
    ESP_LOGE(kTag, "Encrypted-image decrypt failed: %s", esp_err_to_name(err));
    std::free(decrypt_args.data_out);
    return err;
  }

  args->data_out = decrypt_args.data_out;
  args->data_out_len = decrypt_args.data_out_len;
  return ESP_OK;
}

bool RunEncryptedOta(const UpdateManifest& manifest) {
  char firmware_url[256]{};
  const int written = std::snprintf(firmware_url, sizeof(firmware_url),
                                    kReleaseUrlFormat, manifest.version_text);
  if (written <= 0 || written >= static_cast<int>(sizeof(firmware_url))) {
    ESP_LOGE(kTag, "Versioned release URL is too long");
    return false;
  }

  esp_decrypt_cfg_t decrypt_cfg{};
  decrypt_cfg.rsa_priv_key = ota_private_pem_start;
  decrypt_cfg.rsa_priv_key_len =
      static_cast<size_t>(ota_private_pem_end - ota_private_pem_start);

  DecryptContext ctx{};
  ctx.decrypt_handle = esp_encrypted_img_decrypt_start(&decrypt_cfg);
  if (ctx.decrypt_handle == nullptr) {
    ESP_LOGE(kTag, "Could not initialize encrypted OTA decryption");
    return false;
  }

  mbedtls_md_init(&ctx.sha_ctx);
  const mbedtls_md_info_t* sha_info = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  if (sha_info == nullptr || mbedtls_md_setup(&ctx.sha_ctx, sha_info, 0) != 0 ||
      mbedtls_md_starts(&ctx.sha_ctx) != 0) {
    ESP_LOGE(kTag, "Could not initialize OTA SHA-256 verification");
    esp_encrypted_img_decrypt_abort(ctx.decrypt_handle);
    mbedtls_md_free(&ctx.sha_ctx);
    return false;
  }
  ctx.sha_ready = true;

  esp_http_client_config_t http_config{};
  http_config.url = firmware_url;
  http_config.timeout_ms = kHttpTimeoutMs;
  http_config.crt_bundle_attach = esp_crt_bundle_attach;
  http_config.keep_alive_enable = true;

  esp_https_ota_config_t ota_config{};
  ota_config.http_config = &http_config;
  ota_config.decrypt_cb = DecryptCallback;
  ota_config.decrypt_user_ctx = &ctx;
  ota_config.enc_img_header_size = esp_encrypted_img_get_header_size();

  esp_https_ota_handle_t ota_handle = nullptr;
  esp_err_t err = esp_https_ota_begin(&ota_config, &ota_handle);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "HTTPS OTA begin failed: %s", esp_err_to_name(err));
    esp_encrypted_img_decrypt_abort(ctx.decrypt_handle);
    mbedtls_md_free(&ctx.sha_ctx);
    return false;
  }

  ESP_LOGI(kTag, "Downloading encrypted release v%s", manifest.version_text);
  do {
    err = esp_https_ota_perform(ota_handle);
  } while (err == ESP_ERR_HTTPS_OTA_IN_PROGRESS);

  bool ok = err == ESP_OK && esp_https_ota_is_complete_data_received(ota_handle) &&
            esp_encrypted_img_is_complete_data_received(ctx.decrypt_handle);
  if (!ok) {
    ESP_LOGE(kTag, "Encrypted OTA download was incomplete: %s",
             esp_err_to_name(err));
  }

  if (ok) {
    err = esp_encrypted_img_decrypt_end(ctx.decrypt_handle);
    ctx.decrypt_handle = nullptr;
    if (err != ESP_OK) {
      ESP_LOGE(kTag, "Encrypted OTA authentication/decryption finalization failed: %s",
               esp_err_to_name(err));
      ok = false;
    }
  }

  uint8_t actual_sha[32]{};
  if (ok && mbedtls_md_finish(&ctx.sha_ctx, actual_sha) != 0) {
    ESP_LOGE(kTag, "Could not finalize OTA SHA-256");
    ok = false;
  }
  mbedtls_md_free(&ctx.sha_ctx);
  ctx.sha_ready = false;

  if (ok && std::memcmp(actual_sha, manifest.sha256, sizeof(actual_sha)) != 0) {
    ESP_LOGE(kTag, "Encrypted release SHA-256 does not match stable.json");
    ok = false;
  }

  if (!ok) {
    esp_https_ota_abort(ota_handle);
    if (ctx.decrypt_handle != nullptr) {
      esp_encrypted_img_decrypt_abort(ctx.decrypt_handle);
    }
    return false;
  }

  const esp_err_t finish_err = esp_https_ota_finish(ota_handle);
  if (finish_err != ESP_OK) {
    ESP_LOGE(kTag, "OTA image validation/activation failed: %s",
             esp_err_to_name(finish_err));
    return false;
  }

  ESP_LOGI(kTag, "Encrypted OTA v%s installed; rebooting", manifest.version_text);
  vTaskDelay(pdMS_TO_TICKS(500));
  esp_restart();
  return true;  // Unreachable after successful restart.
}

#endif  // FALL_ENCRYPTED_INTERNET_OTA

}  // namespace

void CheckAndUpdate() {
  ESP_LOGI(kTag, "Installed firmware version: %s", FALL_FIRMWARE_VERSION);

#if !FALL_ENCRYPTED_INTERNET_OTA
  ESP_LOGW(kTag,
           "Encrypted GitHub Internet OTA disabled in this build. Run "
           "tools/generate_ota_keys.sh before building to enable it.");
  return;
#else
  if (!ConnectWifi()) return;
  if (!SynchronizeClock()) return;

  UpdateManifest manifest{};
  if (!FetchManifest(&manifest)) return;
  if (!manifest.enabled) {
    ESP_LOGI(kTag, "GitHub OTA manifest is disabled");
    return;
  }

  Version current{};
  if (!ParseVersion(FALL_FIRMWARE_VERSION, &current)) {
    ESP_LOGE(kTag, "Compiled firmware version is malformed");
    return;
  }

  ESP_LOGI(kTag, "Published stable version: %s", manifest.version_text);
  if (Compare(manifest.version, current) <= 0) {
    ESP_LOGI(kTag, "No newer firmware available");
    return;
  }

  if (!RunEncryptedOta(manifest)) {
    ESP_LOGE(kTag, "GitHub OTA failed; continuing with installed firmware");
  }
#endif
}

}  // namespace fall_internet_ota
