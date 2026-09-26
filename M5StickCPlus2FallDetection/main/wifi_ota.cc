#include "wifi_ota.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

namespace fall_ota {
namespace {

constexpr char kTag[] = "fall_ota";
constexpr char kSsid[] = "FallDetector-OTA";
constexpr char kPassword[] = "fallupdate";
constexpr char kIpAddress[] = "192.168.4.1";
constexpr size_t kReceiveBufferSize = 4096;

volatile bool g_update_in_progress = false;
bool g_started = false;
httpd_handle_t g_server = nullptr;

const char kIndexHtml[] = R"HTML(
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fall Detector OTA</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;max-width:560px;margin:40px auto;padding:0 18px;color:#111}
.card{border:1px solid #ccc;border-radius:14px;padding:22px}
h1{font-size:1.45rem;margin-top:0}
input,button{font:inherit;margin-top:14px}
button{padding:10px 16px;border:0;border-radius:8px;background:#111;color:white}
button:disabled{opacity:.45}
#status{margin-top:16px;white-space:pre-wrap}
.small{color:#555;font-size:.92rem}
</style>
</head>
<body>
<div class="card">
<h1>Fall Detector Software Update</h1>
<p>Select the <b>firmware.bin</b> produced by PlatformIO. The firmware and embedded TFLite model are updated together.</p>
<input id="file" type="file" accept=".bin,application/octet-stream">
<br><button id="upload">Upload and reboot</button>
<div id="status" class="small">Do not power off the device during the upload.</div>
</div>
<script>
const file=document.getElementById('file');
const button=document.getElementById('upload');
const status=document.getElementById('status');
button.addEventListener('click',async()=>{
  if(!file.files.length){status.textContent='Choose firmware.bin first.';return;}
  const f=file.files[0];
  button.disabled=true;
  file.disabled=true;
  status.textContent='Uploading '+f.name+' ('+f.size+' bytes)...';
  try{
    const r=await fetch('/update',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:f});
    const text=await r.text();
    if(!r.ok) throw new Error(text||('HTTP '+r.status));
    status.textContent=text+'\nThe device is rebooting. Reconnect to FallDetector-OTA after it restarts.';
  }catch(e){
    status.textContent='Update failed: '+e.message;
    button.disabled=false;
    file.disabled=false;
  }
});
</script>
</body>
</html>
)HTML";

esp_err_t SendError(httpd_req_t* req, const char* status, const char* message) {
  httpd_resp_set_status(req, status);
  httpd_resp_set_type(req, "text/plain");
  httpd_resp_sendstr(req, message);
  return ESP_FAIL;
}

esp_err_t RootHandler(httpd_req_t* req) {
  httpd_resp_set_type(req, "text/html");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, kIndexHtml, HTTPD_RESP_USE_STRLEN);
}

esp_err_t UpdateHandler(httpd_req_t* req) {
  if (g_update_in_progress) {
    return SendError(req, "409 Conflict", "Another OTA update is already running.");
  }

  const esp_partition_t* update_partition = esp_ota_get_next_update_partition(nullptr);
  if (update_partition == nullptr) {
    return SendError(req, "500 Internal Server Error", "No inactive OTA partition is available.");
  }

  const int content_length = req->content_len;
  if (content_length <= 0) {
    return SendError(req, "400 Bad Request", "Empty firmware image.");
  }
  if (static_cast<size_t>(content_length) > update_partition->size) {
    return SendError(req, "400 Bad Request", "Firmware image is larger than the OTA slot.");
  }

  ESP_LOGI(kTag, "OTA upload: %d bytes -> %s @ 0x%lx (%lu-byte slot)",
           content_length, update_partition->label,
           static_cast<unsigned long>(update_partition->address),
           static_cast<unsigned long>(update_partition->size));

  g_update_in_progress = true;
  esp_ota_handle_t ota_handle = 0;
  bool ota_started = false;

  esp_err_t err = esp_ota_begin(update_partition,
                                static_cast<size_t>(content_length),
                                &ota_handle);
  if (err != ESP_OK) {
    g_update_in_progress = false;
    ESP_LOGE(kTag, "esp_ota_begin failed: %s", esp_err_to_name(err));
    return SendError(req, "500 Internal Server Error", "Could not start OTA write.");
  }
  ota_started = true;

  uint8_t buffer[kReceiveBufferSize];
  int remaining = content_length;
  int received_total = 0;
  bool first_chunk = true;

  while (remaining > 0) {
    const int want = std::min<int>(remaining, sizeof(buffer));
    const int received = httpd_req_recv(req, reinterpret_cast<char*>(buffer), want);
    if (received == HTTPD_SOCK_ERR_TIMEOUT) {
      continue;
    }
    if (received <= 0) {
      ESP_LOGE(kTag, "OTA receive failed after %d/%d bytes", received_total,
               content_length);
      esp_ota_abort(ota_handle);
      g_update_in_progress = false;
      return SendError(req, "500 Internal Server Error", "Firmware upload was interrupted.");
    }

    // Every normal ESP32 application image begins with the image magic byte 0xE9.
    // This catches common mistakes such as selecting a ZIP or source file.
    if (first_chunk) {
      first_chunk = false;
      if (buffer[0] != 0xE9) {
        ESP_LOGE(kTag, "Uploaded file is not an ESP32 application image (magic=0x%02X)",
                 buffer[0]);
        esp_ota_abort(ota_handle);
        g_update_in_progress = false;
        return SendError(req, "400 Bad Request", "Not a valid ESP32 firmware.bin image.");
      }
    }

    err = esp_ota_write(ota_handle, buffer, static_cast<size_t>(received));
    if (err != ESP_OK) {
      ESP_LOGE(kTag, "esp_ota_write failed: %s", esp_err_to_name(err));
      esp_ota_abort(ota_handle);
      g_update_in_progress = false;
      return SendError(req, "500 Internal Server Error", "Flash write failed.");
    }

    remaining -= received;
    received_total += received;
  }

  if (!ota_started) {
    g_update_in_progress = false;
    return SendError(req, "500 Internal Server Error", "OTA did not start.");
  }

  err = esp_ota_end(ota_handle);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "esp_ota_end/image validation failed: %s", esp_err_to_name(err));
    g_update_in_progress = false;
    return SendError(req, "400 Bad Request", "Firmware image validation failed.");
  }

  err = esp_ota_set_boot_partition(update_partition);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "esp_ota_set_boot_partition failed: %s", esp_err_to_name(err));
    g_update_in_progress = false;
    return SendError(req, "500 Internal Server Error", "Could not activate new firmware.");
  }

  ESP_LOGI(kTag, "OTA image verified. Next boot partition: %s", update_partition->label);
  httpd_resp_set_type(req, "text/plain");
  httpd_resp_sendstr(req, "Update successful.");

  // Give the TCP response time to leave the device before restarting.
  vTaskDelay(pdMS_TO_TICKS(1000));
  esp_restart();
  return ESP_OK;
}

bool StartHttpServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.stack_size = 8192;
  config.max_uri_handlers = 4;
  config.lru_purge_enable = true;

  if (httpd_start(&g_server, &config) != ESP_OK) {
    ESP_LOGE(kTag, "Failed to start OTA HTTP server");
    g_server = nullptr;
    return false;
  }

  httpd_uri_t root{};
  root.uri = "/";
  root.method = HTTP_GET;
  root.handler = RootHandler;
  root.user_ctx = nullptr;

  httpd_uri_t update{};
  update.uri = "/update";
  update.method = HTTP_POST;
  update.handler = UpdateHandler;
  update.user_ctx = nullptr;

  if (httpd_register_uri_handler(g_server, &root) != ESP_OK ||
      httpd_register_uri_handler(g_server, &update) != ESP_OK) {
    ESP_LOGE(kTag, "Failed to register OTA HTTP handlers");
    httpd_stop(g_server);
    g_server = nullptr;
    return false;
  }
  return true;
}

bool IsPendingVerify() {
  const esp_partition_t* running = esp_ota_get_running_partition();
  if (running == nullptr) return false;

  esp_ota_img_states_t state = ESP_OTA_IMG_UNDEFINED;
  const esp_err_t err = esp_ota_get_state_partition(running, &state);
  if (err != ESP_OK) return false;
  return state == ESP_OTA_IMG_PENDING_VERIFY;
}

}  // namespace

bool Start() {
  if (g_started) return true;

  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_LOGW(kTag, "NVS needs reinitialization");
    if (nvs_flash_erase() != ESP_OK) return false;
    err = nvs_flash_init();
  }
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "nvs_flash_init failed: %s", esp_err_to_name(err));
    return false;
  }

  err = esp_netif_init();
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
    ESP_LOGE(kTag, "esp_netif_init failed: %s", esp_err_to_name(err));
    return false;
  }

  err = esp_event_loop_create_default();
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
    ESP_LOGE(kTag, "event loop init failed: %s", esp_err_to_name(err));
    return false;
  }

  if (esp_netif_create_default_wifi_ap() == nullptr) {
    ESP_LOGE(kTag, "Could not create Wi-Fi AP network interface");
    return false;
  }

  wifi_init_config_t wifi_init = WIFI_INIT_CONFIG_DEFAULT();
  err = esp_wifi_init(&wifi_init);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "esp_wifi_init failed: %s", esp_err_to_name(err));
    return false;
  }
  esp_wifi_set_storage(WIFI_STORAGE_RAM);

  wifi_config_t wifi_config{};
  std::memcpy(wifi_config.ap.ssid, kSsid, sizeof(kSsid) - 1);
  wifi_config.ap.ssid_len = sizeof(kSsid) - 1;
  std::memcpy(wifi_config.ap.password, kPassword, sizeof(kPassword) - 1);
  wifi_config.ap.channel = 6;
  wifi_config.ap.max_connection = 2;
  wifi_config.ap.authmode = WIFI_AUTH_WPA2_PSK;

  if (esp_wifi_set_mode(WIFI_MODE_AP) != ESP_OK ||
      esp_wifi_set_config(WIFI_IF_AP, &wifi_config) != ESP_OK ||
      esp_wifi_start() != ESP_OK) {
    ESP_LOGE(kTag, "Failed to start Wi-Fi OTA access point");
    return false;
  }

  if (!StartHttpServer()) {
    esp_wifi_stop();
    return false;
  }

  g_started = true;
  ESP_LOGI(kTag, "Wireless software update ready");
  ESP_LOGI(kTag, "Wi-Fi SSID: %s", kSsid);
  ESP_LOGI(kTag, "Wi-Fi password: %s", kPassword);
  ESP_LOGI(kTag, "Update page: http://%s/", kIpAddress);
  return true;
}

bool InProgress() {
  return g_update_in_progress;
}

bool MarkRunningImageValid() {
  if (!IsPendingVerify()) return true;

  const esp_err_t err = esp_ota_mark_app_valid_cancel_rollback();
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "Could not mark OTA image valid: %s", esp_err_to_name(err));
    return false;
  }
  ESP_LOGI(kTag, "OTA image passed startup checks and is now marked valid");
  return true;
}

bool RollbackPendingImageAndReboot() {
  if (!IsPendingVerify()) return false;

  ESP_LOGE(kTag, "Pending OTA image failed startup checks; rolling back");
  const esp_err_t err = esp_ota_mark_app_invalid_rollback_and_reboot();
  ESP_LOGE(kTag, "Rollback request failed: %s", esp_err_to_name(err));
  return false;
}

}  // namespace fall_ota
