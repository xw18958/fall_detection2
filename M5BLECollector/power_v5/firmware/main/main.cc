#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "driver/gpio.h"
#include "buzzer.h"
#include "power_config.h"
#include "power_management.h"
#include "power_policy.h"
#include "trigger_model.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "fall_op_resolver.h"
#include "simple_display.h"
#include "device_ui.h"
#include "battery_monitor.h"
#include "wifi_ota.h"
#include "ble_collector.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_system.h"
#include "model_v2_config.h"
#include "input_pipeline.h"
#include "model_identity.h"
#include "mbedtls/sha256.h"
#include "model_v2_replay.h"
#include "inference_profiler.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr char kTag[] = "fall_tflm";
InferenceProfiler g_profiler;

// M5StickC PLUS2 pins.
constexpr gpio_num_t kHoldPin = GPIO_NUM_4;
constexpr i2c_port_t kI2CPort = I2C_NUM_0;
constexpr gpio_num_t kI2CSda = GPIO_NUM_21;
constexpr gpio_num_t kI2CScl = GPIO_NUM_22;
constexpr uint8_t kMpuAddress = 0x68;

// MPU6886 register addresses.
constexpr uint8_t kRegSampleRateDiv = 0x19;
constexpr uint8_t kRegConfig = 0x1A;
constexpr uint8_t kRegGyroConfig = 0x1B;
constexpr uint8_t kRegAccelConfig = 0x1C;
constexpr uint8_t kRegAccelConfig2 = 0x1D;
constexpr uint8_t kRegFifoEn = 0x23;
constexpr uint8_t kRegIntPinCfg = 0x37;
constexpr uint8_t kRegIntEnable = 0x38;
constexpr uint8_t kRegAccelXoutH = 0x3B;
constexpr uint8_t kRegUserCtrl = 0x6A;
constexpr uint8_t kRegPwrMgmt1 = 0x6B;
constexpr uint8_t kRegWhoAmI = 0x75;

constexpr int kTimesteps = fall_v2::kTimesteps;
constexpr int kChannels = fall_v2::kChannels;
constexpr int64_t kSamplePeriodUs = 1000000 / fall_v2::kSampleHz;

// Checkpoint normalization from the uploaded conversion package.
constexpr float kMean[kChannels] = {
    fall_v2::kMean[0], fall_v2::kMean[1], fall_v2::kMean[2],
    fall_v2::kMean[3], fall_v2::kMean[4], fall_v2::kMean[5],
};
constexpr float kSigma[kChannels] = {
    fall_v2::kSigma[0], fall_v2::kSigma[1], fall_v2::kSigma[2],
    fall_v2::kSigma[3], fall_v2::kSigma[4], fall_v2::kSigma[5],
};

// Selected using private validation after quantization, before test evaluation.
constexpr float kPrototypeFallThreshold = fall_v2::kThreshold;
constexpr int64_t kFallAlertCooldownUs = 3000000;
constexpr int kFallBeepMs = 100;

// MPU6886 configured to ±8 g and ±2000 deg/s.
constexpr float kAccelGPerLsb = 8.0f / 32768.0f;
constexpr float kGyroDpsPerLsb = 2000.0f / 32768.0f;

// Embedded by PlatformIO's board_build.embed_files mechanism.
extern const uint8_t model_tflite_start[] asm("_binary_model_tflite_start");
extern const uint8_t model_tflite_end[] asm("_binary_model_tflite_end");

float g_ring[kTimesteps][kChannels] = {};
int g_ring_pos = 0;
int g_ring_count = 0;
uint64_t g_sample_count = 0;
uint32_t g_window_generation=0;
portMUX_TYPE g_ring_lock = portMUX_INITIALIZER_UNLOCKED;
float g_window[kTimesteps][kChannels] = {};

uint8_t* g_tensor_arena = nullptr;
size_t g_tensor_arena_size = 0;
uint8_t* g_persistent_arena=nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;
fall_tflm::FallOpResolver g_resolver;

bool g_collection_mode = false;
struct PowerStats {
 uint64_t samples=0,gaps=0,sensor_us=0,full_n=0,full_us=0,full_path_us=0,display_us=0,wifi_us=0;
 uint64_t trigger_n=0,trigger_us=0,wake_n=0,burst_caps=0,active_us=0;
 uint64_t trigger_deadline_misses=0,full_deadline_misses=0,queue_drops=0;
 uint64_t shadow_wake_n=0,shadow_caps=0,shadow_active_us=0;
};
PowerStats g_power; portMUX_TYPE g_power_lock=portMUX_INITIALIZER_UNLOCKED;
int64_t g_alert_until=0;
struct ResultState {float probability=0;bool fall=false,valid=false;int milliseconds=0;int64_t cooldown_us=0;};
ResultState g_result;
std::atomic<bool> g_cascade{false};
QueueHandle_t g_full_queue=nullptr;
TaskHandle_t g_alert_task=nullptr;
uint64_t g_boot_full_worst_us=0;


struct RawImu {
  int16_t counts[6]{};
  float ax_g;
  float ay_g;
  float az_g;
  float gx_dps;
  float gy_dps;
  float gz_dps;
};

esp_err_t WriteRegister(uint8_t reg, uint8_t value) {
  uint8_t data[2] = {reg, value};
  return i2c_master_write_to_device(kI2CPort, kMpuAddress, data, sizeof(data),
                                    pdMS_TO_TICKS(100));
}

esp_err_t ReadRegisters(uint8_t reg, uint8_t* data, size_t len) {
  return i2c_master_write_read_device(kI2CPort, kMpuAddress, &reg, 1, data, len,
                                      pdMS_TO_TICKS(100));
}

bool InitPowerHold() {
  gpio_config_t cfg{};
  cfg.pin_bit_mask = 1ULL << kHoldPin;
  cfg.mode = GPIO_MODE_OUTPUT;
  cfg.pull_up_en = GPIO_PULLUP_DISABLE;
  cfg.pull_down_en = GPIO_PULLDOWN_DISABLE;
  cfg.intr_type = GPIO_INTR_DISABLE;
  if (gpio_config(&cfg) != ESP_OK) return false;
  return gpio_set_level(kHoldPin, 1) == ESP_OK;
}

bool InitBuzzer() {
  const esp_err_t err = fall_buzzer::Init();
  if (err != ESP_OK) ESP_LOGE(kTag, "Buzzer initialization failed: %s", esp_err_to_name(err));
  else ESP_LOGI(kTag, "Buzzer ready: 2 kHz; output low and timer paused when idle");
  return err == ESP_OK;
}

void Beep(int milliseconds) {
  const esp_err_t err = fall_buzzer::Beep(milliseconds);
  if (err != ESP_OK) ESP_LOGE(kTag, "Buzzer beep failed: %s", esp_err_to_name(err));
}

bool InitI2C() {
  i2c_config_t config{};
  config.mode = I2C_MODE_MASTER;
  config.sda_io_num = kI2CSda;
  config.scl_io_num = kI2CScl;
  config.sda_pullup_en = GPIO_PULLUP_ENABLE;
  config.scl_pullup_en = GPIO_PULLUP_ENABLE;
  config.master.clk_speed = 400000;
  config.clk_flags = 0;
  if (i2c_param_config(kI2CPort, &config) != ESP_OK) return false;
  return i2c_driver_install(kI2CPort, config.mode, 0, 0, 0) == ESP_OK;
}

bool InitMpu6886() {
  uint8_t who = 0;
  if (ReadRegisters(kRegWhoAmI, &who, 1) != ESP_OK) {
    ESP_LOGE(kTag, "MPU6886 WHO_AM_I read failed");
    return false;
  }
  ESP_LOGI(kTag, "MPU6886 WHO_AM_I = 0x%02X", who);

  // Reset then select a stable clock source.
  ESP_ERROR_CHECK(WriteRegister(kRegPwrMgmt1, 0x80));
  vTaskDelay(pdMS_TO_TICKS(100));
  ESP_ERROR_CHECK(WriteRegister(kRegPwrMgmt1, 0x01));
  vTaskDelay(pdMS_TO_TICKS(10));

  // ±8 g accelerometer, ±2000 deg/s gyroscope.
  ESP_ERROR_CHECK(WriteRegister(kRegAccelConfig, 0x10));
  ESP_ERROR_CHECK(WriteRegister(kRegGyroConfig, 0x18));
  ESP_ERROR_CHECK(WriteRegister(kRegConfig, 0x01));
  ESP_ERROR_CHECK(WriteRegister(kRegSampleRateDiv, 0x01));
  ESP_ERROR_CHECK(WriteRegister(kRegAccelConfig2, 0x00));
  ESP_ERROR_CHECK(WriteRegister(kRegIntEnable, 0x00));
  ESP_ERROR_CHECK(WriteRegister(kRegUserCtrl, 0x00));
  ESP_ERROR_CHECK(WriteRegister(kRegFifoEn, 0x00));
  ESP_ERROR_CHECK(WriteRegister(kRegIntPinCfg, 0x22));
  return true;
}

int16_t ReadBe16(const uint8_t* p) {
  return static_cast<int16_t>((static_cast<uint16_t>(p[0]) << 8) | p[1]);
}

bool ReadImuCounts(int16_t out[6]) {
  uint8_t data[14] = {};
  if (ReadRegisters(kRegAccelXoutH, data, sizeof(data)) != ESP_OK) return false;
  out[0] = ReadBe16(data + 0);
  out[1] = ReadBe16(data + 2);
  out[2] = ReadBe16(data + 4);
  out[3] = ReadBe16(data + 8);
  out[4] = ReadBe16(data + 10);
  out[5] = ReadBe16(data + 12);
  return true;
}

bool ReadImu(RawImu* out) {
  if (!ReadImuCounts(out->counts)) return false;
  out->ax_g = out->counts[0] * kAccelGPerLsb;
  out->ay_g = out->counts[1] * kAccelGPerLsb;
  out->az_g = out->counts[2] * kAccelGPerLsb;
  out->gx_dps = out->counts[3] * kGyroDpsPerLsb;
  out->gy_dps = out->counts[4] * kGyroDpsPerLsb;
  out->gz_dps = out->counts[5] * kGyroDpsPerLsb;
  return true;
}

const char* TypeName(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32: return "float32";
    case kTfLiteInt8: return "int8";
    case kTfLiteInt16: return "int16";
    default: return "other";
  }
}

void PrintTensorShape(const char* label, const TfLiteTensor* tensor) {
  char shape[96] = {};
  int used = 0;
  used += snprintf(shape + used, sizeof(shape) - used, "[");
  for (int i = 0; i < tensor->dims->size && used < static_cast<int>(sizeof(shape)); ++i) {
    used += snprintf(shape + used, sizeof(shape) - used, "%s%d", i ? "," : "",
                     tensor->dims->data[i]);
  }
  snprintf(shape + std::min<int>(used, sizeof(shape) - 1),
           sizeof(shape) - std::min<int>(used, sizeof(shape) - 1), "]");
  ESP_LOGI(kTag, "%s shape=%s type=%s scale=%g zp=%ld", label, shape,
           TypeName(tensor->type), tensor->params.scale,
           static_cast<long>(tensor->params.zero_point));
}

bool InitModel() {
  const size_t model_size = model_tflite_end - model_tflite_start;
  ESP_LOGI(kTag, "Embedded TFLite model: %u bytes",
           static_cast<unsigned>(model_size));

  unsigned char digest[32];
  char hex[65];
  if (mbedtls_sha256(model_tflite_start, model_size, digest, 0) != 0) return false;
  for (int i = 0; i < 32; ++i) std::snprintf(hex + 2*i, 3, "%02x", digest[i]);
  if (std::strcmp(hex, m5ble::kModelSha256) != 0) return false;
  ESP_LOGI(kTag, "Verified model=%s checkpoint=%s threshold=%.9f", hex,
           fall_v2::kCheckpointSha256, fall_v2::kThreshold);
  const tflite::Model* model = tflite::GetModel(model_tflite_start);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(kTag, "TFLite schema mismatch: model=%d runtime=%d",
             model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  const size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  ESP_LOGI(kTag, "PSRAM initialized=%d, largest free block=%u bytes",
           esp_psram_is_initialized(), static_cast<unsigned>(largest));
  if (!esp_psram_is_initialized() || largest < 384 * 1024) {
    ESP_LOGE(kTag, "Not enough PSRAM for the TensorFlow Lite Micro arena");
    return false;
  }

  constexpr size_t kInternalArena = 64 * 1024;
  constexpr size_t kPreferredArena = 256 * 1024;
  constexpr size_t kReserve = 96 * 1024;
  g_tensor_arena_size = std::min(kPreferredArena,
                                 largest > kReserve ? largest - kReserve : largest);
  g_tensor_arena = static_cast<uint8_t*>(
      CONFIG_FALL_INTERNAL_ARENA ? heap_caps_malloc(kInternalArena, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) : nullptr);
  bool internal_arena = g_tensor_arena != nullptr;
  ESP_LOGI(kTag, "Internal arena requested=%d, largest free internal=%u bytes", int(CONFIG_FALL_INTERNAL_ARENA), unsigned(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
  if (internal_arena) {
    g_tensor_arena_size = kInternalArena;
  } else {
    g_tensor_arena = static_cast<uint8_t*>(
        heap_caps_malloc(g_tensor_arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  }
  if (g_tensor_arena == nullptr) {
    ESP_LOGE(kTag, "Failed to allocate %u-byte tensor arena",
             static_cast<unsigned>(g_tensor_arena_size));
    return false;
  }
  ESP_LOGI(kTag, "Tensor arena: %u bytes in %s",
           static_cast<unsigned>(g_tensor_arena_size), internal_arena ? "internal SRAM" : "PSRAM");

  if(internal_arena) {
    constexpr size_t kPersistentArena=128*1024;
    g_persistent_arena=static_cast<uint8_t*>(heap_caps_malloc(kPersistentArena,MALLOC_CAP_SPIRAM|MALLOC_CAP_8BIT));
    if(!g_persistent_arena) return false;
    // Supported split allocator: all runtime activation buffers are internal.
    // Node/tensor metadata remains in PSRAM, leaving room for startup planning.
    auto* allocator=tflite::MicroAllocator::Create(g_persistent_arena,kPersistentArena,g_tensor_arena,g_tensor_arena_size);
    if(!allocator) return false;
    g_interpreter=new (std::nothrow) tflite::MicroInterpreter(model,g_resolver,allocator,nullptr,&g_profiler);
    ESP_LOGI(kTag,"Split allocator: %u-byte internal activation arena; %u-byte PSRAM metadata arena",unsigned(g_tensor_arena_size),unsigned(kPersistentArena));
  } else {
    g_interpreter = new (std::nothrow)
        tflite::MicroInterpreter(model, g_resolver, g_tensor_arena,
                                 g_tensor_arena_size, nullptr, &g_profiler);
  }
  if (g_interpreter == nullptr) {
    ESP_LOGE(kTag, "Failed to create MicroInterpreter");
    return false;
  }

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag,
             "AllocateTensors failed. Check the first 'op not found' message. "
             "If it is an arena error, increase available PSRAM.");
    return false;
  }

  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);
  if (g_input == nullptr || g_output == nullptr) return false;
  PrintTensorShape("input", g_input);
  PrintTensorShape("output", g_output);

  if (g_input->dims->size != 3 || g_input->dims->data[0] != 1 ||
      g_input->dims->data[1] != kTimesteps ||
      g_input->dims->data[2] != kChannels) {
    ESP_LOGE(kTag, "Unexpected model input shape; expected [1,90,6]");
    return false;
  }
  if (g_output->dims->size != 2 || g_output->dims->data[0] != 1 ||
      g_output->dims->data[1] != 2) {
    ESP_LOGE(kTag, "Unexpected model output shape; expected [1,2]");
    return false;
  }
  if (g_input->type != kTfLiteInt8 || g_output->type != kTfLiteInt8 ||
      g_input->params.scale != fall_v2::kInputScale || g_input->params.zero_point != fall_v2::kInputZero ||
      g_output->params.scale != fall_v2::kOutputScale || g_output->params.zero_point != fall_v2::kOutputZero) return false;
  return true;
}

inline float Normalize(int channel, float value) {
  return fall_v2::NormalizeM5(channel, value);
}

void PushSample(const RawImu& imu) {
  // Full MPU range -> SI units -> this checkpoint's M5 normalization.
  portENTER_CRITICAL(&g_ring_lock);
  for (int c = 0; c < kChannels; ++c) {
    g_ring[g_ring_pos][c] = Normalize(c, fall_v2::PhysicalValue(c, imu.counts[c]));
  }
  g_ring_pos = (g_ring_pos + 1) % kTimesteps;
  if (g_ring_count < kTimesteps) ++g_ring_count;
  ++g_sample_count;

  portEXIT_CRITICAL(&g_ring_lock);
}

bool FillModelInput() {
  if (g_input == nullptr) return false;
  portENTER_CRITICAL(&g_ring_lock);
  if (g_ring_count < kTimesteps) {
    portEXIT_CRITICAL(&g_ring_lock);
    return false;
  }
  for (int t = 0; t < kTimesteps; ++t) {
    const int src_t = (g_ring_pos + t) % kTimesteps;
    std::memcpy(g_window[t], g_ring[src_t], sizeof(g_window[t]));
  }
  portEXIT_CRITICAL(&g_ring_lock);

  // When full, g_ring_pos points at the oldest sample.
  if (g_input->type == kTfLiteFloat32) {
    float* dst = g_input->data.f;
    for (int t = 0; t < kTimesteps; ++t) {
      for (int c = 0; c < kChannels; ++c) {
        dst[t * kChannels + c] = g_window[t][c];
      }
    }
    return true;
  }

  if (g_input->type == kTfLiteInt8) {
    if (g_input->params.scale <= 0.0f) return false;
    int8_t* dst = g_input->data.int8;
    for (int t = 0; t < kTimesteps; ++t) {
      for (int c = 0; c < kChannels; ++c) {
        const float x = g_window[t][c];
        dst[t * kChannels + c] = fall_v2::QuantizeInput(
            x, g_input->params.scale, g_input->params.zero_point);
      }
    }
    return true;
  }

  ESP_LOGE(kTag, "Unsupported model input type: %d", g_input->type);
  return false;
}

bool ModelReplaySelfTest() {
  fall_power::FastCpu clock;
  if (g_input->type != kTfLiteInt8 || g_output->type != kTfLiteInt8) return false;
  for (int i = 0; i < fall_v2::kPreprocessCount; ++i) {
    for (int c = 0; c < kChannels; ++c) {
      const float normalized = Normalize(c, fall_v2::PhysicalValue(c, fall_v2::kPreprocessCounts[i][c]));
      if (fall_v2::QuantizeInput(normalized, g_input->params.scale, g_input->params.zero_point) !=
          fall_v2::kPreprocessExpected[i][c]) return false;
    }
  }
  for (int i = 0; i < fall_v2::kReplayCount; ++i) {
    std::memcpy(g_input->data.int8, fall_v2::kReplayInputs[i], 540);
    g_profiler.Reset();
    const int64_t replay_start=esp_timer_get_time();
    if (g_interpreter->Invoke() != kTfLiteOk) return false;
    g_boot_full_worst_us=std::max(g_boot_full_worst_us,uint64_t(esp_timer_get_time()-replay_start));
    for (int c = 0; c < 2; ++c) {
      const int error = std::abs(static_cast<int>(g_output->data.int8[c]) -
                                 static_cast<int>(fall_v2::kReplayOutputs[i][c]));
      if (error > 4) {
        ESP_LOGE(kTag, "Model replay %d output %d mismatch: %d", i, c, error);
        return false;
      }
    }
    vTaskDelay(1);
  }
  if(CONFIG_FALL_DEBUG_LOGS) g_profiler.Dump();
  ESP_LOGI(kTag, "V5 INT8 model replay passed; arena used=%u bytes; startup worst=%llu us",
           static_cast<unsigned>(g_interpreter->arena_used_bytes()),g_boot_full_worst_us);
  return true;
}

void ResetSampleWindow() {
  portENTER_CRITICAL(&g_ring_lock);
  g_ring_count = 0;
  g_ring_pos = 0;
  ++g_window_generation;
  portEXIT_CRITICAL(&g_ring_lock);
}

void TimerWake(void* task) { xTaskNotifyGive(static_cast<TaskHandle_t>(task)); }
void WaitDeadline(esp_timer_handle_t timer,int64_t deadline) {
 const int64_t wait=deadline-esp_timer_get_time();
 if(wait<=0) return;
 ESP_ERROR_CHECK(esp_timer_start_once(timer,wait));
 ulTaskNotifyTake(pdTRUE,portMAX_DELAY);
}
void SensorTask(void*) {
  esp_timer_handle_t timer=nullptr;esp_timer_create_args_t args{};
  args.callback=TimerWake;args.arg=xTaskGetCurrentTaskHandle();args.name="imu_30hz";
  ESP_ERROR_CHECK(esp_timer_create(&args,&timer));
  int64_t origin = esp_timer_get_time();
  uint64_t tick = 0;
  while (true) {
    if (fall_ota::InProgress()) {
      ResetSampleWindow();
      vTaskDelay(pdMS_TO_TICKS(20));
      origin = esp_timer_get_time();
      tick = 0;
      continue;
    }
    const int64_t target=fall_power::SampleDeadline(origin,tick);
    WaitDeadline(timer,target);
    uint16_t flags = 0;
    if (esp_timer_get_time() - target > kSamplePeriodUs) {
      flags |= m5ble::TimingGap;
      // A missing interval invalidates the rolling window; never compress time.
      ResetSampleWindow();
      origin = esp_timer_get_time();
      tick = 0;
      ESP_LOGW(kTag, "Sampling overrun; rebuilding the 3-second window");
    }
    const uint64_t acquisition_us = esp_timer_get_time();
    if (g_collection_mode) {
      int16_t counts[6] = {};
      if (ReadImuCounts(counts)) {
        for (int16_t value : counts)
          if (value == INT16_MIN || value == INT16_MAX) flags |= m5ble::Saturated;
      } else {
        flags |= m5ble::ReadError;
        ESP_LOGW(kTag, "IMU read failed during collection");
      }
      m5ble::Capture(acquisition_us, counts, flags);
    } else {
      RawImu imu{};
      if (ReadImu(&imu)) {
        PushSample(imu);
      } else {
        flags |= m5ble::ReadError;
        ResetSampleWindow();
        ESP_LOGW(kTag, "IMU read failed; rebuilding the window");
      }
    }
    const uint64_t sensor_elapsed_us=esp_timer_get_time()-acquisition_us;
    portENTER_CRITICAL(&g_power_lock);
    if(!(flags & m5ble::ReadError)) ++g_power.samples;
    if(flags & (m5ble::ReadError | m5ble::TimingGap)) ++g_power.gaps;
    g_power.sensor_us+=sensor_elapsed_us;
    portEXIT_CRITICAL(&g_power_lock);
    ++tick;
  }
}

bool ReadLogits(float logits[2]) {
  if (g_output->type == kTfLiteFloat32) {
    logits[0] = g_output->data.f[0];
    logits[1] = g_output->data.f[1];
    return true;
  }
  if (g_output->type == kTfLiteInt8) {
    for (int i = 0; i < 2; ++i) {
      logits[i] = (static_cast<int32_t>(g_output->data.int8[i]) -
                   g_output->params.zero_point) *
                  g_output->params.scale;
    }
    return true;
  }
  ESP_LOGE(kTag, "Unsupported output type: %d", g_output->type);
  return false;
}

void Softmax2(const float logits[2], float probs[2]) {
  const float m = std::max(logits[0], logits[1]);
  const float e0 = std::exp(logits[0] - m);
  const float e1 = std::exp(logits[1] - m);
  const float denom = e0 + e1;
  probs[0] = e0 / denom;
  probs[1] = e1 / denom;
}

bool CaptureWindow(fall_power::Window* window,int64_t deadline) {
  portENTER_CRITICAL(&g_ring_lock);
  if(g_ring_count<kTimesteps) { portEXIT_CRITICAL(&g_ring_lock);return false; }
  for(int t=0;t<kTimesteps;++t) std::memcpy(window->x[t],g_ring[(g_ring_pos+t)%kTimesteps],sizeof(window->x[t]));
  window->deadline_us=deadline;window->last_sample=g_sample_count;window->generation=g_window_generation;
  portEXIT_CRITICAL(&g_ring_lock);return true;
}
bool FillFrozenInput(const fall_power::Window& window) {
  if(!g_input || g_input->type!=kTfLiteInt8 || g_input->params.scale<=0) return false;
  for(int t=0;t<kTimesteps;++t) for(int c=0;c<kChannels;++c)
    g_input->data.int8[t*kChannels+c]=fall_v2::QuantizeInput(window.x[t][c],g_input->params.scale,g_input->params.zero_point);
  return true;
}

void AlertTask(void*) {
  while(true) { ulTaskNotifyTake(pdTRUE,portMAX_DELAY);Beep(kFallBeepMs); }
}

bool RunInference(const fall_power::Window* captured=nullptr) {
  fall_power::FastCpu clock;
  const int64_t path_begin_us = esp_timer_get_time();
  if (!(captured ? FillFrozenInput(*captured) : FillModelInput())) return false;
  static bool profile_once = true;
  if (profile_once) g_profiler.Reset();
  const int64_t begin_us = esp_timer_get_time();
  if (g_interpreter->Invoke() != kTfLiteOk) {
    ESP_LOGE(kTag, "TFLM Invoke() failed");
    return false;
  }
  const int64_t elapsed_us = esp_timer_get_time() - begin_us;
  if (profile_once) {
    if(CONFIG_FALL_DEBUG_LOGS) g_profiler.Dump();
    g_profiler.Disable();
    profile_once = false;
  }

  float logits[2] = {};
  if (!ReadLogits(logits)) return false;
  float probs[2] = {};
  Softmax2(logits, probs);

  const int64_t path_us = esp_timer_get_time() - path_begin_us;
  static int64_t latency[128], path_latency[128], worst_us = 0, worst_path_us = 0;
  static unsigned measured = 0;
  latency[measured % 128] = elapsed_us; path_latency[measured % 128] = path_us;
  worst_us = std::max(worst_us, elapsed_us); worst_path_us = std::max(worst_path_us, path_us);
  ++measured;
  if (measured % 128 == 0) {
    int64_t sorted[128], path_sorted[128];
    std::memcpy(sorted, latency, sizeof(sorted)); std::memcpy(path_sorted, path_latency, sizeof(path_sorted));
    std::sort(sorted, sorted + 128); std::sort(path_sorted, path_sorted + 128);
    ESP_LOGI(kTag, "latency n=%u rolling128 invoke median=%lld p95=%lld p99=%lld worst=%lld us | input-to-prob median=%lld p95=%lld p99=%lld worst=%lld us | arena=%u free_internal=%u free_psram=%u",
        measured, (sorted[63]+sorted[64])/2, sorted[121], sorted[126], worst_us,
        (path_sorted[63]+path_sorted[64])/2, path_sorted[121], path_sorted[126], worst_path_us,
        static_cast<unsigned>(g_interpreter->arena_used_bytes()),
        static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
        static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
  }
  portENTER_CRITICAL(&g_power_lock);
  ++g_power.full_n;g_power.full_us+=elapsed_us;g_power.full_path_us+=path_us;
  portEXIT_CRITICAL(&g_power_lock);
  const bool fall_triggered = probs[1] >= kPrototypeFallThreshold;
  if(fall_triggered) {portENTER_CRITICAL(&g_power_lock);g_alert_until=esp_timer_get_time()+15000000;portEXIT_CRITICAL(&g_power_lock);}
  static int64_t last_beep_us = -kFallAlertCooldownUs;
  int64_t now_us = esp_timer_get_time();
  bool beep_now = false;

  if (fall_triggered && now_us - last_beep_us >= kFallAlertCooldownUs) {
    ESP_LOGW(kTag, "*** PROTOTYPE FALL TRIGGER %.4f ***", probs[1]);
    if(g_alert_task) xTaskNotifyGive(g_alert_task);
    else Beep(kFallBeepMs);
    last_beep_us = esp_timer_get_time();
    now_us = last_beep_us;
    beep_now = true;
  } else if (fall_triggered && CONFIG_FALL_DEBUG_LOGS) {
    ESP_LOGW(kTag, "*** PROTOTYPE FALL TRIGGER %.4f (beep cooldown) ***", probs[1]);
  }

  int64_t cooldown_remaining_us = 0;
  if (last_beep_us >= 0) {
    cooldown_remaining_us = std::max<int64_t>(
        0, kFallAlertCooldownUs - (now_us - last_beep_us));
  }

  portENTER_CRITICAL(&g_power_lock);
  g_result={probs[1],fall_triggered,true,static_cast<int>((elapsed_us+500)/1000),cooldown_remaining_us};
  portEXIT_CRITICAL(&g_power_lock);

  portENTER_CRITICAL(&g_ring_lock);
  const uint64_t sample_count = g_sample_count;
  portEXIT_CRITICAL(&g_ring_lock);
  // Index 1 is fall in the locked binary classifier.
  if(CONFIG_FALL_DEBUG_LOGS) ESP_LOGI(kTag,
           "infer=%lld us | logits=[%.4f %.4f] | normal=%.4f fall=%.4f | sample=%llu%s",
           static_cast<long long>(elapsed_us), logits[0], logits[1], probs[0],
           probs[1], static_cast<unsigned long long>(sample_count),
           beep_now ? " | beep" : "");
  if(CONFIG_FALL_DEBUG_LOGS) ESP_LOGI(kTag, "complete_path=%lld us cadence_budget=%lld us",
      esp_timer_get_time() - path_begin_us, fall_v2::kInferencePeriodUs);
  return true;
}

void TriggerTask(void*) {
  esp_timer_handle_t timer=nullptr;esp_timer_create_args_t args{};
  args.callback=TimerWake;args.arg=xTaskGetCurrentTaskHandle();args.name="trigger_4hz";
  ESP_ERROR_CHECK(esp_timer_create(&args,&timer));
  const int64_t origin=esp_timer_get_time();uint64_t step=1;
  int64_t previous=origin;uint32_t generation=0;
  fall_power::FallbackCadence fallback(origin);
  fall_power::Policy policy;fall_power::Window window{};
  while(true) {
    int64_t deadline=fall_power::StepDeadline(origin,step++);WaitDeadline(timer,deadline);
    const int64_t now=esp_timer_get_time();
    portENTER_CRITICAL(&g_power_lock);
    if(policy.active()) {
      if(g_cascade) g_power.active_us+=now-previous;
      else g_power.shadow_active_us+=now-previous;
    }
    portEXIT_CRITICAL(&g_power_lock);previous=now;
    if(now-deadline>=fall_power::kStrideUs) {
      portENTER_CRITICAL(&g_power_lock);g_power.trigger_deadline_misses+=(now-deadline)/fall_power::kStrideUs;portEXIT_CRITICAL(&g_power_lock);
      g_cascade=false;policy.Reset();step=(now-origin)/fall_power::kStrideUs+1;
      deadline=fall_power::StepDeadline(origin,step-1);
      ESP_LOGE(kTag,"Trigger deadline missed; explicit 750 ms fallback retained");
    }
    if(fall_ota::InProgress() || !CaptureWindow(&window,deadline)) { policy.Reset();continue; }
    if(window.generation!=generation) { policy.Reset();generation=window.generation; }
    float probability=0;const int64_t started=esp_timer_get_time();
    const bool ok=fall_trigger::Evaluate(window,&probability);
    const uint64_t elapsed=esp_timer_get_time()-started;
    portENTER_CRITICAL(&g_power_lock);++g_power.trigger_n;g_power.trigger_us+=elapsed;portEXIT_CRITICAL(&g_power_lock);
    if(!ok) { g_cascade=false;ESP_LOGE(kTag,"Trigger fault; original full-model cadence retained"); }
    const auto decision=policy.Step(deadline,ok && probability>=fall_trigger::Threshold());
    portENTER_CRITICAL(&g_power_lock);
    if(g_cascade) {g_power.wake_n+=decision.woke;g_power.burst_caps+=decision.capped;}
    else {g_power.shadow_wake_n+=decision.woke;g_power.shadow_caps+=decision.capped;}
    portEXIT_CRITICAL(&g_power_lock);
    // Compare nominal grid deadlines: comparing wake-time jitter can postpone
    // a 750 ms fallback by an extra quarter-second even without a missed tick.
    const bool run=g_cascade ? decision.run_full : fallback.Step(deadline);
    if(run) {
      // Queue the immutable W0 used by the trigger, never a recaptured window.
      if(xQueueSend(g_full_queue,&window,0)!=pdPASS) {
        portENTER_CRITICAL(&g_power_lock);++g_power.queue_drops;portEXIT_CRITICAL(&g_power_lock);
        g_cascade=false;policy.Reset();ESP_LOGE(kTag,"Full queue overflow; 750 ms fallback retained");
      }
    }
  }
}

void FullTask(void*) {
  fall_power::Window window{};
  while(true) {
    if(xQueueReceive(g_full_queue,&window,portMAX_DELAY)!=pdPASS) continue;
    if(fall_ota::InProgress()) {
      portENTER_CRITICAL(&g_power_lock);++g_power.queue_drops;portEXIT_CRITICAL(&g_power_lock);continue;
    }
    // A complete pre-gap triggering window stays valid even if later sensing
    // resets. The next trigger waits for a newly complete continuous history.
    const bool valid=RunInference(&window);
    if(!valid || (g_cascade && esp_timer_get_time()-window.deadline_us>=fall_power::kStrideUs)) {
      portENTER_CRITICAL(&g_power_lock);++g_power.full_deadline_misses;portEXIT_CRITICAL(&g_power_lock);
      g_cascade=false;
      const unsigned discarded=uxQueueMessagesWaiting(g_full_queue);xQueueReset(g_full_queue);
      portENTER_CRITICAL(&g_power_lock);g_power.queue_drops+=discarded;portEXIT_CRITICAL(&g_power_lock);
      ESP_LOGE(kTag,"Full model fault/deadline failure; cascade disabled, 750 ms fallback retained");
    }
  }
}


// Mode is stored in a separate namespace and takes effect on a deliberate reboot.
// Never run BLE/recording and detector inference together.
bool ReadMode() {
  nvs_handle_t h;
  if (nvs_open("m5ble", NVS_READWRITE, &h) != ESP_OK) return false;
  uint8_t mode = 0; nvs_get_u8(h, "mode", &mode); nvs_close(h);
  g_collection_mode = mode == 1; return true;
}

void SwitchMode(bool collection) {
  if (fall_ota::InProgress() || (g_collection_mode && !m5ble::CanLeave())) {
    ESP_LOGW(kTag, "Finish recording and save all pending samples before switching modes");
    Beep(30); return;
  }
  nvs_handle_t h;
  if (nvs_open("m5ble", NVS_READWRITE, &h) != ESP_OK) return;
  esp_err_t err = nvs_set_u8(h, "mode", collection ? 1 : 0);
  if (err == ESP_OK) err = nvs_commit(h);
  nvs_close(h);
  if (err != ESP_OK) { ESP_LOGE(kTag, "Mode change not saved"); return; }
  Beep(30); vTaskDelay(pdMS_TO_TICKS(50)); Beep(30); esp_restart();
}

bool InitButtons() {
  gpio_config_t cfg{};
  cfg.pin_bit_mask = (1ULL << GPIO_NUM_37) | (1ULL << GPIO_NUM_39);
  if (g_collection_mode) cfg.pin_bit_mask |= (1ULL << GPIO_NUM_35);  // C button, collection UI only.
  cfg.mode = GPIO_MODE_INPUT;
  cfg.pull_up_en = GPIO_PULLUP_DISABLE;
  cfg.pull_down_en = GPIO_PULLDOWN_DISABLE;
  return gpio_config(&cfg) == ESP_OK;
}

struct Button {
  bool raw = false, stable = false, held = false;
  int64_t changed = 0, pressed = 0;
  int Poll(gpio_num_t pin, int64_t now) {
    bool down = gpio_get_level(pin) == 0;
    if (down != raw) { raw = down; changed = now; }
    if (raw != stable && now - changed >= 30000) {
      stable = raw;
      if (stable) { pressed = now; held = false; }
      else if (!held) return 1;
    }
    if (stable && !held && now - pressed >= 2000000) { held = true; return 2; }
    return 0;
  }
};


float CpuDuty() {
#if CONFIG_FREERTOS_GENERATE_RUN_TIME_STATS
 TaskStatus_t tasks[32];uint32_t total=0;const unsigned n=uxTaskGetSystemState(tasks,32,&total);
 static uint32_t previous_idle[2]={};static int64_t previous_time=0;
 uint32_t idle[2]={};unsigned found=0;
 for(unsigned core=0;core<2;++core) for(unsigned i=0;i<n;++i) if(tasks[i].xHandle==xTaskGetIdleTaskHandleForCore(core)) {idle[core]=tasks[i].ulRunTimeCounter;++found;break;}
 // Use interval deltas modulo uint32, avoiding the runtime counter's ~71 min
 // wrap. Reports are one minute apart; idle tasks retain their identities.
 const int64_t now=esp_timer_get_time(),elapsed=now-previous_time;
 if(found!=2 || elapsed<=0) return -1.f;
 uint64_t idle_delta=0;
 for(unsigned i=0;i<2;++i) { idle_delta+=uint32_t(idle[i]-previous_idle[i]);previous_idle[i]=idle[i]; }
 previous_time=now;
 return std::clamp(1.f-float(idle_delta)/(2.f*elapsed),0.f,1.f);
#else
 return -1.f;
#endif
}
void ReportPower(int64_t now) {
 PowerStats stats;portENTER_CRITICAL(&g_power_lock);stats=g_power;portEXIT_CRITICAL(&g_power_lock);
 ESP_LOGI("fall_power","POWER_JSON {\"uptime_us\":%lld,\"cascade\":%d,\"collection\":%d,\"samples\":%llu,\"gaps\":%llu,\"sensor_us\":%llu,\"trigger_n\":%llu,\"trigger_us\":%llu,\"wake_n\":%llu,\"burst_caps\":%llu,\"full_n\":%llu,\"full_us\":%llu,\"full_path_us\":%llu,\"active_us\":%llu,\"trigger_deadline_misses\":%llu,\"full_deadline_misses\":%llu,\"queue_drops\":%llu,\"display_us\":%llu,\"wifi_us\":%llu,\"shadow_wake_n\":%llu,\"shadow_caps\":%llu,\"shadow_active_us\":%llu,\"cpu_active_fraction\":%.5f,\"full_arena_bytes\":%u,\"trigger_arena_bytes\":%u,\"free_internal\":%u,\"free_psram\":%u}",
   now,int(g_cascade.load()),int(g_collection_mode),stats.samples,stats.gaps,stats.sensor_us,stats.trigger_n,stats.trigger_us,stats.wake_n,stats.burst_caps,stats.full_n,stats.full_us,stats.full_path_us,stats.active_us,stats.trigger_deadline_misses,stats.full_deadline_misses,stats.queue_drops,stats.display_us,stats.wifi_us,stats.shadow_wake_n,stats.shadow_caps,stats.shadow_active_us,CpuDuty(),g_interpreter?unsigned(g_interpreter->arena_used_bytes()):0,fall_trigger::ArenaBytes(),unsigned(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),unsigned(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));
}

}  // namespace

extern "C" void app_main(void) {
  ESP_LOGI(kTag, "M5 BLE collector firmware; locked detector retained");
  if (nvs_flash_init() != ESP_OK || !ReadMode()) {
    ESP_LOGE(kTag, "NVS unavailable; refusing to erase existing settings");
    fall_ota::RollbackPendingImageAndReboot(); return;
  }
  ESP_LOGI(kTag, "Model window: 90 x 6, 30 Hz, inference every 0.75 s");

  if (!InitPowerHold()) {
    ESP_LOGE(kTag, "Failed to drive HOLD (GPIO4) high");
    fall_ota::RollbackPendingImageAndReboot();
    return;
  }
  if (!fall_display::Init()) {
    ESP_LOGW(kTag, "Display initialization failed; continuing without screen UI");
  }
  device_ui::DrawStarting(g_collection_mode);
  InitBuzzer();
  if(!fall_power::InitPowerManagement()) { fall_ota::RollbackPendingImageAndReboot(); return; }
  if (!InitButtons()) { fall_ota::RollbackPendingImageAndReboot(); return; }
  if (!InitI2C() || !InitMpu6886()) {
    ESP_LOGE(kTag, "IMU initialization failed");
    fall_ota::RollbackPendingImageAndReboot();
    return;
  }
  if (!InitModel() || !ModelReplaySelfTest()) {
    ESP_LOGE(kTag, "Model initialization failed");
    fall_ota::RollbackPendingImageAndReboot();
    return;
  }

  // Collector BLE and explicitly requested OTA retain their existing roles.
  // Radio-off detector startup is healthy; acquisition and replay verify it.
  if (g_collection_mode) {
    // Release only RAM allocations after replay; model bytes in flash are untouched.
    delete g_interpreter; g_interpreter = nullptr; g_input = g_output = nullptr;
    heap_caps_free(g_tensor_arena); g_tensor_arena = nullptr;
    heap_caps_free(g_persistent_arena);g_persistent_arena=nullptr;
    if (!m5ble::Init()) {
      ESP_LOGE(kTag, "BLE initialization failed");
      fall_ota::RollbackPendingImageAndReboot(); return;
    }
  } else if(CONFIG_FALL_LEGACY_POWER_BASELINE) {
    if(!fall_ota::Start()) { fall_ota::RollbackPendingImageAndReboot(); return; }
  }
  if (xTaskCreatePinnedToCore(SensorTask, "imu_30hz", 4096, nullptr, 8, nullptr, 1) != pdPASS) {
    ESP_LOGE(kTag, "Failed to start IMU task");
    fall_ota::RollbackPendingImageAndReboot();
    return;
  }
  // Verify acquisition actually runs before accepting a pending update.
  bool acquired=false;
  for(int attempt=0;attempt<100;++attempt) {
    portENTER_CRITICAL(&g_power_lock);acquired=g_power.samples>=30 && g_power.gaps==0;portEXIT_CRITICAL(&g_power_lock);
    if(acquired) break;
    vTaskDelay(pdMS_TO_TICKS(20));
  }
  if(!acquired) { ESP_LOGE(kTag,"Startup acquisition check failed");fall_ota::RollbackPendingImageAndReboot();return; }
  if(CONFIG_FALL_TRIGGER_MODEL && !g_collection_mode) {
    if(!fall_trigger::Init(m5ble::kModelSha256)) {
      ESP_LOGE(kTag,"Trigger identity/replay check failed");fall_ota::RollbackPendingImageAndReboot();return;
    }
    g_cascade=CONFIG_FALL_CASCADE && fall_trigger::Qualified() &&
      g_boot_full_worst_us+fall_trigger::StartupWorstUs()+10000<fall_power::kStrideUs;
    ESP_LOGI(kTag,"Cascade=%d; full startup worst=%llu us, trigger at 80 MHz worst=%llu us",int(g_cascade.load()),g_boot_full_worst_us,fall_trigger::StartupWorstUs());
    if(CONFIG_FALL_CASCADE && !g_cascade) ESP_LOGW(kTag,"Data/deadline gate failed; original 750 ms full-model cadence retained");
    g_full_queue=xQueueCreate(4,sizeof(fall_power::Window));
    if(!g_full_queue ||
       xTaskCreatePinnedToCore(AlertTask,"fall_alert",2048,nullptr,5,&g_alert_task,0)!=pdPASS ||
       xTaskCreatePinnedToCore(FullTask,"full_tcn",8192,nullptr,4,nullptr,0)!=pdPASS ||
       xTaskCreatePinnedToCore(TriggerTask,"trigger_4hz",6144,nullptr,6,nullptr,0)!=pdPASS) {
      ESP_LOGE(kTag,"Cascade task allocation failed");fall_ota::RollbackPendingImageAndReboot();return;
    }
    // Keep buttons/display responsive while the lower-priority TCN computes.
    vTaskPrioritySet(nullptr,5);
  }
  if (!fall_ota::MarkRunningImageValid()) {
    ESP_LOGE(kTag, "Could not confirm the running OTA image");
    fall_ota::RollbackPendingImageAndReboot();
    return;
  }

  // No Internet OTA check: the locked model cannot be replaced automatically.
  // In collection mode Wi-Fi is never initialized.

  ESP_LOGI(kTag,"Normal DETECT: radio/display off. A=status; hold A=OTA; hold B=COLLECT. Existing collection controls retained.");
  ESP_LOGI(kTag, "Ready. Collecting MPU6886 at 30 Hz on a separate sampling task");
  ESP_LOGI(kTag, "DETECT input: full-range m/s^2 and rad/s, checkpoint M5 normalization; cadence=%lld us",
           fall_v2::kInferencePeriodUs);
  m5_battery::Start();
  int64_t next_inference = esp_timer_get_time();
  Button button_a, button_b, button_c;
  int64_t last_ui = 0, last_activity = esp_timer_get_time();
  int64_t both_pressed = 0;
  m5ble::RecordState previous_state = m5ble::RecordState::Ready;
  bool previous_link = true;
  bool screen_awake=g_collection_mode || CONFIG_FALL_LEGACY_POWER_BASELINE;
  fall_display::SetAwake(screen_awake);
  int64_t ota_until=0,last_power_report=esp_timer_get_time(),last_power_loop=last_power_report;
  bool suppress_buttons_until_release = false;
  while (true) {
    const int64_t now = esp_timer_get_time();
    int a = button_a.Poll(GPIO_NUM_37, now);
    int b = button_b.Poll(GPIO_NUM_39, now);
    int c = g_collection_mode ? button_c.Poll(GPIO_NUM_35, now) : 0;

    portENTER_CRITICAL(&g_power_lock);
    if(screen_awake) g_power.display_us+=now-last_power_loop;
    if(fall_ota::Started()) g_power.wifi_us+=now-last_power_loop;
    portEXIT_CRITICAL(&g_power_lock);last_power_loop=now;
    if(now-last_power_report>=60000000) { ReportPower(now);last_power_report=now; }
    if (g_collection_mode && !screen_awake && (a || b || c)) {
      fall_display::SetAwake(true);
      screen_awake = true;
      last_activity = now;
      suppress_buttons_until_release = true;
      a = b = c = 0;
    }
    if (suppress_buttons_until_release) {
      a = b = c = 0;
      if (!button_a.stable && !button_b.stable && (!g_collection_mode || !button_c.stable))
        suppress_buttons_until_release = false;
    }
    if (a || b || c) { last_activity = now; fall_display::SetAwake(true); screen_awake = true; }

    if (g_collection_mode && !suppress_buttons_until_release && button_a.stable && button_b.stable) {
      if (!both_pressed) both_pressed = now;
      if (now - both_pressed >= 5000000 && m5ble::ResetPairing()) {
        Beep(100); esp_restart();
      }
    } else both_pressed = 0;

    if (!g_collection_mode && b == 2 && !button_a.stable) SwitchMode(true);
    if(!g_collection_mode && a==2 && !button_b.stable) {
      if(fall_ota::Started()) { if(fall_ota::Stop()) ota_until=0; }
      else if(fall_ota::Start()) ota_until=esp_timer_get_time()+120000000;
    }
    if(ota_until && now>=ota_until && !fall_ota::InProgress()) { if(fall_ota::Stop()) ota_until=0; }
    if(!g_collection_mode) {
      const bool awake=CONFIG_FALL_LEGACY_POWER_BASELINE || fall_ota::Started() || now<g_alert_until || now-last_activity<15000000;
      fall_display::SetAwake(awake);screen_awake=awake;
    }
    if (g_collection_mode) {
      m5ble::Tick();
      if (m5ble::DetectRequested()) SwitchMode(false);
      auto status = m5ble::GetStatus();

      if (a == 1) {
        if (status.state == m5ble::RecordState::Ready || status.state == m5ble::RecordState::Complete)
          m5ble::StartRecording();
        else if (status.state == m5ble::RecordState::Recording)
          m5ble::StopRecording();
        else if (status.state == m5ble::RecordState::Review || status.state == m5ble::RecordState::Full)
          m5ble::KeepRecording();
      }
      if (b == 1 && (status.state == m5ble::RecordState::Review || status.state == m5ble::RecordState::Full))
        m5ble::DiscardRecording();
      if (c == 1 && m5ble::CanLeave()) SwitchMode(false);

      status = m5ble::GetStatus();
      if (status.state != previous_state) {
        Beep(status.state == m5ble::RecordState::Full ? 150 : 30);
        previous_state = status.state;
        last_activity = now;
        fall_display::SetAwake(true);
        screen_awake = true;
      }
      if (!status.connected && previous_link && status.state == m5ble::RecordState::Saving) Beep(60);
      previous_link = status.connected;
      if (screen_awake && now - last_ui >= 250000) {
        device_ui::DrawCollectionScreen(status, m5_battery::Percentage(), now); last_ui = now;
      }

      const bool may_sleep = status.state == m5ble::RecordState::Ready || status.state == m5ble::RecordState::Complete;
      if (may_sleep && screen_awake && now - last_activity > 60000000) {
        fall_display::SetAwake(false);
        screen_awake = false;
      }
      vTaskDelay(pdMS_TO_TICKS(5)); continue;
    }
    portENTER_CRITICAL(&g_ring_lock);
    const bool ready = g_ring_count == kTimesteps;
    portEXIT_CRITICAL(&g_ring_lock);
    if (!g_full_queue && !fall_ota::InProgress() && ready && esp_timer_get_time() >= next_inference) {
      const int64_t started = esp_timer_get_time();
      RunInference();
      next_inference = started + fall_v2::kInferencePeriodUs;
    }
    if (screen_awake && esp_timer_get_time() - last_ui >= 250000) {
      ResultState result;portENTER_CRITICAL(&g_power_lock);result=g_result;portEXIT_CRITICAL(&g_power_lock);
      if(result.valid) fall_display::ShowResult(result.probability,result.fall,result.milliseconds,result.cooldown_us);
      device_ui::DrawDetectScreen(m5_battery::Percentage()); last_ui = esp_timer_get_time();
    }
    // Five milliseconds rounds to zero at a 100 Hz RTOS tick. Yield at least
    // one tick so the idle task can service the watchdog between predictions.
    vTaskDelay(pdMS_TO_TICKS(25));
  }
}
