#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#include "driver/gpio.h"
#include "driver/i2c.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "fall_op_resolver.h"
#include "simple_display.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr char kTag[] = "fall_tflm";

// M5StickC PLUS2 pins.
constexpr gpio_num_t kHoldPin = GPIO_NUM_4;
constexpr gpio_num_t kBuzzerPin = GPIO_NUM_2;
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

constexpr int kTimesteps = 60;
constexpr int kChannels = 6;
constexpr int kStride = 15;
constexpr float kSampleHz = 20.0f;
constexpr int64_t kSamplePeriodUs = 50000;

// Checkpoint normalization from the uploaded conversion package.
constexpr float kMean[kChannels] = {
    1.6522459984f, -1.6432752609f, 0.7402734160f,
   -0.0017731582f, -0.0025698263f, -0.0017951268f,
};
constexpr float kSigma[kChannels] = {
    5.0859293938f, 7.0654754639f, 4.4901218414f,
    0.1828159988f, 0.2244342566f, 0.1733540148f,
};

// Prototype live threshold chosen from real-device testing of the fine-tuned C16 model.
constexpr float kPrototypeFallThreshold = 0.88f;
constexpr int64_t kFallAlertCooldownUs = 3000000;
constexpr int kFallBeepMs = 100;

// MPU6886 configured to ±8 g and ±2000 deg/s.
constexpr float kAccelGPerLsb = 8.0f / 32768.0f;
constexpr float kGyroDpsPerLsb = 2000.0f / 32768.0f;
constexpr float kGravityMps2 = 9.80665f;
constexpr float kDegToRad = 0.01745329251994329577f;

// Embedded by PlatformIO's board_build.embed_files mechanism.
extern const uint8_t model_tflite_start[] asm("_binary_model_tflite_start");
extern const uint8_t model_tflite_end[] asm("_binary_model_tflite_end");

float g_ring[kTimesteps][kChannels] = {};
int g_ring_pos = 0;
int g_ring_count = 0;
int g_since_inference = 0;
uint64_t g_sample_count = 0;

uint8_t* g_tensor_arena = nullptr;
size_t g_tensor_arena_size = 0;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;
fall_tflm::FallOpResolver g_resolver;

struct RawImu {
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
  gpio_config_t cfg{};
  cfg.pin_bit_mask = 1ULL << kBuzzerPin;
  cfg.mode = GPIO_MODE_OUTPUT;
  cfg.pull_up_en = GPIO_PULLUP_DISABLE;
  cfg.pull_down_en = GPIO_PULLDOWN_DISABLE;
  cfg.intr_type = GPIO_INTR_DISABLE;
  if (gpio_config(&cfg) != ESP_OK) return false;
  gpio_set_level(kBuzzerPin, 0);
  return true;
}

void Beep(int milliseconds) {
  // Passive buzzer: ~2 kHz square wave.
  const int half_period_us = 250;
  const int cycles = (milliseconds * 1000) / (half_period_us * 2);
  for (int i = 0; i < cycles; ++i) {
    gpio_set_level(kBuzzerPin, 1);
    esp_rom_delay_us(half_period_us);
    gpio_set_level(kBuzzerPin, 0);
    esp_rom_delay_us(half_period_us);
  }
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

bool ReadImu(RawImu* out) {
  uint8_t data[14] = {};
  if (ReadRegisters(kRegAccelXoutH, data, sizeof(data)) != ESP_OK) return false;

  const int16_t ax = ReadBe16(data + 0);
  const int16_t ay = ReadBe16(data + 2);
  const int16_t az = ReadBe16(data + 4);
  const int16_t gx = ReadBe16(data + 8);
  const int16_t gy = ReadBe16(data + 10);
  const int16_t gz = ReadBe16(data + 12);

  out->ax_g = ax * kAccelGPerLsb;
  out->ay_g = ay * kAccelGPerLsb;
  out->az_g = az * kAccelGPerLsb;
  out->gx_dps = gx * kGyroDpsPerLsb;
  out->gy_dps = gy * kGyroDpsPerLsb;
  out->gz_dps = gz * kGyroDpsPerLsb;
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

  const tflite::Model* model = tflite::GetModel(model_tflite_start);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(kTag, "TFLite schema mismatch: model=%d runtime=%d",
             model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  const size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  ESP_LOGI(kTag, "PSRAM initialized=%d, largest free block=%u bytes",
           esp_psram_is_initialized(), static_cast<unsigned>(largest));
  if (!esp_psram_is_initialized() || largest < 768 * 1024) {
    ESP_LOGE(kTag, "Not enough PSRAM for the TensorFlow Lite Micro arena");
    return false;
  }

  constexpr size_t kPreferredArena = 1500 * 1024;
  constexpr size_t kReserve = 96 * 1024;
  g_tensor_arena_size = std::min(kPreferredArena,
                                 largest > kReserve ? largest - kReserve : largest);
  g_tensor_arena = static_cast<uint8_t*>(
      heap_caps_malloc(g_tensor_arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (g_tensor_arena == nullptr) {
    ESP_LOGE(kTag, "Failed to allocate %u-byte tensor arena",
             static_cast<unsigned>(g_tensor_arena_size));
    return false;
  }
  ESP_LOGI(kTag, "Tensor arena: %u bytes in PSRAM",
           static_cast<unsigned>(g_tensor_arena_size));

  g_interpreter = new (std::nothrow)
      tflite::MicroInterpreter(model, g_resolver, g_tensor_arena,
                               g_tensor_arena_size);
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
    ESP_LOGE(kTag, "Unexpected model input shape; expected [1,60,6]");
    return false;
  }
  if (g_output->dims->size != 2 || g_output->dims->data[0] != 1 ||
      g_output->dims->data[1] != 2) {
    ESP_LOGE(kTag, "Unexpected model output shape; expected [1,2]");
    return false;
  }
  return true;
}

inline float Normalize(int channel, float value) {
  return (value - kMean[channel]) / (kSigma[channel] + 1.0e-6f);
}

void PushSample(const RawImu& imu) {
  // Best-effort unit alignment to the SI-style training stream:
  // M5 MPU6886: accel is read in g; gyro is read in deg/s.
  // Feed the model acceleration in m/s^2 and angular velocity in rad/s.
  const float physical[kChannels] = {
      imu.ax_g * kGravityMps2,
      imu.ay_g * kGravityMps2,
      imu.az_g * kGravityMps2,
      imu.gx_dps * kDegToRad,
      imu.gy_dps * kDegToRad,
      imu.gz_dps * kDegToRad,
  };

  for (int c = 0; c < kChannels; ++c) {
    g_ring[g_ring_pos][c] = Normalize(c, physical[c]);
  }
  g_ring_pos = (g_ring_pos + 1) % kTimesteps;
  if (g_ring_count < kTimesteps) ++g_ring_count;
  ++g_sample_count;

  if (g_ring_count == kTimesteps) {
    ++g_since_inference;
  }
}

bool FillModelInput() {
  if (g_ring_count < kTimesteps || g_input == nullptr) return false;

  // When full, g_ring_pos points at the oldest sample.
  if (g_input->type == kTfLiteFloat32) {
    float* dst = g_input->data.f;
    for (int t = 0; t < kTimesteps; ++t) {
      const int src_t = (g_ring_pos + t) % kTimesteps;
      for (int c = 0; c < kChannels; ++c) {
        dst[t * kChannels + c] = g_ring[src_t][c];
      }
    }
    return true;
  }

  if (g_input->type == kTfLiteInt8) {
    if (g_input->params.scale <= 0.0f) return false;
    int8_t* dst = g_input->data.int8;
    for (int t = 0; t < kTimesteps; ++t) {
      const int src_t = (g_ring_pos + t) % kTimesteps;
      for (int c = 0; c < kChannels; ++c) {
        const float x = g_ring[src_t][c];
        int32_t q = static_cast<int32_t>(std::lround(x / g_input->params.scale)) +
                    g_input->params.zero_point;
        q = std::max<int32_t>(-128, std::min<int32_t>(127, q));
        dst[t * kChannels + c] = static_cast<int8_t>(q);
      }
    }
    return true;
  }

  ESP_LOGE(kTag, "Unsupported model input type: %d", g_input->type);
  return false;
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

void RunInference() {
  if (!FillModelInput()) return;
  const int64_t begin_us = esp_timer_get_time();
  if (g_interpreter->Invoke() != kTfLiteOk) {
    ESP_LOGE(kTag, "TFLM Invoke() failed");
    return;
  }
  const int64_t elapsed_us = esp_timer_get_time() - begin_us;

  float logits[2] = {};
  if (!ReadLogits(logits)) return;
  float probs[2] = {};
  Softmax2(logits, probs);

  const bool fall_triggered = probs[1] >= kPrototypeFallThreshold;
  fall_display::ShowResult(probs[1], fall_triggered,
                           static_cast<int>((elapsed_us + 500) / 1000));

  // Assumption inherited from the original binary classifier: index 1 = fall.
  ESP_LOGI(kTag,
           "infer=%lld us | logits=[%.4f %.4f] | normal=%.4f fall=%.4f | sample=%llu",
           static_cast<long long>(elapsed_us), logits[0], logits[1], probs[0],
           probs[1], static_cast<unsigned long long>(g_sample_count));

  static int64_t last_beep_us = -kFallAlertCooldownUs;
  if (fall_triggered) {
    ESP_LOGW(kTag, "*** PROTOTYPE FALL TRIGGER %.4f ***", probs[1]);
    const int64_t now_us = esp_timer_get_time();
    if (now_us - last_beep_us >= kFallAlertCooldownUs) {
      last_beep_us = now_us;
      Beep(kFallBeepMs);
    }
  }
}

}  // namespace

extern "C" void app_main(void) {
  ESP_LOGI(kTag, "M5StickC PLUS2 fall-model smoke-test firmware");
  ESP_LOGI(kTag, "Model window: 60 x 6, 20 Hz, stride 15 (3.0 s / 0.75 s)");

  if (!InitPowerHold()) {
    ESP_LOGE(kTag, "Failed to drive HOLD (GPIO4) high");
    return;
  }
  if (!fall_display::Init()) {
    ESP_LOGW(kTag, "Display initialization failed; continuing without screen UI");
  }
  InitBuzzer();
  if (!InitI2C() || !InitMpu6886()) {
    ESP_LOGE(kTag, "IMU initialization failed");
    return;
  }
  if (!InitModel()) {
    ESP_LOGE(kTag, "Model initialization failed");
    return;
  }

  ESP_LOGI(kTag, "Ready. Collecting MPU6886 at 20 Hz...");
  ESP_LOGI(kTag, "CSV: ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");

  int64_t next_sample_us = esp_timer_get_time();
  bool first_inference = true;
  while (true) {
    next_sample_us += kSamplePeriodUs;
    RawImu imu{};
    if (ReadImu(&imu)) {
      PushSample(imu);

      // Print raw sensor values at 5 Hz so serial output does not dominate timing.
      if ((g_sample_count % 4) == 0) {
        ESP_LOGI(kTag, "IMU,%lld,%.5f,%.5f,%.5f,%.3f,%.3f,%.3f",
                 static_cast<long long>(esp_timer_get_time() / 1000),
                 imu.ax_g, imu.ay_g, imu.az_g,
                 imu.gx_dps, imu.gy_dps, imu.gz_dps);
      }

      if (g_ring_count == kTimesteps &&
          (first_inference || g_since_inference >= kStride)) {
        first_inference = false;
        g_since_inference = 0;
        RunInference();
      }
    } else {
      ESP_LOGW(kTag, "IMU read failed");
    }

    const int64_t sleep_us = next_sample_us - esp_timer_get_time();
    if (sleep_us > 1000) {
      vTaskDelay(pdMS_TO_TICKS(static_cast<uint32_t>(sleep_us / 1000)));
    } else if (sleep_us < -kSamplePeriodUs) {
      // If inference or logging overruns substantially, restart the cadence
      // instead of trying to execute a burst of catch-up samples.
      next_sample_us = esp_timer_get_time();
    }
  }
}
