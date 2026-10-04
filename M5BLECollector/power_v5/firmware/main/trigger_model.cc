#include "trigger_model.h"
#include "sdkconfig.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include "esp_timer.h"
#if CONFIG_FALL_TRIGGER_MODEL
#include "trigger_bundle.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
namespace {
alignas(16) uint8_t arena[24*1024];
tflite::MicroMutableOpResolver<8> resolver;
tflite::MicroInterpreter* interpreter = nullptr;
uint64_t startup_worst_us=0;
}
#endif
namespace fall_trigger {
bool Init(const char* full_model_sha256) {
#if CONFIG_FALL_TRIGGER_MODEL
  if (std::strcmp(full_model_sha256, trigger_bundle::kFullModelSha256) ||
      trigger_bundle::kSampleHz != 30 || trigger_bundle::kSamples != 90) return false;
  if (resolver.AddConv2D()!=kTfLiteOk || resolver.AddDepthwiseConv2D()!=kTfLiteOk ||
      resolver.AddReshape()!=kTfLiteOk || resolver.AddMean()!=kTfLiteOk ||
      resolver.AddFullyConnected()!=kTfLiteOk || resolver.AddPad()!=kTfLiteOk ||
      resolver.AddReduceMax()!=kTfLiteOk || resolver.AddConcatenation()!=kTfLiteOk) return false;
  const auto* model = tflite::GetModel(trigger_bundle::kModel);
  if (model->version()!=TFLITE_SCHEMA_VERSION) return false;
  static tflite::MicroInterpreter instance(model, resolver, arena, sizeof(arena));
  interpreter = &instance;
  if (interpreter->AllocateTensors()!=kTfLiteOk) return false;
  auto* in=interpreter->input(0); auto* out=interpreter->output(0);
  if (in->type!=kTfLiteInt8 || out->type!=kTfLiteInt8 || in->dims->size!=3 ||
      in->dims->data[0]!=1 || in->dims->data[1]!=90 || in->dims->data[2]!=6 ||
      out->bytes!=2 || in->params.scale<=0 || out->params.scale<=0) return false;
  for (int n=0;n<trigger_bundle::kReplayCount;++n) {
    std::memcpy(in->data.int8, trigger_bundle::kReplayInputs[n], 540);
    const int64_t started=esp_timer_get_time();
    if (interpreter->Invoke()!=kTfLiteOk) return false;
    startup_worst_us=std::max(startup_worst_us,uint64_t(esp_timer_get_time()-started));
    for(int c=0;c<2;++c) if(std::abs(int(out->data.int8[c])-int(trigger_bundle::kReplayOutputs[n][c]))>1) return false;
  }
  return true;
#else
  (void)full_model_sha256; return true;
#endif
}
bool Evaluate(const fall_power::Window& window, float* probability) {
#if CONFIG_FALL_TRIGGER_MODEL
  if (!interpreter || !probability) return false;
  auto* in=interpreter->input(0);
  for(int t=0;t<90;++t) for(int c=0;c<6;++c) {
    const int q=std::lround(window.x[t][c]/in->params.scale)+in->params.zero_point;
    in->data.int8[t*6+c]=std::max(-128,std::min(127,q));
  }
  if (interpreter->Invoke()!=kTfLiteOk) return false;
  auto* out=interpreter->output(0);
  const float difference=(int(out->data.int8[0])-int(out->data.int8[1]))*out->params.scale;
  *probability=1.f/(1.f+std::exp(difference)); return std::isfinite(*probability);
#else
  (void)window; (void)probability; return false;
#endif
}
bool Qualified() {
#if CONFIG_FALL_TRIGGER_MODEL
  return trigger_bundle::kQualified;
#else
  return false;
#endif
}
float Threshold() {
#if CONFIG_FALL_TRIGGER_MODEL
  return trigger_bundle::kThreshold;
#else
  return 1.f;
#endif
}
unsigned ArenaBytes() {
#if CONFIG_FALL_TRIGGER_MODEL
  return interpreter ? interpreter->arena_used_bytes() : 0;
#else
  return 0;
#endif
}
uint64_t StartupWorstUs() {
#if CONFIG_FALL_TRIGGER_MODEL
  return startup_worst_us;
#else
  return 0;
#endif
}
}  // namespace fall_trigger
