#include "parallel_conv1d.h"
#include "fast_conv1d.h"
#ifdef ESP_PLATFORM
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#endif
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace fall_tflm {
namespace {
// One full interpreter invokes serially. A worker only computes disjoint rows;
// it never enters Micro, allocates memory, or accesses the rolling IMU buffer.
struct Job {
  tflite::ConvParams params;
  int width, channels, kw, rows, outputs;
  const int8_t *input, *filter;
  const int32_t *bias, *multiplier, *shift;
  int8_t* output;
};
void Compute(const Job& j) {
  FastConv1D(j.params,j.width,j.channels,j.kw,j.rows,j.outputs,j.input,
             j.filter,j.bias,j.multiplier,j.shift,j.output);
}
#ifdef ESP_PLATFORM
Job job{};
TaskHandle_t worker=nullptr;
SemaphoreHandle_t complete=nullptr;
void Worker(void*) {
  while(true) {
    ulTaskNotifyTake(pdTRUE,portMAX_DELAY);
    Compute(job);
    xSemaphoreGive(complete);
  }
}
#endif
TfLiteStatus Invoke(TfLiteContext* ctx,TfLiteNode* node) {
  const auto* in=tflite::micro::GetEvalInput(ctx,node,0);
  const auto* weights=tflite::micro::GetEvalInput(ctx,node,1);
  const auto* bias=tflite::NumInputs(node)==3?tflite::micro::GetEvalInput(ctx,node,2):nullptr;
  auto* out=tflite::micro::GetEvalOutput(ctx,node,0);
  const auto& params=*static_cast<const TfLiteConvParams*>(node->builtin_data);
  const auto& data=*static_cast<const tflite::OpDataConv*>(node->user_data);
  Job first{tflite::ConvParamsQuantized(params,data),in->dims->data[2],in->dims->data[3],
    weights->dims->data[2],out->dims->data[2],out->dims->data[3],in->data.int8,
    weights->data.int8,bias?bias->data.i32:nullptr,data.per_channel_output_multiplier,
    data.per_channel_output_shift,out->data.int8};
  const int64_t macs=int64_t(first.rows)*first.outputs*first.kw*first.channels;
  const bool available=
#ifdef ESP_PLATFORM
      worker && xPortGetCoreID()==0;
#else
      true;
#endif
  if(!available || first.rows<2 || macs<8192) {
    Compute(first);return kTfLiteOk;
  }
  const int split=(first.rows+1)/2;
#ifndef ESP_PLATFORM
  Job job{};
#endif
  job=first;
  job.rows-=split;
  // For a local output x, origin becomes (x+split)*stride-original_pad.
  // Keep the complete input: boundary and dilation arithmetic stays identical.
  job.params.padding_values.width-=split*job.params.stride_width;
  job.output+=split*job.outputs;
  first.rows=split;
#ifdef ESP_PLATFORM
  xTaskNotifyGive(worker);
  Compute(first);
  xSemaphoreTake(complete,portMAX_DELAY); // Both row ranges complete before next op.
#else
  Compute(first);Compute(job); // Native parity check of the same partition math.
#endif
  return kTfLiteOk;
}
}
TFLMRegistration RegisterParallelConv1D() {
  auto registration=RegisterFastConv1D();
#ifdef ESP_PLATFORM
  if(!complete) complete=xSemaphoreCreateBinary();
  if(complete && !worker)
    xTaskCreatePinnedToCore(Worker,"conv_rows",4096,nullptr,4,&worker,1);
  // Allocation failure safely retains serial inference and its deadline gate.
  if(worker) registration.invoke=Invoke;
#else
  registration.invoke=Invoke;
#endif
  return registration;
}
}
