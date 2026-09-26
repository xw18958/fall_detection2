#include "gelu_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "tensorflow/compiler/mlir/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace fall_tflm {
namespace {

struct GeluOpData {
  TfLiteType type;
  float input_scale;
  int32_t input_zero_point;
  float output_scale;
  int32_t output_zero_point;
  bool approximate;
};

void* GeluInit(TfLiteContext* context, const char*, size_t) {
  return context->AllocatePersistentBuffer(context, sizeof(GeluOpData));
}

TfLiteStatus GeluPrepare(TfLiteContext* context, TfLiteNode* node) {
  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteInt8);

  auto* data = static_cast<GeluOpData*>(node->user_data);
  TF_LITE_ENSURE(context, data != nullptr);
  data->type = input->type;
  data->input_scale = input->params.scale;
  data->input_zero_point = input->params.zero_point;
  data->output_scale = output->params.scale;
  data->output_zero_point = output->params.zero_point;
  const auto* params = static_cast<const TfLiteGeluParams*>(node->builtin_data);
  data->approximate = params != nullptr ? params->approximate : false;

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

inline float GeluFloat(float x, bool approximate) {
  if (approximate) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    const float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
  }
  constexpr float kInvSqrt2 = 0.7071067811865475f;
  return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
}

TfLiteStatus GeluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);
  auto* data = static_cast<GeluOpData*>(node->user_data);
  TF_LITE_ENSURE(context, data != nullptr);

  const size_t count = tflite::ElementCount(*input->dims);
  if (input->type == kTfLiteFloat32) {
    const float* src = tflite::micro::GetTensorData<float>(input);
    float* dst = tflite::micro::GetTensorData<float>(output);
    for (size_t i = 0; i < count; ++i) {
      dst[i] = GeluFloat(src[i], data->approximate);
    }
    return kTfLiteOk;
  }

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE(context, data->input_scale > 0.0f);
    TF_LITE_ENSURE(context, data->output_scale > 0.0f);
    const int8_t* src = tflite::micro::GetTensorData<int8_t>(input);
    int8_t* dst = tflite::micro::GetTensorData<int8_t>(output);
    for (size_t i = 0; i < count; ++i) {
      const float x =
          (static_cast<int32_t>(src[i]) - data->input_zero_point) *
          data->input_scale;
      const float y = GeluFloat(x, data->approximate);
      int32_t q = static_cast<int32_t>(std::lround(y / data->output_scale)) +
                  data->output_zero_point;
      q = std::max(-128, std::min(127, q));
      dst[i] = static_cast<int8_t>(q);
    }
    return kTfLiteOk;
  }

  tflite::MicroPrintf("GELU: unsupported type %d", input->type);
  return kTfLiteError;
}

}  // namespace

TFLMRegistration RegisterGelu() {
  return tflite::micro::RegisterOp(GeluInit, GeluPrepare, GeluEval);
}

}  // namespace fall_tflm
