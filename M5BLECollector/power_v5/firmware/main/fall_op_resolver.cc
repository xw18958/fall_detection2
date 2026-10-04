#include "fall_op_resolver.h"

#include "gelu_kernel.h"
#include "layer_norm_v4.h"
#include "fast_conv1d.h"
#ifdef ESP_PLATFORM
#include "sdkconfig.h"
#if CONFIG_FALL_PARALLEL_CONV
#include "parallel_conv1d.h"
#endif
#endif
#ifdef FALL_NATIVE_SPLIT_CONV
#include "parallel_conv1d.h"
#endif
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace fall_tflm {
namespace {

TfLiteStatus ParseGelu(const tflite::Operator* op,
                       tflite::ErrorReporter* error_reporter,
                       tflite::BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  return tflite::ParseOpData(op, tflite::BuiltinOperator_GELU, error_reporter,
                             allocator, builtin_data);
}

void Check(TfLiteStatus status, const char* name) {
  if (status != kTfLiteOk) {
    MicroPrintf("Failed to register TFLM op: %s", name);
  }
}

}  // namespace

FallOpResolver::FallOpResolver() : gelu_registration_(RegisterGelu()), layer_norm_registration_(RegisterLayerNormV4()) {
  Check(base_.AddCustom("LayerNormV4", &layer_norm_registration_), "LayerNormV4");
  Check(base_.AddTranspose(), "TRANSPOSE");
#if (defined(ESP_PLATFORM) && CONFIG_FALL_PARALLEL_CONV) || defined(FALL_NATIVE_SPLIT_CONV)
  Check(base_.AddConv2D(RegisterParallelConv1D()), "CONV_2D");
#else
  Check(base_.AddConv2D(RegisterFastConv1D()), "CONV_2D");
#endif
  Check(base_.AddReshape(), "RESHAPE");
  Check(base_.AddConcatenation(), "CONCATENATION");
  Check(base_.AddPack(), "PACK");
  Check(base_.AddStridedSlice(), "STRIDED_SLICE");
  Check(base_.AddAdd(), "ADD");
  Check(base_.AddSoftmax(), "SOFTMAX");
  Check(base_.AddMul(), "MUL");
  Check(base_.AddSum(), "SUM");
  Check(base_.AddMean(), "MEAN");
  Check(base_.AddSub(), "SUB");
  Check(base_.AddSqrt(), "SQRT");
  Check(base_.AddDiv(), "DIV");
  Check(base_.AddNeg(), "NEG");
  Check(base_.AddSquaredDifference(), "SQUARED_DIFFERENCE");
  Check(base_.AddRsqrt(), "RSQRT");
  Check(base_.AddFullyConnected(), "FULLY_CONNECTED");
  Check(base_.AddQuantize(), "QUANTIZE");
  Check(base_.AddDequantize(), "DEQUANTIZE");
}

const TFLMRegistration* FallOpResolver::FindOp(
    tflite::BuiltinOperator op) const {
  if (op == tflite::BuiltinOperator_GELU) {
    return &gelu_registration_;
  }
  return base_.FindOp(op);
}

const TFLMRegistration* FallOpResolver::FindOp(const char* op) const {
  return base_.FindOp(op);
}

tflite::TfLiteBridgeBuiltinParseFunction FallOpResolver::GetOpDataParser(
    tflite::BuiltinOperator op) const {
  if (op == tflite::BuiltinOperator_GELU) {
    return ParseGelu;
  }
  return base_.GetOpDataParser(op);
}

}  // namespace fall_tflm
