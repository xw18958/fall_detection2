#include "fall_op_resolver.h"

#include "gelu_kernel.h"
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
    tflite::MicroPrintf("Failed to register TFLM op: %s", name);
  }
}

}  // namespace

FallOpResolver::FallOpResolver() : gelu_registration_(RegisterGelu()) {
  Check(base_.AddTranspose(), "TRANSPOSE");
  Check(base_.AddConv2D(), "CONV_2D");
  Check(base_.AddReshape(), "RESHAPE");
  Check(base_.AddAdd(), "ADD");
  Check(base_.AddSoftmax(), "SOFTMAX");
  Check(base_.AddMul(), "MUL");
  Check(base_.AddSum(), "SUM");
  Check(base_.AddMean(), "MEAN");
  Check(base_.AddSub(), "SUB");
  Check(base_.AddSqrt(), "SQRT");
  Check(base_.AddDiv(), "DIV");
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
