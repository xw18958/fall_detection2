#pragma once

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"

namespace fall_tflm {

class FallOpResolver final : public tflite::MicroOpResolver {
 public:
  FallOpResolver();

  const TFLMRegistration* FindOp(
      tflite::BuiltinOperator op) const override;
  const TFLMRegistration* FindOp(const char* op) const override;
  tflite::TfLiteBridgeBuiltinParseFunction GetOpDataParser(
      tflite::BuiltinOperator op) const override;

 private:
  // Slightly over-provisioned so conversion variants with Q/DQ at the I/O work.
  tflite::MicroMutableOpResolver<20> base_;
  TFLMRegistration gelu_registration_;
};

}  // namespace fall_tflm
