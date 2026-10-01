#pragma once
#include <cstdint>
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace fall_tflm {
// Same integer accumulation, padding and requantization as the TFLM reference.
void FastConv1D(const tflite::ConvParams& params, int width, int channels,
                int filter_width, int output_width, int output_channels,
                const int8_t* input, const int8_t* filter, const int32_t* bias,
                const int32_t* multiplier, const int32_t* shift, int8_t* output);
TFLMRegistration RegisterFastConv1D();
}
