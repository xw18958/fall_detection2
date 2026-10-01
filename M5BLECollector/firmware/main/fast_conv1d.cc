#include "fast_conv1d.h"
#include <algorithm>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace fall_tflm {
void FastConv1D(const tflite::ConvParams& p, int width, int channels,
                int kw, int out_width, int out_channels,
                const int8_t* input, const int8_t* filter, const int32_t* bias,
                const int32_t* multiplier, const int32_t* shift, int8_t* output) {
  const int dilation = p.dilation_width_factor;
  for (int x = 0; x < out_width; ++x) {
    const int origin = x * p.stride_width - p.padding_values.width;
    const int lo = std::max(0, (-origin + dilation - 1) / dilation);
    const int hi = origin >= width ? 0 : std::min(kw, (width - 1 - origin) / dilation + 1);
    int oc = 0;
    for (; oc + 1 < out_channels; oc += 2) {
      int32_t a = bias ? bias[oc] : 0, b = bias ? bias[oc + 1] : 0;
      for (int k = lo; k < hi; ++k) {
        const int8_t* s = input + (origin + k * dilation) * channels;
        const int8_t* w0 = filter + (oc * kw + k) * channels;
        const int8_t* w1 = w0 + kw * channels;
        int c = 0;
        for (; c + 3 < channels; c += 4) {
          const int32_t v0 = s[c] + p.input_offset;
          const int32_t v1 = s[c + 1] + p.input_offset;
          const int32_t v2 = s[c + 2] + p.input_offset;
          const int32_t v3 = s[c + 3] + p.input_offset;
          a += v0*w0[c] + v1*w0[c+1] + v2*w0[c+2] + v3*w0[c+3];
          b += v0*w1[c] + v1*w1[c+1] + v2*w1[c+2] + v3*w1[c+3];
        }
        for (; c < channels; ++c) {
          const int32_t v = s[c] + p.input_offset;
          a += v * w0[c]; b += v * w1[c];
        }
      }
      a = tflite::MultiplyByQuantizedMultiplier(a, multiplier[oc], shift[oc]) + p.output_offset;
      b = tflite::MultiplyByQuantizedMultiplier(b, multiplier[oc+1], shift[oc+1]) + p.output_offset;
      output[x*out_channels+oc] = static_cast<int8_t>(std::clamp(a, p.quantized_activation_min, p.quantized_activation_max));
      output[x*out_channels+oc+1] = static_cast<int8_t>(std::clamp(b, p.quantized_activation_min, p.quantized_activation_max));
    }
    if (oc < out_channels) {
      int32_t a = bias ? bias[oc] : 0;
      for (int k = lo; k < hi; ++k) {
        const int8_t* s = input + (origin+k*dilation)*channels;
        const int8_t* w = filter + (oc*kw+k)*channels;
        for (int c=0; c<channels; ++c) a += (s[c]+p.input_offset)*w[c];
      }
      a = tflite::MultiplyByQuantizedMultiplier(a, multiplier[oc], shift[oc]) + p.output_offset;
      output[x*out_channels+oc] = static_cast<int8_t>(std::clamp(a, p.quantized_activation_min, p.quantized_activation_max));
    }
  }
}

namespace {
TfLiteStatus Prepare(TfLiteContext* ctx, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(tflite::ConvPrepare(ctx, node));
  const auto* in = tflite::micro::GetEvalInput(ctx,node,0);
  const auto* w = tflite::micro::GetEvalInput(ctx,node,1);
  const auto* out = tflite::micro::GetEvalOutput(ctx,node,0);
  TF_LITE_ENSURE_EQ(ctx,in->type,kTfLiteInt8);
  TF_LITE_ENSURE_EQ(ctx,w->type,kTfLiteInt8);
  TF_LITE_ENSURE_EQ(ctx,out->type,kTfLiteInt8);
  TF_LITE_ENSURE_EQ(ctx,in->dims->size,4);
  TF_LITE_ENSURE_EQ(ctx,w->dims->size,4);
  TF_LITE_ENSURE_EQ(ctx,out->dims->size,4);
  TF_LITE_ENSURE_EQ(ctx,in->dims->data[0],1);
  TF_LITE_ENSURE_EQ(ctx,in->dims->data[1],1);
  TF_LITE_ENSURE_EQ(ctx,w->dims->data[1],1);
  TF_LITE_ENSURE_EQ(ctx,out->dims->data[0],1);
  TF_LITE_ENSURE_EQ(ctx,out->dims->data[1],1);
  TF_LITE_ENSURE_EQ(ctx,in->dims->data[3],w->dims->data[3]);
  TF_LITE_ENSURE_EQ(ctx,w->dims->data[0],out->dims->data[3]);
  const auto& p = *static_cast<const TfLiteConvParams*>(node->builtin_data);
  TF_LITE_ENSURE_EQ(ctx,p.stride_height,1);
  TF_LITE_ENSURE_EQ(ctx,p.dilation_height_factor,1);
  TF_LITE_ENSURE(ctx,p.dilation_width_factor>0);
  const auto& data = *static_cast<const tflite::OpDataConv*>(node->user_data);
  TF_LITE_ENSURE_EQ(ctx,data.filter_zero_point,0);
  if (tflite::NumInputs(node)==3) {
    const auto* bias=tflite::micro::GetEvalInput(ctx,node,2);
    TF_LITE_ENSURE_EQ(ctx,bias->type,kTfLiteInt32);
  }
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* ctx,TfLiteNode* node) {
  const auto* in=tflite::micro::GetEvalInput(ctx,node,0);
  const auto* w=tflite::micro::GetEvalInput(ctx,node,1);
  const auto* bias=tflite::NumInputs(node)==3 ? tflite::micro::GetEvalInput(ctx,node,2) : nullptr;
  auto* out=tflite::micro::GetEvalOutput(ctx,node,0);
  const auto& p=*static_cast<const TfLiteConvParams*>(node->builtin_data);
  const auto& d=*static_cast<const tflite::OpDataConv*>(node->user_data);
  FastConv1D(tflite::ConvParamsQuantized(p,d), in->dims->data[2],in->dims->data[3],
    w->dims->data[2],out->dims->data[2],out->dims->data[3],in->data.int8,w->data.int8,
    bias ? bias->data.i32 : nullptr,d.per_channel_output_multiplier,d.per_channel_output_shift,out->data.int8);
  return kTfLiteOk;
}
}
TFLMRegistration RegisterFastConv1D() {
  return tflite::micro::RegisterOp(tflite::ConvInit,Prepare,Eval);
}
}
