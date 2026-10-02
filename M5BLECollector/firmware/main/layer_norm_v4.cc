#include "layer_norm_v4.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"
namespace fall_tflm {
namespace {
struct Data {int channels, count, zero; float input_scale, output_scale;};
void* Init(TfLiteContext* c,const char*,size_t) {return c->AllocatePersistentBuffer(c,sizeof(Data));}
TfLiteStatus Prepare(TfLiteContext* c,TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(c,tflite::NumInputs(node),1);
  TF_LITE_ENSURE_EQ(c,tflite::NumOutputs(node),1);
  auto* mc=tflite::GetMicroContext(c);
  auto* x=mc->AllocateTempInputTensor(node,0);
  auto* y=mc->AllocateTempOutputTensor(node,0);
  TF_LITE_ENSURE(c,x && y && x->type==kTfLiteInt8 && y->type==kTfLiteInt8);
  TF_LITE_ENSURE_EQ(c,x->dims->size,y->dims->size);
  for(int i=0;i<x->dims->size;++i) TF_LITE_ENSURE_EQ(c,x->dims->data[i],y->dims->data[i]);
  auto* d=static_cast<Data*>(node->user_data);
  d->channels=x->dims->data[x->dims->size-1];
  TF_LITE_ENSURE(c,d->channels==24 || d->channels==48);
  d->count=tflite::ElementCount(*x->dims);
  d->input_scale=x->params.scale; d->output_scale=y->params.scale; d->zero=y->params.zero_point;
  TF_LITE_ENSURE(c,d->input_scale>0 && d->output_scale>0);
  mc->DeallocateTempTfLiteTensor(x); mc->DeallocateTempTfLiteTensor(y);
  return kTfLiteOk;
}
TfLiteStatus Eval(TfLiteContext* c,TfLiteNode* node) {
  auto* x=tflite::micro::GetEvalInput(c,node,0); auto* y=tflite::micro::GetEvalOutput(c,node,0);
  const auto* d=static_cast<const Data*>(node->user_data);
  const auto* src=tflite::micro::GetTensorData<int8_t>(x); auto* dst=tflite::micro::GetTensorData<int8_t>(y);
  for(int i=0;i<d->count;i+=d->channels) LayerNormRow(src+i,dst+i,d->channels,d->input_scale,d->output_scale,d->zero);
  return kTfLiteOk;
}
}
TFLMRegistration RegisterLayerNormV4() {return tflite::micro::RegisterOp(Init,Prepare,Eval);}
}
