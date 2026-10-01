#include <algorithm>
#include <fstream>
#include <iterator>
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <random>
#include "fast_conv1d.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "fall_op_resolver.h"
#include "model_v2_replay.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
int KernelChecks() {
 std::mt19937 rng(42);
 int cases=0;
 for (int width: {3,30,90}) for (int channels: {1,3,24})
 for (int outputs: {1,5,24}) for (int kw: {1,5})
 for (int dilation: {1,2,4}) for (int stride: {1,2})
 for (int pad: {0,(kw-1)*dilation/2}) {
  int outw=(width+2*pad-(kw-1)*dilation-1)/stride+1;
  if (width+2*pad<(kw-1)*dilation+1 || outw<1) continue;
  tflite::ConvParams p{};
  p.stride_width=stride; p.stride_height=1;
  p.dilation_width_factor=dilation; p.dilation_height_factor=1;
  p.padding_values.width=pad; p.input_offset=int(rng()%257)-128;
  p.output_offset=int(rng()%41)-20;
  p.quantized_activation_min=cases%2 ? -30:-128;
  p.quantized_activation_max=cases%2 ? 90:127;
  std::vector<int8_t> in(width*channels),w(outputs*kw*channels),a(outw*outputs),b(a.size());
  std::vector<int32_t> bias(outputs),mult(outputs),shift(outputs);
  for (auto& q:in) q=int(rng()%256)-128;
  for (auto& q:w) q=int(rng()%255)-127;
  for (int c=0;c<outputs;++c) {bias[c]=int(rng()%2001)-1000;mult[c]=1073741824+int(rng()%1073741824);shift[c]=-3-int(rng()%6);}
  const int32_t* bp=cases%3 ? bias.data():nullptr;
  const int32_t id[4]={1,1,width,channels},wd[4]={outputs,1,kw,channels},bd[1]={outputs},od[4]={1,1,outw,outputs};
  tflite::RuntimeShape is(4,id),ws(4,wd),bs(1,bd),os(4,od);
  tflite::reference_integer_ops::ConvPerChannel(p,mult.data(),shift.data(),is,in.data(),ws,w.data(),bs,bp,os,a.data());
  fall_tflm::FastConv1D(p,width,channels,kw,outw,outputs,in.data(),w.data(),bp,mult.data(),shift.data(),b.data());
  if (a!=b) {std::cerr<<"KERNEL_MISMATCH case="<<cases<<"\n";return 8;}
  ++cases;
 }
 std::cout<<"BIT_EXACT_KERNEL_CASES_PASS="<<cases<<"\n";
 return 0;
}
int main(int argc, char** argv) {
 if (argc==2 && std::strcmp(argv[1],"--kernel-test")==0) return KernelChecks();
 if (argc != 2 && argc != 4) return 2;
 std::ifstream file(argv[1], std::ios::binary);
 std::vector<unsigned char> model_bytes((std::istreambuf_iterator<char>(file)), {});
 if (model_bytes.empty()) return 3;
 alignas(16) static unsigned char arena[256*1024];
 fall_tflm::FallOpResolver resolver;
 const auto* model = tflite::GetModel(model_bytes.data());
 tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
 if (interpreter.AllocateTensors() != kTfLiteOk) return 4;
 std::cout << "arena_used_bytes=" << interpreter.arena_used_bytes() << "\n";
 if (argc == 4) {
  std::ifstream inputs(argv[2], std::ios::binary);
  std::ofstream outputs(argv[3], std::ios::binary);
  int windows=0;
  while (inputs.read(reinterpret_cast<char*>(interpreter.input(0)->data.int8),540)) {
   if (interpreter.Invoke() != kTfLiteOk) return 5;
   outputs.write(reinterpret_cast<char*>(interpreter.output(0)->data.int8),2);
   ++windows;
  }
  std::cout << "TFLM_WINDOWS_PASS=" << windows << "\n";
  return windows>0 ? 0 : 7;
 }
 for (int i=0;i<2;++i) {
  std::memcpy(interpreter.input(0)->data.int8, fall_v2::kReplayInputs[i], 540);
  if (interpreter.Invoke() != kTfLiteOk) return 5;
  for (int c=0;c<2;++c) {
   int actual=interpreter.output(0)->data.int8[c];
   int expected=fall_v2::kReplayOutputs[i][c];
   std::cout << "replay=" << i << " channel=" << c << " actual=" << actual << " expected=" << expected << "\n";
   if (std::abs(actual-expected)>4) return 6;
  }
 }
 std::cout << "TFLM_REPLAY_PASS\n";
 return 0;
}
