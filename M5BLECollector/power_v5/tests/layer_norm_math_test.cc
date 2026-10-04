#include "../firmware/main/layer_norm_math.h"
#include <random>
#include <cstdio>
#include <cassert>
int main() {
 std::mt19937 random(42); int total=0, differed=0;
 for(int n: {24,48}) for(float scale:{.0001f,.01f,.1f,1.f,10.f}) for(float out:{.005f,.05f,.5f}) for(int k=0;k<1002;++k) {
  int8_t x[48],y[48]; int zero=int(random()%256)-128;
  double mean=0;
  for(int i=0;i<n;++i) {x[i]= k==0?127:k==1?-128:int(random()%256)-128;mean+=(int(x[i])-zero)*double(scale);}
  mean/=n; double variance=0;
  for(int i=0;i<n;++i) {double centered=(int(x[i])-zero)*double(scale)-mean;variance+=centered*centered;}
  variance/=n;
  fall_tflm::LayerNormRow(x,y,n,scale,out,zero);
  for(int i=0;i<n;++i) {
   double value=((int(x[i])-zero)*double(scale)-mean)/std::sqrt(variance+1e-5);
   long expected=std::max(-128L,std::min(127L,std::lround(value/out)+zero));
   assert(std::abs(int(y[i])-expected)<=1); differed+=int(y[i])!=expected;
  }
  ++total;
 }
 printf("LAYERNORM_PASS rows=%d max_error<=1LSB differing_elements=%d\n",total,differed);
}
