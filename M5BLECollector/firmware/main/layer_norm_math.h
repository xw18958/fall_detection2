#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
namespace fall_tflm {
// INT8 tensor interface, exact integer statistics, float32 inverse square root.
// The zero point cancels in centering. Epsilon is in original real units.
inline void LayerNormRow(const int8_t* x, int8_t* y, int n,
                         float in_scale, float out_scale, int out_zero) {
  int64_t sum=0, squares=0;
  for(int i=0;i<n;++i) { sum+=x[i]; squares+=int(x[i])*int(x[i]); }
  const int64_t variance=squares*n-sum*sum;
  const float denom=std::sqrt(float(variance)+1e-5f*n*n/(in_scale*in_scale));
  for(int i=0;i<n;++i) {
    const float normalized=float(int64_t(x[i])*n-sum)/denom;
    const long q=std::lround(normalized/out_scale)+out_zero;
    y[i]=static_cast<int8_t>(std::max(-128L,std::min(127L,q)));
  }
}
}
