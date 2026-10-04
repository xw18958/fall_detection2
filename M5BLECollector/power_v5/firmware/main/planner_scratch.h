#pragma once
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"

namespace fall_power {
// Same greedy placement algorithm; only startup planning storage changes.
// The tensor arena remains internal RAM throughout inference.
class ScratchPlanner final : public tflite::GreedyMemoryPlanner {
 public:
  void Bind(unsigned char* scratch,int bytes) {scratch_=scratch;bytes_=bytes;}
  TfLiteStatus Init(unsigned char*,int) override {
    if(!scratch_ || bytes_<=0) return kTfLiteError;
    return tflite::GreedyMemoryPlanner::Init(scratch_,bytes_);
  }
 private:
  unsigned char* scratch_=nullptr;
  int bytes_=0;
  TF_LITE_REMOVE_VIRTUAL_DELETE
};
constexpr int kPlannerScratchBytes=32*1024;
}
