#pragma once
#include "power_policy.h"
namespace fall_trigger {
bool Init(const char* full_model_sha256);
bool Evaluate(const fall_power::Window& window, float* probability);
bool Qualified();
float Threshold();
unsigned ArenaBytes();
uint64_t StartupWorstUs();
}  // namespace fall_trigger
