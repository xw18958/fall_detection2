#pragma once
// Locked QAT export; training-only normalization, joint-validation threshold.
namespace fall_v2 {
constexpr int kTimesteps=90, kChannels=6, kSampleHz=30;
constexpr long long kInferencePeriodUs=750000;
constexpr float kThreshold=2.726432085037e-01f;
constexpr float kGravity=9.806650161743e+00f;
constexpr float kDegreesToRadians=1.745329238474e-02f;
constexpr float kInputScale=9.824229776859e-02f;
constexpr float kOutputScale=7.270766794682e-02f;
constexpr int kInputZero=-1, kOutputZero=-5;
constexpr float kMean[6]={1.538258492947e-01f,6.978156566620e+00f,-1.832675099373e+00f,2.068609185517e-02f,-8.936759084463e-02f,-6.220293696970e-03f};
constexpr float kSigma[6]={5.668211460114e+00f,6.612070560455e+00f,6.552240848541e+00f,2.700584888458e+00f,4.197855949402e+00f,2.409795999527e+00f};
constexpr char kCheckpointSha256[]="d12ba0f0915859ae6ae4056b21d8ab2730142dfc5b3299cbaaabe71c98529866";
}
