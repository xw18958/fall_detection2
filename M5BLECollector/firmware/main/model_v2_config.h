#pragma once
// Exact V4 checkpoint M5 SI normalization; unchanged float threshold.
namespace fall_v2 {
constexpr int kTimesteps=90, kChannels=6, kSampleHz=30;
// Retain 750 ms until physical latency and acquisition are measured.
constexpr long long kInferencePeriodUs=750000;
constexpr float kThreshold=2.602156102657e-01f;
constexpr float kGravity=9.806650161743e+00f;
constexpr float kDegreesToRadians=1.745329238474e-02f;
constexpr float kInputScale=3.469818830490e-01f;
constexpr float kOutputScale=7.270710915327e-02f;
constexpr int kInputZero=9, kOutputZero=-5;
constexpr float kMean[6]={1.538258492947e-01f,6.978156566620e+00f,-1.832675099373e+00f,2.068609185517e-02f,-8.936759084463e-02f,-6.220293696970e-03f};
constexpr float kSigma[6]={5.668211460114e+00f,6.612070560455e+00f,6.552240848541e+00f,2.700584888458e+00f,4.197855949402e+00f,2.409795999527e+00f};
constexpr char kCheckpointSha256[]="3c92a68f6e1440a83880665d6634b01e7cb5724374a7e3f7a3eb76892e210b2e";
}
