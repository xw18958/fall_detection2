#pragma once
// Generated from the locked V2 export. Collection scale is inferred, not documented.
namespace fall_v2 {
constexpr int kTimesteps = 90;
constexpr int kChannels = 6;
constexpr int kSampleHz = 30;
constexpr long long kInferencePeriodUs = 750000;
constexpr float kThreshold = 4.1500000000e-01f;
constexpr float kMean[6] = {3.4157230577e+03f, -7.5216521183e+03f, 1.5700171107e+01f, -5.2228506947e+00f, -6.2451702187e+00f, -1.2628369227e+01f};
constexpr float kSigma[6] = {8.0534263770e+03f, 1.0512396740e+04f, 6.8477520967e+03f, 4.8062610525e+02f, 6.0144056276e+02f, 4.5518131122e+02f};
constexpr float kTrainingAccelCountsPerG = 1.6384000000e+04f;
constexpr float kTrainingGyroCountsPerDps = 1.6400000000e+01f;
constexpr char kCheckpointSha256[] = "477aefa99826dfaa1bc23117a78c812247ad4135886caff7ffcab92c57c23138";
}  // namespace fall_v2
