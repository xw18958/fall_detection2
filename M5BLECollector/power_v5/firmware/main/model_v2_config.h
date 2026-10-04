#pragma once
// Experimental V5 PTQ: user-requested device test; full validation gates FAILED.
namespace fall_v2 {
constexpr int kTimesteps=90, kChannels=6, kSampleHz=30;
constexpr long long kInferencePeriodUs=750000;
constexpr float kThreshold=6.313204765320e-01f;
constexpr float kGravity=9.806650161743e+00f;
constexpr float kDegreesToRadians=1.745329238474e-02f;
constexpr float kInputScale=3.469818830490e-01f;
constexpr float kOutputScale=6.329617649317e-02f;
constexpr int kInputZero=9, kOutputZero=-5;
constexpr float kMean[6]={2.000569552183e-01f,6.971700668335e+00f,-1.755554318428e+00f,2.056334726512e-02f,-7.300961017609e-02f,-9.963835589588e-03f};
constexpr float kSigma[6]={5.583704948425e+00f,6.432809352875e+00f,6.477667331696e+00f,2.639466285706e+00f,4.051290512085e+00f,2.314336061478e+00f};
constexpr char kCheckpointSha256[]="69df2aed93411a7ec78afb6b726c558196182870595e4fb98028d5de533dfe55";
}
