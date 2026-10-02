#!/usr/bin/env python3
"""Render compact V4 classification/runtime results without publishing replay inputs."""
import argparse,csv,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();r=a.results
 load=lambda n:json.loads((r/n).read_text())
 tests=load('test_runtime_comparison.json');exp=load('export_lock.json');manifest=load('package_manifest.json');val=load('validation_report.json');retr=load('retrieval_reference.json')
 order=['own','m5_hard_negatives','CGU_BES','Cogent','SFU_IMU','UCI_SimulatedFalls','PAMAP2','pooled']
 lines=['# V4 seed42 INT8 collector development candidate','',
 'Status: export, exhaustive evaluation, native Micro/ESP-NN checks and combined firmware/Mac builds complete. **Not installed; performance acceptance is incomplete.** No USB serial device was present. Original checkpoint, experiment, dirty Mac checkout and recordings were not modified. No retraining or additional seeds were run.','',
 'The same locked model and float threshold were fixed before final testing. No calibration/model/threshold candidate was selected using the held-out test. INT8 has measurable degradation; it is not a performance-preserving acceptance result.','',
 '## Held-out classification: 3-second windows','',
 'All 66,539 complete windows use the inherited holdouts, cumulative 0.25-second grid plus final complete window, source-specific stored normalization and no gap crossing or caps. Windows overlap and are correlated. Fall is class 1; predicted fall means two-class softmax probability >= 0.2602156102657318.','',
 'Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN), specificity=TN/(TN+FP), negative-window FPR=FP/(FP+TN). Percentages below use those denominators; negative-only fall metrics are N/A. Per-source rows precede pooled results because pooled accuracy is dominated by negative windows.','',
 '| Source | Runtime | Windows | TP | FN | FP | TN | Precision % | Recall % | F1 % | Specificity % | Negative-window FPR % |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 table=[]
 for d in order:
  for name,label in [('float','PyTorch float'),('desktop_int8','Desktop INT8'),('micro_reference','Micro / ESP-NN INT8')]:
   m=tests['domains'][d][name];table.append({'source':d,'runtime':label,**m})
   fmt=lambda x:'N/A' if x is None else f'{x*100:.5f}'
   values=[d,label,*[str(m[k]) for k in ['windows','tp','fn','fp','tn']],*[fmt(m[k]) for k in ['precision','recall','f1','specificity','negative_window_fpr']]]
   lines.append('| '+' | '.join(values)+' |')
 lines+=['','Micro relative to new float: Private TP158→157, FN4→5, FP19→35, TN15435→15419; recall 97.53086%→96.91358% (−0.61728 percentage points), F1 93.21534%→88.70056% (−4.51477 pp), negative-window FPR 0.12295%→0.22648%. M5 FP67→69 of 1,082 negative windows (6.19224%→6.37708%). Pooled TP3376→3373, FN44→47, FP250→277, TN62869→62842. Desktop is not a substitute for these Micro results.','',
 '## Export and calibration','',
 f'Checkpoint SHA-256: `{exp["checkpoint_sha256"]}`. Model SHA-256: `{exp["model_sha256"]}`. Model bytes: {exp["model_bytes"]:,}. Split fingerprint: `{exp["split_fingerprint"]}`. Normalization SHA-256: `{exp["normalization_sha256"]}`.',
 '',f'PyTorch→TF maximum logit error {exp["pytorch_tf_logit_error"]["max_absolute"]:.9g}; PyTorch→float TFLite {exp["pytorch_float_tflite_logit_error"]["max_absolute"]:.9g}. Frozen BatchNorm folded exactly; full 90-sample and all three 30-sample branches retained. Encoder 24 channels, three blocks; no architectural change.',
 '',f'Calibration uses {sum(exp["calibration"]["source_windows"].values())} training-only windows from {exp["calibration"]["unique_recordings"]} recordings, seed 42. Rounding the source quotas yields 2,049 windows from nominal 2,048. Own and M5 each 512; each public source 205. Fall-capable sources balance classes and groups. M5 includes each training session and per-channel absolute-peak windows, with full SI range and no count clipping. Calibration manifest stays outside Git; its hash and source/class coverage are in export_lock.json.',
 '',f'INT8 input scale/zero point: {exp["inventory"]["input_scale_zero"]}; logits output: {exp["inventory"]["output_scale_zero"]}. Per-channel convolution weights (72 convolution nodes, including single-channel score heads). Tensor inventory: {exp["inventory"]["tensor_types"]}; no float tensors or fallback. Operator counts: `{json.dumps(exp["inventory"]["operators"],sort_keys=True)}`.',
 '', 'No validation/test input values saturated the INT8 input range. Per-source logit/probability mean, p99 and worst errors and saturation counts are in test_report.json. Calibration ranges and p99 magnitude by source/channel are in calibration_coverage.json. The common activation input step is approximately 0.347 normalized units; training-source extrema widen that range. Integer attention/layer-normalization and accumulated kernel rounding remain precision-sensitive. These observations diagnose limitations; no new calibration variant was chosen on test results.',
 '', '## Validation and runtime diagnostics','',
 '61,686 validation windows were evaluated. M5 remained FP0/1082 in float, desktop and Micro. The unchanged-threshold desktop seven-source group-macro weighted error is 0.0017077905360709201 versus float 0.0013422465811370773. These are not pooled window error rates.',
 '', 'Inherited guard failures for desktop INT8: SFU negative-recording specificity 0.8589743589743593 below 0.8646153846153848; UCI 0.9749520521672421 below 0.98. The earlier unconstrained validation diagnostic optimum 0.46371036767959595 was never adopted or tested. The deployed-candidate threshold remains unchanged; no post-test adjustment was made.',
 '', 'The optimized convolution matches Micro reference bit-for-bit on 594 randomized shape/dilation/stride/bias/activation cases. Tensor tracing first found a desktop-versus-Micro difference at operator 9 (CONV_2D): two of 720 values differ by one LSB; differences propagate through later operations. A single-rounding diagnostic did not eliminate the discrepancy; firmware/custom kernels remain unchanged. This is not evidence of complete desktop/Micro numerical parity.',
 '', 'Portable Micro and the ESP32 firmware’s generic ESP-NN FC/softmax paths produced identical output bytes on every validation and test window (128,225 windows). Native ESP-NN arena use is 101,600 bytes; portable reference 101,376. Neither is a measured ESP32 arena/free-heap result. Native bulk elapsed time is host timing, never a device-speed claim.',
 '', 'Eleven startup replay cases cover training calibration examples, zero counts, signed range extremes, alternating extremes and four joint-validation windows closest to threshold. Expectations are pinned to the verified Micro/ESP-NN path; desktop differences are retained in replay_manifest.json. The prior 4-LSB startup tolerance remains unchanged. Five count-vector goldens and all 393,216 signed-count/channel combinations verify raw counts→float32 SI→stored M5 normalization without a second epsilon→half-away rounding→saturated INT8 bytes. Native replay verifies logits; runtime_comparison.json verifies probabilities and decisions.',
 '', '## Collector preservation and hardware status','',
 'All ten Python guard tests, sanitizer-enabled C++ recording-buffer/wire and battery/display tests, Swift protocol/journal/recovery/deduplication/completed-count tests, old/new trusted-identity tests, and the receiver build passed. Collector SensorTask, ReadImuCounts, range/filter initialization, BLE backend, recording buffer, Recorder/Protocol, button/wake/sleep regions, battery UI and recovery OTA source remain pinned to their existing hashes. Only documented detector/preprocessing/model-package, receiver identity and additive test pins changed. Generic upload/erase guards remain enabled.',
 '', 'Both boot modes retain startup model initialization/replay; COLLECT frees interpreter/arena before BLE and PSRAM recording-buffer allocation. No inference or SI conversion was added to COLLECT. Saved CSV schema 3, original signed counts, KEEP/DISCARD, durable journal ACK, retransmission/deduplication and count-agreement completion are unchanged. Existing old-model trust and recovery model/boot/session identity checks remain strict.',
 '', 'Generated sdkconfig verified: ESP32, 240 MHz, performance optimization, 8 MB flash, PSRAM enabled, CONFIG_BT_NIMBLE_HS_FLOW_CTRL disabled. Native arena fits the nominal 112 KiB internal-arena size, but actual device internal allocation and PSRAM fallback need physical confirmation.',
 '', 'The previous IDF embedding ALIGN argument was ignored, leaving the model unaligned. Explicit .balign 16 assembly now embeds the exact model once, without a second RAM copy. ELF alignment/extent and application size checks pass; config, replay, model identities and actual input/output scales are verified at startup, with rollback on startup failure.',
 '', f'Application-only firmware bytes: {manifest["firmware_bytes"]:,}, below 0x3d0000. Firmware SHA-256: `{manifest["firmware_sha256"]}`. Embedded model start `{manifest["embedded_model_start"]}`; extent {exp["model_bytes"]:,} bytes. Bootloader/partitions/NVS were not written.',
 '', 'Pending physical acceptance: identify installed firmware/transport and active OTA slot; finish/save any acquisition and establish pending=0; back up installed image/boot state/settings; application-only OTA to the inactive slot; measure median/p95/p99/worst invoke, input-to-probability and complete-path times plus arena/free-memory; verify 30 Hz raw capture, KEEP/transfer/COMPLETE count agreement, DISCARD/reconnect/deduplication, mode exits/buttons/wake/battery and rollback. The firmware retains 750 ms cadence until hardware measurements justify a change. Neither <300 ms inference nor a 250 ms complete path is claimed.',
 '', '## Retrieval and zero-shot scope','',
 '| Check | Result |', '|---|---:|',
 f'| Float binary-class retrieval P@1 (51 test query recordings, 245 gallery recordings) | {retr["float_checkpoint_retrieval"]["precision_at_1"]} |',
 f'| Float hit rate@5 | {retr["float_checkpoint_retrieval"]["hit_rate_at_5"]} |',
 f'| Float mAP | {retr["float_checkpoint_retrieval"]["mAP"]} |',
 '| INT8 retrieval | N/A: classifier graph exports logits, not retrieval embeddings |',
 '| Zero-shot | N/A: all seven sources contribute training data |','',
 '## Reproduction and locations','',
 'See REPRODUCE.md. Exact private deployment package, replays, firmware ELF/bin and build logs are delivered outside Git. Compact reports/source are on codex/v4-int8-collector. Server export: `/raid1/xwan0900/v4_int8_seed42_20261003`; server source: `/raid1/xwan0900/fall_detection2_v4_quantization`; Mac isolated checkout: `/private/tmp/fall_detection2-v4-quantization`; durable delivery: `/Users/henrywang/Documents/fall_detection2/deployment_artifacts/v4_seed42_20261003`. Raw recording folders and hidden journals were untouched.']
 (r/'REPORT.md').write_text('\n'.join(lines)+'\n')
 with (r/'classification_windows.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
if __name__=='__main__':main()
