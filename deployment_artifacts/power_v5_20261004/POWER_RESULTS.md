# V5 whole-system power optimization — 4 October 2026

## Power measurement build and subsequent changes

The measured firmware **1.4.3-v5-power-dev** was installed in application slot 1, OTA sequence 10, in DETECT mode. The subsequent `1.4.4-v5-power-buzzer` update is documented in `../power_v5_buzzer_20261004/BUZZER_RESULTS.md`. Startup model replay and acquisition health passed; the application was marked valid. **The trained trigger, cascade gating and parallel convolution are disabled. The unchanged full V5 model still runs at the explicit 750 ms fallback cadence.** Trigger-only normal monitoring remains unqualified and disabled. The original 250 ms target has been relaxed as an immediate requirement by the user; it remains a future architectural target.

Firmware SHA256: `b5b002d7686b031e3408a68dc4edaa19338a0a01cb23aaf36817c92e02ac236f`. Size: 1,506,768 bytes. See `FINAL_DEPLOYMENT.json` for the final device measurements and deployment state.

The actual deployed detector/collector source is `M5BLECollector/power_v5`. The older standalone `M5StickCPlus2FallDetection` has a different model; changing its rate constants would not reproduce V5. Training and comparative evaluation are in `FallDetection30HzTraining`. The source and compact reports are being published from the latest GitHub main; see the subsequent buzzer report for the publication scope.

## Implemented and deployed

| Area | Change and verification |
|---|---|
| Sensor/model contract | 30 Hz, six axes, 90 × 6 / three-second history. V5 INT8 weights, SI units, normalization, decision threshold, labels and original splits preserved. |
| Acquisition | Rational timer deadlines, continuous history and explicit gap detection/recovery. Final two-minute DETECT capture had zero sampling gaps. |
| CPU | 80 MHz normal active operation; temporary 240 MHz full-model execution; compatible tickless idle/light sleep between work. |
| Memory | Supported split allocator: 64 KiB internal activation arena plus 128 KiB PSRAM metadata. 80,204 bytes used. 61,155 validation windows retained byte-identical model outputs. |
| Display | Normal DETECT display expires after about 15 seconds; buttons/alerts wake it. Sensing and detection continue while the display is off. |
| Wi-Fi/OTA | Wi-Fi off in normal DETECT; existing HTTP OTA is an explicit bounded session. Offline startup is healthy. Rollback retained. Live OTA network start/stop remains unverified. |
| Logging/buzzer | Production logging reduced to periodic telemetry and meaningful alerts/faults; debug option retained. Hardware buzzer timing avoids the former busy loop. |
| Collector | Review/KEEP/discard and BLE ACK workflow preserved. An isolated KEEP test saved/acknowledged 662 samples; existing recordings were untouched. |
| USB deployment | Backup and inactive-slot write/readback verification before selecting the new application. Bootloader, partition layout and unrelated NVS state protected. |

## Measured device performance

| Profile | Full TCN median | p95 | Worst | Status |
|---|---:|---:|---:|---|
| Prior actual V5 | 478.318 ms | — | — | Reference |
| Final serial/internal-activation image | 401.864 ms | 402.880 ms | 403.179 ms | Installed; trigger off |
| Exact dual-core convolution | 272.716 ms | 273.534 ms | 275.489 ms | Benchmark only; disabled |

Final input-to-probability median was 402.397 ms (worst 403.738 ms), over 128 invocations. The installed profile is about 16% faster than the prior V5. Neither profile meets the 250 ms full-inference budget, before accounting for trigger and scheduling overhead. The dual-core profile had higher aggregate CPU active fraction (~0.302 versus ~0.272); its lower latency does not establish lower power.

Final telemetry showed zero sampling gaps and Wi-Fi on-time of zero. Display cumulative on-time stopped at 15.250 seconds. Free internal memory was 137,651 bytes and free PSRAM 1,963,584 bytes. CPU duty is telemetry aggregated across two cores, not current consumption. No external current measurement or battery-life claim was made.

## Trigger architecture and training completed

- Full 90 × 6 trained-trigger input, rational 250 ms schedule (7/8 newly acquired samples), immutable copied W0 immediately queued for the full model. Following awake windows use the same 250 ms grid.
- One-second quiet timeout and bounded eight-second maintenance bursts. Persistent suspicion is one episode; maintenance rearms are tracked separately. Trigger/queue/deadline faults disable gating and explicitly use the 750 ms fallback.
- Build/runtime guards reject unqualified bundles, wrong full-model/config hashes, wrong shape/rate and unmet timing budgets. Current research bundles remain unqualified.
- Seven matched candidates: depthwise-separable CNN widths 8/16/24 with mean or mean+max pooling, plus tiny MLP16. Trained on the remote GPU; full-integer INT8 PTQ export and native Micro evaluation completed.
- Multiple controlled loss/threshold experiments were completed. The latest robust-event experiment selected `cnn24_mean_max` on validation before auditing test. No test-based winner switching was performed within that experiment. Prior test exposure is explicitly recorded, so this remains exploratory.
- The earlier `cnn16_mean` shadow hardware benchmark measured worst startup trigger latency 14.419 ms at 80 MHz and 3,500 bytes of used tensor arena (24 KiB reserved). Latest width-24 hardware latency is not measured.
- An early shadow-run cadence bug stretched the fallback under wake-time jitter. It was fixed using nominal deadlines; the host one-hour check passes exactly 4,800 fallback evaluations. That corrected shadow profile still needs a repeat hardware run.

## Latest trigger qualification results

| Split | Trigger fall-recording recall | Lost/late V5 timing references | Normal awake fraction | False wakes/hour |
|---|---:|---:|---:|---:|
| Validation | 427/427 (100.00%) | 0/434 | 4.68% | 92.44 |
| Test | 391/391 (100.00%) | 2/400 | 3.93% | 76.66 |

The two test timing failures occur in Cogent and own data. A trigger can wake somewhere in every fall recording yet wake too late for a particular full-model detection window. That is why 100% recording recall does not qualify it. Timing references are the first positive of each frozen-V5 positive run in a fall-labelled recording, not annotated clinical fall onset.

False wakes/hour use only continuous negative segments at least 30 seconds long, excluding the first three-second warmup. Test exposure is only 2.426 hours in total. Awake fractions describe simulated scheduling time, not measured energy savings or CPU duty. Wake events do not themselves raise fall alarms.

### Overall test classification

Window/container-label metrics:

| Classifier | Recall | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|
| Unchanged full V5 | 98.05% | 99.37% | 89.51% | 93.59% |
| Latest gated simulation | 97.62% | 99.81% | 96.58% | 97.09% |
| Trigger alone (wake classifier) | 97.53% | 97.21% | 65.75% | 78.55% |

Recording-level classification (positive if the classifier fires somewhere in the labelled recording):

| Classifier | Recall | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|
| Unchanged full V5 | 98.98% | 96.17% | 94.62% | 96.75% |
| Latest gated simulation | 98.98% | 97.39% | 96.27% | 97.60% |
| Trigger alone | 100.00% | 61.57% | 63.89% | 77.97% |

The simulation improves overall specificity/precision/F1 but reduces window recall and loses two timing references. It was rejected for deployment. The unchanged full model and simulated cascade both detect 387/391 positive test recordings; the trigger wakes on 391/391, so the trigger does not repair the full model’s existing four recording misses.

### Complete per-source test window results

| Source | Classifier | Recall | Specificity | Precision | F1 |
|---|---|---:|---:|---:|---:|
| CGU_BES | Full V5 | 100.00% | 98.97% | 88.89% | 94.12% |
| CGU_BES | Gated simulation | 100.00% | 99.54% | 94.74% | 97.30% |
| Cogent | Full V5 | 98.10% | 99.91% | 98.56% | 98.33% |
| Cogent | Gated simulation | 96.83% | 99.91% | 98.55% | 97.68% |
| PAMAP2 | Full V5 | N/A | 98.56% | 0.00% | 0.00% |
| PAMAP2 | Gated simulation | N/A | 99.95% | 0.00% | 0.00% |
| SFU_IMU | Full V5 | 94.85% | 99.80% | 97.77% | 96.29% |
| SFU_IMU | Gated simulation | 94.85% | 99.83% | 98.04% | 96.42% |
| UCI_SimulatedFalls | Full V5 | 98.72% | 99.99% | 99.91% | 99.31% |
| UCI_SimulatedFalls | Gated simulation | 98.72% | 100.00% | 100.00% | 99.36% |
| M5 falls + hard negatives | Full V5 | 89.47% | 91.12% | 15.04% | 25.76% |
| M5 falls + hard negatives | Gated simulation | 89.47% | 94.08% | 20.99% | 34.00% |
| own | Full V5 | 96.30% | 99.81% | 84.32% | 89.91% |
| own | Gated simulation | 91.98% | 99.82% | 84.66% | 88.17% |

PAMAP2 contains only negatives in these folds; fall recall is unavailable there. Metrics use each source’s original frozen normalization. They do not establish performance under a single universal deployment normalization.

### M5-specific problem

The latest trigger wakes on all three held-out M5 fall recordings. However, the unchanged full model detects only two of those three. On the one held-out continuous M5 negative recording (~4.50 minutes usable exposure), the trigger simulation keeps the full model awake **35.25%** of the time and produces **533.27 wake-ups/hour** when extrapolated. These are much worse than the overall 3.93% / 76.66 per hour. The short exposure makes that hourly estimate uncertain; it is a wake count, not an alarm count.

M5 full-model test window recall/specificity/precision/F1 are 89.47% / 91.12% / 15.04% / 25.76%. Gating changes them to 89.47% / 94.08% / 20.99% / 34.00%; it does not resolve the missed M5 fall recording. The denominator contains only 19 positive windows and 1,081 negative windows, and windows overlap. Recording and window figures describe different questions. Overall averages therefore obscure the M5 weakness.

## Data contract and limitations

- Original V5 split: 4,641 training / 1,052 validation / 966 test recordings. Own data, CGU_BES, Cogent, PAMAP2, SFU_IMU, UCI_SimulatedFalls and the historical `m5_hard_negatives` domain are preserved.
- M5 combined data: 19 falls (13/3/3 train/validation/test) plus seven negatives (5/1/1). Requested existing five-second crop method, units, labels, roles and source hashes retained. The historical domain name now includes both falls and negatives.
- Participant and sensor placement are not established; M5 falls share one boot. Three test falls cannot establish independent participant/placement generalization.
- Recordings have labels, but no independently annotated clinical onset times. Test folds have been examined in prior work. Independent untouched validation remains necessary for a commercial claim.
- Retrieval and zero-shot metrics are not applicable to this binary sensor classifier.

## Smaller full-model benchmark completed

Matched exploratory five epochs × 6,400 draws, frozen source normalization/splits, identical seed/sampler budget, inherited BN statistics and original threshold. Candidates must preserve recall, specificity and F1 on each source before INT8/hardware qualification.

| Variant | Classifier MACs | Test recall | Test specificity | Test precision | Test F1 | Preservation |
|---|---:|---:|---:|---:|---:|---|
| matched_control | 6,466,464 | 97.50% | 99.79% | 96.16% | 96.82% | fail |
| width16 | 2,904,128 | 95.61% | 97.11% | 64.50% | 77.03% | fail |
| blocks2 | 4,392,864 | 94.97% | 98.95% | 83.19% | 88.69% | fail |
| mean_attention | 6,355,056 | 94.62% | 99.10% | 85.18% | 89.65% | fail |

No candidate, including the matched fine-tuning control, passed every per-source preservation gate. None replaced the deployed model. Reduced models were not advanced to INT8/hardware qualification. The attention simplification saves only about 1.7% classifier MACs; the convolutional backbone remains dominant. A short warm-start failure does not prove these architectures cannot work with a longer, redesigned controlled experiment.

## Verification completed

- Final host smoke: 18 Python guards; C++ recording/UI/battery/layernorm and scheduler checks; Swift protocol/journal/trust checks.
- 594 kernel edge cases and 61,155 validation outputs bit-identical for split memory and exact parallel row arithmetic. Exhaustive input quantization check: 393,216 cases, zero mismatches.
- Fast real-data/GPU/export/native smoke tests before substantial training; all seven robust candidates validated through the pipeline.
- Hardware model replay/startup health, rollback after a rejected allocation experiment, steady 30 Hz capture, display expiry, Wi-Fi-off monitoring and isolated BLE KEEP/ACK verified.
- Detailed evidence is in `smoke_system_final.log`, `FINAL_DEPLOYMENT.json`, `COMPACT_TRIGGER_SOURCES.json`, hardware reports, source snapshot and USB proofs. Private raw logs, models, backups, replay inputs and recordings are excluded from publication.

## Not completed / blocking problems

1. **Future 250 ms full-model target (not a current-stage acceptance requirement):** installed path is ~402 ms; exact dual-core path is ~273 ms before trigger cost. Four full evaluations per second would overload the current execution path.
2. **Qualified trigger:** two held-out timing references are lost; M5 normal activity wakes it too often. Production gating remains off. Latest width-24 trigger latency and corrected shadow scheduling have not been retested on hardware.
3. **Cheaper full model preserving each source:** explored candidates failed. Longer training/distillation and further exact kernel optimization remain future experiments.
4. **Matched trigger PTQ versus QAT study:** not completed. Trigger exports used full-integer PTQ. There is no evidence here that QAT would remove the timing/generalization failures; no method is labelled universally best.
5. **Random Forest baseline:** not implemented/evaluated; the tiny MLP comparison was completed.
6. **Actual energy/battery measurement:** instrumentation exists, but there is no external current trace or matched before/after battery test. Clock/screen/Wi-Fi changes are deployed; quantified battery savings are unknown.
7. **Excluded testing:** live OTA network exercises and prolonged reliability tests will not be performed, as requested by the user. Rollback behavior was already exercised.
8. **Commercial/generalization qualification:** independent participants, placements, boots, long normal activity and independently annotated fall timing are missing.
9. **Publication:** source and compact reports are prepared on a branch based on the latest GitHub main. Private data, binaries, backups and raw logs remain local.
