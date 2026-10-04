# Trigger training and power benchmarking

## Actual installed V5 path

The installed V5 power implementation is documented in
[`M5BLECollector/power_v5`](../M5BLECollector/power_v5/README.md).
Use `train_power_trigger_v5.py`, `trigger_data_v5.py`, `power_events_v5.py` and
`benchmark_power_tcn_v5.py` for its exact frozen SI model/data contract.
Results are in `deployment_artifacts/power_v5_20261004`. V5 does **not** use the
V2 compatibility count/clipping view described below. Labels, 5-second crop,
normalization and existing V5 group splits remain unchanged.

## Earlier standalone V2 experiment

The standalone firmware holds an older V2 model. This work preserves that full
model, its original six-source checkpoint normalization, threshold and group
partitions. The existing V5 combined collector is a different input contract and
was not modified/flashed by that V2 experiment. M5 inputs are an explicit compatibility **view** of the
prepared SI data: acceleration / 9.80665 * 16384, gyro / (pi/180) * 16.4, int16
clamp, then original private normalization. This matches the preserved standalone
firmware's historical inferred count mapping. Source CSVs and V5 SI normalization
are untouched; this compatibility view does not cure their domain mismatch.

Original train/val/test recording keys and group separation are checked exactly.
The existing M5 13/3/3 fall split and 5/1/1 negative split are added unchanged;
SHA256 checks verify every prepared M5 input. Unknown subject/placement and falls
from one boot limit generalization. No event timings or new labels are inferred.
Prepared falls retain the previously requested 5-second crop. Full 3-second views
inherit recording labels. This can label preparation/recovery as positive and
makes very high *window* recall an intentionally conservative trigger criterion.

`train_power_trigger.py --smoke` must pass before a matching source-hash full run.
It checks splits, model/checkpoint/config identity, normalization, real segmented
90x6 views, finite CUDA gradients, PyTorch→TF graph equality, INT8-only conversion
and native Micro execution for CNN/MLP/full TCN. RF is host-only comparison.
Calibration uses training inputs only with recorded source/class coverage.

Seed42 candidates: 8-channel depthwise temporal CNN (506 parameters), flattened
540-input MLP (16 hidden units), RF (16 trees, depth4). Inputs always use every
sample and all six axes. CNN uses static batch1, INT8 input/output, integer-only
convolution/depthwise/mean/dense operations. No heuristic substitutes for training.

Validation selects epochs and threshold: at least 99.5% fall-window recall for
each positive source, zero added baseline true-positive misses, no lost baseline
fall recording. At 4 Hz the full invocation fraction must be below 1/3 to beat the
existing 1.333 Hz full-inference count even before paying trigger cost. This is a
necessary screening condition, not proof of net energy savings. The threshold
and candidate lock is written before test is opened. Test failure rejects
eligibility; it never changes threshold or chooses another test-ranked model.
No candidate is deployed when the gate fails.

`trigger_data.evaluation` explicitly uses production quarter-second starts
`floor(7.5*k)` and does not append the historical off-grid endpoint. Source
resampling/units/normalization/labels/splits remain unchanged. This cadence change
is recorded in the protocol. An earlier exploratory run (`train`) used the
historical endpoint sampler; the production-grid rerun (`train_grid`) supersedes
its runtime metrics. Prior inspection of the same test sources means these are
exploratory engineering evaluations, not a fresh independent commercial test.
Further phase/jitter and event-level validation on new continuous M5 recordings
is required before relying on trigger suppression.

False wake-ups/hour counts **wake episodes**, including cap/rearm events, on
continuous negative segments at least 30 seconds, excluding 3-second warmup.
Short fall/non-fall crops do not manufacture hours of normal monitoring.
Interrupted segments reset the state machine. Reports include per-source trigger
recall, final/baseline recall/specificity/precision/F1, recording recall, lost
baseline fall recordings, active fractions and exposure. They do not imply real
world false alarms/hour. Host replay timing is not M5 latency/current.

`benchmark_power_tcn.py` isolates width16, two blocks, and mean attention against
matched full-width/control fine-tuning. Every model uses seed42, 5×6400 draws,
8e-5 learning rate, same source/group-balanced sampler, frozen original BN stats
and fixed original decision threshold; architecture-specific warm starts slice
acc/gyro fuse axes separately. A comprehensive short smoke runs first. Per-source
recall, specificity and F1 must not decline. Parameter/MAC/host-latency reductions
alone never authorize replacing the full model. Unqualified float variants are
not exported for deployment; the unchanged baseline is already INT8. Further
INT8 native/device qualification is mandatory if a float variant passes.

Results and commands are saved under
`deployment_artifacts/power_optimization_20261004/` locally and
`/raid1/xwan0900/power_optimization_20261004/` remotely. Raw data, checkpoints,
calibration and firmware binaries are not public-release artifacts. Zero-shot
and retrieval metrics do not apply to this supervised sensor classifier.

See the firmware's `README_POWER_OPTIMIZATION.md` for build modes and controls.
Use `power_measurements.py` for telemetry/current integration; `power_cascade.py`
contains the mirrored state machine and a current-profile-based energy estimator.
Neither invents a battery/current measurement.
