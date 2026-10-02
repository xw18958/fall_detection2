# Seven-source V2 retraining

This experiment preserves the original six-source participant/recording partitions
and starts from their original SSL checkpoint. The M5 boot split is appended,
never used to reshuffle public participants. Checkpoint ancestry and normalization
counts must match the inherited split before any optimization starts.

Private V2 and M5 each receive 25% of sampled inputs and of source-normalized
classification loss. Each public source receives 10%. Classes and source groups
are balanced within each source. Global class weighting is disabled. Epoch length
is computed from cycling source/class/group queues so every eligible training
recording is visited; per-epoch coverage audits record actual draws.

Use `preprocess_m5_hard_negatives.py --root ORIGINAL_M5_ROOT --output-root NEW_M5_ROOT
--units si` to produce a separate full-range version. Original raw files and the
historical count-converted M5 version stay available. SI values are m/s² and rad/s,
using the metadata scales, with no signed-int16 compatibility clamp. Actual sensor
saturation flags remain in the audit. M5 normalization is fitted only on train boots.
Private V2 retains its exported numbers while its physical calibration is unresolved.

Documented public export units (numeric exports are preserved):

| Source | Acceleration | Gyroscope | Primary evidence |
|---|---|---|---|
| CGU_BES | g | rad/s | [Wang et al. 2018, Figure 1](https://doi.org/10.1088/1361-6579/aae0eb); original decimal files corroborated |
| Cogent | g | degrees/s | [Ojetola 2013, Appendix A.1](https://pure.coventry.ac.uk/ws/portalfiles/portal/40391885/Ojetola_2013.pdf); original CSV excerpt corroborated |
| SFU_IMU | m/s² | rad/s | Original IMU dataset README and column names |
| UCI_SimulatedFalls | m/s² | rad/s | [Xsens MTw manual, calibrated Acc/Gyr export columns](https://www.movella.com/hubfs/Downloads/Manuals/MTw_Awinda_User_Manual.pdf?hs=) |
| PAMAP2 | m/s² | rad/s | [UCI variable documentation](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) |

Each positive 5-second fall container remains positive. Random 3-second training
views inherit that label. No peak pseudo-labeling or phase supervision is used.
M5 negatives remain full sessions. Fine-tune validation and testing use every
complete 3-second window on the cumulative 0.25-second grid, including the last
window, without crossing gaps. SSL validation uses up to four fixed, spread views
from **every** validation recording, with deterministic augmentation pairs.

Budgets: additional SSL 10–30 epochs (patience 8), head warm-up 3, full-model
fine-tuning 10–50 (patience 10, learning-rate reduction after a plateau).
Budgets are starting limits, not established optimal epoch counts.

Model/threshold selection minimizes weighted validation error across all seven
sources. For fall-containing sources, error is half fall-recording FNR plus half
negative-window FPR; negative-only sources use negative-window FPR. Rates average
recordings within groups, then groups. Guards preserve baseline Private recall,
protect each public source's recall and recording specificity within two percentage
points, and prevent worse M5/PAMAP negative-window FPR. Failed guards remain reported;
test outcomes never relax them. Historical baseline uses its own original M5 input
pipeline and a jointly validation-calibrated threshold. The comparison is between
complete pipelines, not an isolated change to model weights.

Run fast unit tests and a complete `--smoke` train/test sequence first. Supply the
successful preflight hash report to `run_weighted_experiments.py`. It serially trains
seeds 42/43/44 on one GPU, selects the primary seed using validation, locks the
selection, then tests all three and the baseline. Shared caches avoid repeat CSV
processing. Checkpoints, provenance, coverage, learning curves, timing, retrieval,
per-source classification and M5 offline alarm episodes are saved. Raw datasets and
signal caches are excluded from Git. Firmware and the physical device are untouched.
