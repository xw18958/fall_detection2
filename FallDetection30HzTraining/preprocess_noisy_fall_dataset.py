#!/usr/bin/env python3
"""
Universal preprocessing for labelled fall recordings that contain surrounding non-fall activity.

Default policy (30 Hz commercial detector):
1. Resample each fall recording from time_ms to true target Hz.
2. Detect a strong temporally paired acceleration/gyroscope event.
3. Define event center as midpoint of the selected Acc and Gyro peaks.
4. If possible, extract a centered 5 s fall container.
5. If a centered 5 s container is impossible because of a recording boundary,
   extract a real 3 s fall window.
6. If the whole recording is shorter than 3 s, preserve all real samples and
   pad only the missing samples with low-amplitude IMU-like noise estimated
   from quiet non-fall recordings.
7. Save every remaining real PRE/POST chunk >=3 s as a hard negative.
8. Keep existing non-fall files unchanged and write a complete audit CSV.

No guard band is discarded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Dataset ZIP or directory containing fall/ and non-fall/.")
    p.add_argument("--output", required=True, help="Output dataset directory.")
    p.add_argument("--fs", type=float, default=30.0, help="Target sampling rate in Hz.")
    p.add_argument("--fall-window-sec", type=float, default=5.0,
                   help="Clean fall-container length when centered extraction is possible.")
    p.add_argument("--model-window-sec", type=float, default=3.0,
                   help="Deployment/model window length; also minimum hard-negative length.")
    p.add_argument("--max-pair-gap-sec", type=float, default=1.0,
                   help="Maximum time gap for pairing Acc and Gyro peaks.")
    p.add_argument("--seed", type=int, default=42, help="Seed for short-recording noise padding.")
    p.add_argument("--zip-output", default=None, help="Optional path for a ZIP of the output dataset.")
    return p.parse_args()


def find_dataset_root(root: Path) -> Path:
    if (root / "fall").is_dir() and (root / "non-fall").is_dir():
        return root
    matches = [p for p in root.rglob("*")
               if p.is_dir() and (p / "fall").is_dir() and (p / "non-fall").is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one dataset root with fall/ and non-fall/, found {len(matches)}.")
    return matches[0]


def load_resampled_df(path: Path, fs: float) -> pd.DataFrame:
    d = pd.read_csv(path)
    for c in ["time_ms", *FEATURES]:
        if c not in d.columns:
            raise ValueError(f"{path}: missing required column {c}")
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["time_ms", *FEATURES]).sort_values("time_ms")
    d = d.groupby("time_ms", as_index=False)[FEATURES].mean()
    if len(d) < 2:
        raise ValueError(f"{path}: fewer than two valid samples.")

    t = d["time_ms"].to_numpy(np.float64) / 1000.0
    x = d[FEATURES].to_numpy(np.float64)
    t -= t[0]

    n = max(2, int(math.floor(t[-1] * fs)) + 1)
    grid = np.arange(n, dtype=np.float64) / fs
    xr = np.stack([np.interp(grid, t, x[:, i]) for i in range(6)], axis=1)
    return as_output_df(xr, fs)


def as_output_df(arr: np.ndarray, fs: float) -> pd.DataFrame:
    arr = np.asarray(arr, dtype=float)
    d = pd.DataFrame(arr, columns=FEATURES)
    d.insert(0, "time_ms", np.arange(len(d), dtype=float) * 1000.0 / fs)
    d.insert(0, "sample_idx", np.arange(len(d), dtype=int))
    return d


def robust_z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = np.std(v) + 1e-9
    return (v - med) / scale


def local_peaks(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if len(v) < 3:
        return np.array([int(np.argmax(v))])
    p = np.where((v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]))[0] + 1
    return p if len(p) else np.array([int(np.argmax(v))])


def detect_event(d: pd.DataFrame, fs: float, max_pair_gap_sec: float) -> dict:
    x = d[FEATURES].to_numpy(float)
    acc_mag = np.linalg.norm(x[:, :3], axis=1)
    gyro_mag = np.linalg.norm(x[:, 3:], axis=1)

    acc_s = pd.Series(acc_mag).rolling(3, center=True, min_periods=1).median().to_numpy()
    gyro_s = pd.Series(gyro_mag).rolling(3, center=True, min_periods=1).median().to_numpy()

    za, zg = robust_z(acc_s), robust_z(gyro_s)
    pa, pg = local_peaks(acc_s), local_peaks(gyro_s)
    pa = pa[np.argsort(za[pa])[-min(30, len(pa)):]]
    pg = pg[np.argsort(zg[pg])[-min(30, len(pg)):]]

    pairs = []
    for ia in pa:
        for ig in pg:
            gap_sec = abs(int(ia) - int(ig)) / fs
            if gap_sec <= max_pair_gap_sec:
                score = max(0.0, float(za[ia])) + max(0.0, float(zg[ig])) - 2.0 * gap_sec
                pairs.append((score, int(ia), int(ig), float(gap_sec)))

    if pairs:
        score, ia, ig, gap = max(pairs, key=lambda z: z[0])
        method = "paired_acc_gyro"
    else:
        ia = int(np.argmax(acc_s))
        ig = int(pg[np.argmin(np.abs(pg - ia))]) if len(pg) else int(np.argmax(gyro_s))
        gap = abs(ia - ig) / fs
        score = max(0.0, float(za[ia])) + max(0.0, float(zg[ig]))
        method = "fallback_acc_nearest_gyro"

    mid = (ia + ig) / 2.0
    return {
        "acc_idx": ia,
        "gyro_idx": ig,
        "mid_idx": mid,
        "acc_time_s": ia / fs,
        "gyro_time_s": ig / fs,
        "mid_time_s": mid / fs,
        "peak_gap_s": gap,
        "score": float(score),
        "method": method,
    }


def estimate_quiet_noise(nonfall_paths: list[Path], fs: float) -> np.ndarray:
    quiet_parts = []
    for p in nonfall_paths:
        d = load_resampled_df(p, fs)
        arr = d[FEATURES].to_numpy(float)
        motion = np.linalg.norm(np.diff(arr, axis=0, prepend=arr[[0]]), axis=1)
        threshold = np.quantile(motion, 0.20)
        quiet_parts.append(arr[motion <= threshold])

    if not quiet_parts:
        return np.full(6, 1e-6, dtype=float)

    quiet = np.concatenate(quiet_parts, axis=0)
    steps = np.diff(quiet, axis=0)
    sigma = np.nanstd(steps, axis=0)
    return np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)


def choose_fitting_window(n: int, mid_idx: float, length: int) -> tuple[int, int]:
    start = int(round(mid_idx - (length - 1) / 2.0))
    start = max(0, min(start, n - length))
    return start, start + length


def pad_short_to_model_window(
    d: pd.DataFrame,
    mid_idx: float,
    model_n: int,
    fs: float,
    noise_std: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int, int]:
    real = d[FEATURES].to_numpy(float)
    n = len(real)
    missing = model_n - n
    target_mid = (model_n - 1) / 2.0

    pad_pre = int(round(target_mid - mid_idx))
    pad_pre = max(0, min(missing, pad_pre))
    pad_post = missing - pad_pre

    def random_walk(anchor: np.ndarray, n_pad: int) -> np.ndarray:
        if n_pad <= 0:
            return np.empty((0, 6), dtype=float)
        steps = rng.normal(0.0, noise_std * 0.25, size=(n_pad, 6))
        return anchor + np.cumsum(steps, axis=0)

    pre = random_walk(real[0], pad_pre)[::-1]
    post = random_walk(real[-1], pad_post)
    return as_output_df(np.vstack([pre, real, post]), fs), pad_pre, pad_post


def duplicate_groups(fall_paths: list[Path]) -> dict[str, str]:
    groups: dict[str, list[str]] = {}
    for p in fall_paths:
        raw = pd.read_csv(p)
        h = hashlib.sha256(raw[FEATURES].to_numpy().tobytes()).hexdigest()
        groups.setdefault(h, []).append(p.name)

    result = {}
    for names in groups.values():
        if len(names) > 1:
            tag = "|".join(names)
            for name in names:
                result[name] = tag
    return result


def preprocess(
    src_root: Path,
    out_root: Path,
    fs: float,
    fall_window_sec: float,
    model_window_sec: float,
    max_pair_gap_sec: float,
    seed: int,
) -> dict:
    src_fall, src_nonfall = src_root / "fall", src_root / "non-fall"
    out_fall, out_nonfall = out_root / "fall", out_root / "non-fall"

    if out_root.exists():
        shutil.rmtree(out_root)
    out_fall.mkdir(parents=True)
    out_nonfall.mkdir(parents=True)

    fall_paths = sorted(src_fall.glob("*.csv"))
    nonfall_paths = sorted(src_nonfall.glob("*.csv"))

    for p in nonfall_paths:
        shutil.copy2(p, out_nonfall / p.name)

    fall_n = int(round(fall_window_sec * fs))
    model_n = int(round(model_window_sec * fs))
    if fall_n < model_n:
        raise ValueError("fall-window-sec must be >= model-window-sec.")

    noise_std = estimate_quiet_noise(nonfall_paths, fs)
    rng = np.random.default_rng(seed)
    dup_map = duplicate_groups(fall_paths)

    audit = []
    generated_negatives = []

    for p in fall_paths:
        d = load_resampled_df(p, fs)
        event = detect_event(d, fs, max_pair_gap_sec)
        n = len(d)
        mid = event["mid_idx"]

        centered_start = int(round(mid - (fall_n - 1) / 2.0))
        can_center_fall = centered_start >= 0 and centered_start + fall_n <= n

        if can_center_fall:
            start, end = centered_start, centered_start + fall_n
            fall_out = as_output_df(d[FEATURES].iloc[start:end].to_numpy(float), fs)
            output_type = f"{fall_window_sec:g}s_centered"
            pad_pre = pad_post = 0
        elif n >= model_n:
            start, end = choose_fitting_window(n, mid, model_n)
            fall_out = as_output_df(d[FEATURES].iloc[start:end].to_numpy(float), fs)
            output_type = f"{model_window_sec:g}s_boundary"
            pad_pre = pad_post = 0
        else:
            start, end = 0, n
            fall_out, pad_pre, pad_post = pad_short_to_model_window(
                d, mid, model_n, fs, noise_std, rng
            )
            output_type = f"{model_window_sec:g}s_padded"

        fall_out.to_csv(out_fall / p.name, index=False)

        pre_name = post_name = ""
        if start >= model_n:
            pre_name = f"NF_FROM_{p.stem}_PRE.csv"
            as_output_df(d[FEATURES].iloc[:start].to_numpy(float), fs).to_csv(
                out_nonfall / pre_name, index=False
            )
            generated_negatives.append(pre_name)

        if n - end >= model_n:
            post_name = f"NF_FROM_{p.stem}_POST.csv"
            as_output_df(d[FEATURES].iloc[end:].to_numpy(float), fs).to_csv(
                out_nonfall / post_name, index=False
            )
            generated_negatives.append(post_name)

        win_center = (start + end - 1) / 2.0
        center_err_ms = (
            abs(mid - win_center) * 1000.0 / fs if can_center_fall else np.nan
        )

        audit.append({
            "source_file": p.name,
            "resampled_source_samples": n,
            "resampled_source_duration_s": n / fs,
            "acc_peak_time_s": event["acc_time_s"],
            "gyro_peak_time_s": event["gyro_time_s"],
            "peak_difference_s": event["peak_gap_s"],
            "event_midpoint_s": event["mid_time_s"],
            "event_method": event["method"],
            "event_score": event["score"],
            "output_type": output_type,
            "fall_start_idx": start,
            "fall_end_idx_exclusive": end,
            "fall_output_samples": len(fall_out),
            "centering_error_ms": center_err_ms,
            "padded_pre_samples": pad_pre,
            "padded_post_samples": pad_post,
            "padded_total_samples": pad_pre + pad_post,
            "generated_pre_negative": pre_name,
            "generated_post_negative": post_name,
            "duplicate_group": dup_map.get(p.name, ""),
        })

    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(out_root / "preprocessing_audit.csv", index=False)

    centered_label = f"{fall_window_sec:g}s_centered"
    boundary_label = f"{model_window_sec:g}s_boundary"
    padded_label = f"{model_window_sec:g}s_padded"
    summary = {
        "target_sampling_rate_hz": fs,
        "fall_files_total": len(audit_df),
        "fall_centered": int((audit_df.output_type == centered_label).sum()),
        "fall_boundary": int((audit_df.output_type == boundary_label).sum()),
        "fall_padded": int((audit_df.output_type == padded_label).sum()),
        "original_nonfall_files_preserved": len(nonfall_paths),
        "generated_hard_negative_files": len(generated_negatives),
        "final_nonfall_files": len(list(out_nonfall.glob("*.csv"))),
        "duplicate_fall_files_flagged": int(
            (audit_df.duplicate_group.fillna("") != "").sum()
        ),
    }
    (out_root / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    for _, r in audit_df.iterrows():
        d = pd.read_csv(out_fall / r.source_file)
        expected = fall_n if r.output_type == centered_label else model_n
        if len(d) != expected:
            raise RuntimeError(f"{r.source_file}: output length {len(d)} != {expected}")
    for name in generated_negatives:
        if len(pd.read_csv(out_nonfall / name)) < model_n:
            raise RuntimeError(f"{name}: hard negative shorter than model window.")

    return summary


def zip_directory(root: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(root.rglob("*")):
            if fp.is_file():
                z.write(fp, arcname=str(root.name / fp.relative_to(root)))


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    with tempfile.TemporaryDirectory(prefix="fall_preprocess_") as tmp:
        tmp = Path(tmp)
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(input_path, "r") as z:
                z.extractall(tmp)
            src_root = find_dataset_root(tmp)
        elif input_path.is_dir():
            src_root = find_dataset_root(input_path)
        else:
            raise FileNotFoundError(input_path)

        summary = preprocess(
            src_root=src_root,
            out_root=output_path,
            fs=args.fs,
            fall_window_sec=args.fall_window_sec,
            model_window_sec=args.model_window_sec,
            max_pair_gap_sec=args.max_pair_gap_sec,
            seed=args.seed,
        )

    if args.zip_output:
        zip_directory(output_path, Path(args.zip_output).expanduser().resolve())

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
