#!/usr/bin/env python3
"""
Earthformer evaluation on Seasonal Matched-Pair Benchmark.

Evaluates a trained Earthformer checkpoint on the EO-WM Seasonal Matched-Pair
benchmark. If --data_csv is omitted, the script expands the public pair CSV into
the two window samples needed for inference.

Usage:
    export EARTHFORMER_REPO=/path/to/earth-forecasting-transformer

    python script/earthformer_eval_seasonal_bench.py \
        --cfg cfg.yaml \
        --ckpt_path /path/to/earthformer_earthnet2021.pt \
        --earthnet_root /path/to/EarthNet2021 \
        --benchmark_pairs_csv benchmark_csv/seasonal_pairs_benchmark.csv \
        --output_dir ./earthformer_seasonal_results \
        --batch_size 4
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

_earthformer_repo = os.environ.get("EARTHFORMER_REPO")
if _earthformer_repo:
    _earthformer_repo = os.path.realpath(_earthformer_repo)
    _earthformer_script_dir = os.path.join(
        _earthformer_repo, "scripts", "cuboid_transformer", "earthnet_w_meso"
    )
    for _path in (_earthformer_script_dir, _earthformer_repo):
        if _path not in sys.path:
            sys.path.insert(0, _path)

from earthformer.cuboid_transformer.cuboid_transformer_unet_dec import CuboidTransformerAuxModel
from earthformer.datasets.earthnet.earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute


# =====================================================================
# NDVI Helper Functions
# =====================================================================

def compute_ndvi(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute NDVI from 4-channel EarthNet data.

    EarthNet channels: [B, G, R, NIR] (indices 0,1,2,3)
    NDVI = (NIR - R) / (NIR + R + eps)

    Args:
        x: (C, T, H, W) or (N, C, T, H, W) with C >= 4.
    Returns:
        NDVI tensor: (T, H, W) or (N, T, H, W), values in [-1, 1].
    """
    if x.ndim == 4:
        nir, red = x[3], x[2]
    elif x.ndim == 5:
        nir, red = x[:, 3], x[:, 2]
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {x.ndim}D")
    return (nir - red) / (nir + red + eps)


def compute_vegetation_mask(ndvi: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
    """Create vegetation mask where NDVI >= threshold."""
    return ndvi >= threshold


# =====================================================================
# JSON Encoder
# =====================================================================

class _NaNSafeEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN/Inf to None and numpy types to Python types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if not math.isfinite(v) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def encode(self, o):
        return super().encode(self._sanitize(o))

    def _sanitize(self, obj):
        if isinstance(obj, float):
            return None if not math.isfinite(obj) else obj
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        return obj


# =====================================================================
# Seasonal Benchmark Metadata Parsing
# =====================================================================

def parse_seasonal_benchmark_meta(meta_json_str: str) -> dict:
    """Parse benchmark_meta_json for seasonal paired benchmark.

    Returns dict with:
        pair_id, sample_side ("a"/"b"), paired_window_uid,
        year_index, window_start_frame_in_year, final_rank,
        n_tracks, track_labels
    """
    raw = json.loads(meta_json_str)

    def _int(key, default=0):
        v = raw.get(key, default)
        if v is None:
            return default
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return default

    return {
        "pair_id": str(raw.get("pair_id", "")),
        "sample_side": str(raw.get("sample_side", "")),
        "paired_window_uid": str(raw.get("paired_window_uid", "")),
        "year_index": _int("year_index"),
        "window_start_frame_in_year": _int("window_start_frame_in_year"),
        "final_rank": _int("final_rank"),
        "n_tracks": _int("n_tracks"),
        "track_labels": str(raw.get("track_labels", "")),
    }


# =====================================================================
# WindowSummary
# =====================================================================


def _nanmean_axis0(arr: np.ndarray) -> np.ndarray:
    """np.nanmean along axis=0, suppressing RuntimeWarning for all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=0)


@dataclasses.dataclass
class WindowSummary:
    """Per-window prediction summary computed during inference."""

    window_uid: str

    # NDVI trajectories — spatial-mean over vegetation pixels
    ndvi_pred_trajectory: np.ndarray      # (N_ensemble, T_out)
    ndvi_gt_trajectory: np.ndarray        # (T_out,)

    # Pixel-level spatial means
    pixel_pred_mean: np.ndarray           # (N_ensemble, T_out, C)
    pixel_gt_mean: np.ndarray             # (T_out, C)

    # Quality
    mask_valid_ratio: np.ndarray          # (T_out,) fraction valid per frame

    # Per-window standard metrics (vs GT)
    pixel_mae: float
    ndvi_mae: float


def compute_window_summary(
    pred_ensemble: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
    window_uid: str,
) -> Tuple[WindowSummary, torch.Tensor]:
    """Compute WindowSummary from ensemble predictions and GT.

    Args:
        pred_ensemble: (N, C, T_out, H, W) ensemble predictions (CPU tensors).
        gt: (C, T_out, H, W) ground truth (CPU tensor).
        valid_mask: (1, T_out, H, W) validity mask (1=valid, CPU tensor).
        window_uid: unique window identifier (= sample_id from inference CSV).

    Returns:
        (WindowSummary, pred_mean) where pred_mean is (C, T_out, H, W).
    """
    N, C, T_out, H, W = pred_ensemble.shape
    mask = valid_mask.squeeze(0)  # (T_out, H, W)

    pred_mean = pred_ensemble.mean(dim=0)  # (C, T_out, H, W)

    # --- Vegetation mask from GT NDVI ---
    gt_ndvi = compute_ndvi(gt)              # (T_out, H, W)
    veg_mask = compute_vegetation_mask(gt_ndvi)  # (T_out, H, W)
    combined_mask = mask * veg_mask.float()  # (T_out, H, W)

    # --- NDVI trajectories (vectorized) ---
    n_veg_per_frame = combined_mask.sum(dim=(-2, -1))  # (T_out,)
    n_veg_safe = n_veg_per_frame.clamp(min=1)

    # GT NDVI trajectory: spatial-mean over vegetation pixels
    gt_ndvi_masked_sum = (gt_ndvi * combined_mask).sum(dim=(-2, -1))  # (T_out,)
    ndvi_gt_traj = (gt_ndvi_masked_sum / n_veg_safe).numpy().astype(np.float64)
    ndvi_gt_traj[n_veg_per_frame.numpy() == 0] = np.nan

    # Pred NDVI trajectory per ensemble member
    pred_ndvi_all = compute_ndvi(pred_ensemble)  # (N, T_out, H, W)
    pred_ndvi_masked_sum = (pred_ndvi_all * combined_mask.unsqueeze(0)).sum(dim=(-2, -1))
    ndvi_pred_traj = (pred_ndvi_masked_sum / n_veg_safe.unsqueeze(0)).numpy().astype(np.float64)
    ndvi_pred_traj[:, n_veg_per_frame.numpy() == 0] = np.nan

    # --- Pixel spatial means (using full mask, not veg-only) ---
    n_pix_per_frame = mask.sum(dim=(-2, -1))  # (T_out,)
    n_pix_safe = n_pix_per_frame.clamp(min=1)

    gt_masked_sum = (gt * mask.unsqueeze(0)).sum(dim=(-2, -1))  # (C, T_out)
    pixel_gt_mean = (gt_masked_sum / n_pix_safe.unsqueeze(0)).permute(1, 0).numpy().astype(np.float64)
    pixel_gt_mean[n_pix_per_frame.numpy() == 0, :] = np.nan

    pred_masked_sum = (pred_ensemble * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))  # (N, C, T_out)
    pixel_pred_mean = (pred_masked_sum / n_pix_safe.unsqueeze(0).unsqueeze(0)).permute(0, 2, 1).numpy().astype(np.float64)
    pixel_pred_mean[:, n_pix_per_frame.numpy() == 0, :] = np.nan

    # --- Valid ratio per frame ---
    mask_valid_ratio = mask.float().mean(dim=(-2, -1)).numpy().astype(np.float64)  # (T_out,)

    # --- Per-window quality: pixel MAE and NDVI MAE (ensemble mean vs GT) ---
    diff = (pred_mean - gt) * mask.unsqueeze(0)  # (C, T_out, H, W)
    n_valid = mask.sum() * C
    pixel_mae = (diff.abs().sum().item() / n_valid.item()) if n_valid > 0 else float("nan")

    ndvi_pred_mean_map = compute_ndvi(pred_mean)  # (T_out, H, W)
    ndvi_diff = (ndvi_pred_mean_map - gt_ndvi) * combined_mask
    n_ndvi_valid = combined_mask.sum()
    ndvi_mae = (ndvi_diff.abs().sum().item() / n_ndvi_valid.item()) if n_ndvi_valid > 0 else float("nan")

    summary = WindowSummary(
        window_uid=window_uid,
        ndvi_pred_trajectory=ndvi_pred_traj,
        ndvi_gt_trajectory=ndvi_gt_traj,
        pixel_pred_mean=pixel_pred_mean,
        pixel_gt_mean=pixel_gt_mean,
        mask_valid_ratio=mask_valid_ratio,
        pixel_mae=pixel_mae,
        ndvi_mae=ndvi_mae,
    )
    return summary, pred_mean


# =====================================================================
# PairMetricComputer
# =====================================================================

class PairMetricComputer:
    """Computes paired comparison metrics across all benchmark pairs.

    Metrics:
        DRR  — Divergence Reproduction Ratio (magnitude fidelity)
        DHR  — Directional Hit Rate (sign fidelity)
        PDC  — Paired Divergence Correlation (ranking fidelity, Spearman)
    """

    DRR_NOISE_THR = 0.02
    DHR_NOISE_THR = 0.02

    def __init__(
        self,
        benchmark_pairs_csv: str,
        window_summaries: Dict[str, WindowSummary],
    ):
        self.pairs_df = pd.read_csv(benchmark_pairs_csv)
        self.summaries = window_summaries

    def compute_all(self) -> Tuple[dict, List[dict]]:
        """Compute all metrics. Returns (overall_metrics_dict, per_pair_records_list)."""
        per_pair_records: List[dict] = []
        n_skipped = 0

        for _, row in self.pairs_df.iterrows():
            pair_id = str(row["pair_id"])
            uid_a = str(row["window_uid_a"])
            uid_b = str(row["window_uid_b"])

            sa = self.summaries.get(uid_a)
            sb = self.summaries.get(uid_b)
            if sa is None or sb is None:
                n_skipped += 1
                continue

            rec = self._compute_pair_metrics(sa, sb, row)
            rec["pair_id"] = pair_id
            rec["cube_id"] = str(row.get("cube_id", ""))
            rec["track_labels"] = str(row.get("track_labels", ""))

            per_pair_records.append(rec)

        if n_skipped > 0:
            print(f"[PairMetricComputer] skipped {n_skipped} pairs (missing window summaries)")

        overall = self._compute_overall(per_pair_records)
        overall["n_pairs_evaluated"] = len(per_pair_records)
        overall["n_pairs_skipped"] = n_skipped

        return overall, per_pair_records

    def _compute_overall(self, per_pair_records: List[dict]) -> dict:
        if not per_pair_records:
            return {}

        drr_vals = [r["DRR"] for r in per_pair_records if np.isfinite(r.get("DRR", np.nan))]
        dhr_hits = sum(r.get("dhr_hits", 0) for r in per_pair_records)
        dhr_total = sum(r.get("dhr_valid_steps", 0) for r in per_pair_records)

        result = {
            "DRR_mean": float(np.nanmean(drr_vals)) if drr_vals else float("nan"),
            "DRR_count": len(drr_vals),
            "DHR": (dhr_hits / dhr_total) if dhr_total > 0 else float("nan"),
            "DHR_total_steps": dhr_total,
            "PDC_spearman": self._compute_pdc(per_pair_records),
        }

        mae_a = [r.get("mae_window_a", np.nan) for r in per_pair_records]
        mae_b = [r.get("mae_window_b", np.nan) for r in per_pair_records]
        all_mae = [v for v in mae_a + mae_b if np.isfinite(v)]
        result["per_window_pixel_mae"] = float(np.mean(all_mae)) if all_mae else float("nan")

        ndvi_a = [r.get("ndvi_mae_a", np.nan) for r in per_pair_records]
        ndvi_b = [r.get("ndvi_mae_b", np.nan) for r in per_pair_records]
        all_ndvi = [v for v in ndvi_a + ndvi_b if np.isfinite(v)]
        result["per_window_ndvi_mae"] = float(np.mean(all_ndvi)) if all_ndvi else float("nan")

        return result

    def _compute_pair_metrics(
        self,
        summary_a: WindowSummary,
        summary_b: WindowSummary,
        pair_row: pd.Series,
    ) -> dict:
        ndvi_pred_a = _nanmean_axis0(summary_a.ndvi_pred_trajectory)
        ndvi_pred_b = _nanmean_axis0(summary_b.ndvi_pred_trajectory)
        ndvi_gt_a = summary_a.ndvi_gt_trajectory
        ndvi_gt_b = summary_b.ndvi_gt_trajectory

        drr = self._compute_drr(ndvi_pred_a, ndvi_pred_b, ndvi_gt_a, ndvi_gt_b)
        dhr_hits, dhr_valid = self._compute_dhr(ndvi_pred_a, ndvi_pred_b, ndvi_gt_a, ndvi_gt_b)
        dhr = (dhr_hits / dhr_valid) if dhr_valid > 0 else float("nan")

        real_div = np.abs(ndvi_gt_a - ndvi_gt_b)
        pred_div = np.abs(ndvi_pred_a - ndvi_pred_b)
        finite_mask = np.isfinite(real_div) & np.isfinite(pred_div)
        real_total = float(np.sum(real_div[finite_mask])) if finite_mask.any() else float("nan")
        pred_total = float(np.sum(pred_div[finite_mask])) if finite_mask.any() else float("nan")

        return {
            "DRR": drr,
            "DHR": dhr,
            "dhr_hits": dhr_hits,
            "dhr_valid_steps": dhr_valid,
            "real_ndvi_div_total": real_total,
            "pred_ndvi_div_total": pred_total,
            "mae_window_a": summary_a.pixel_mae,
            "mae_window_b": summary_b.pixel_mae,
            "ndvi_mae_a": summary_a.ndvi_mae,
            "ndvi_mae_b": summary_b.ndvi_mae,
        }

    def _compute_drr(
        self,
        ndvi_pred_a: np.ndarray,
        ndvi_pred_b: np.ndarray,
        ndvi_gt_a: np.ndarray,
        ndvi_gt_b: np.ndarray,
    ) -> float:
        real_div = np.abs(ndvi_gt_a - ndvi_gt_b)
        pred_div = np.abs(ndvi_pred_a - ndvi_pred_b)

        valid = (np.isfinite(real_div) & np.isfinite(pred_div)
                 & (real_div > self.DRR_NOISE_THR))

        if not valid.any():
            return float("nan")

        mean_real = float(np.mean(real_div[valid]))
        mean_pred = float(np.mean(pred_div[valid]))

        if mean_real < 1e-10:
            return float("nan")

        return mean_pred / mean_real

    def _compute_dhr(
        self,
        ndvi_pred_a: np.ndarray,
        ndvi_pred_b: np.ndarray,
        ndvi_gt_a: np.ndarray,
        ndvi_gt_b: np.ndarray,
    ) -> Tuple[int, int]:
        gt_diff = ndvi_gt_a - ndvi_gt_b
        pred_diff = ndvi_pred_a - ndvi_pred_b

        valid = (np.isfinite(gt_diff) & np.isfinite(pred_diff)
                 & (np.abs(gt_diff) > self.DHR_NOISE_THR))

        n_valid = int(valid.sum())
        if n_valid == 0:
            return 0, 0

        hits = int(np.sum(np.sign(gt_diff[valid]) == np.sign(pred_diff[valid])))
        return hits, n_valid

    def _compute_pdc(self, per_pair_records: List[dict]) -> float:
        real_divs = []
        pred_divs = []
        for r in per_pair_records:
            rv = r.get("real_ndvi_div_total", np.nan)
            pv = r.get("pred_ndvi_div_total", np.nan)
            if np.isfinite(rv) and np.isfinite(pv):
                real_divs.append(rv)
                pred_divs.append(pv)

        if len(real_divs) < 3:
            return float("nan")

        rho, _ = scipy_stats.spearmanr(real_divs, pred_divs)
        return float(rho)

# =====================================================================
# Dataset: CSV-driven NPZ loading for Seasonal Benchmark
# =====================================================================

def resolve_earthnet_path(path_value: str, earthnet_root, split_name: str) -> str:
    """Resolve public benchmark CSV paths against a local EarthNet2021 root."""
    path_str = str(path_value)
    if os.path.exists(path_str):
        return path_str
    if not earthnet_root:
        return path_str

    if split_name in path_str:
        rel = path_str.split(split_name, 1)[1].lstrip("/\\")
        return os.path.join(earthnet_root, split_name, rel)
    return os.path.join(earthnet_root, path_str.lstrip("/\\"))


def _json_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def build_inference_df_from_pairs(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Expand the public pair CSV into per-window inference rows."""
    rows = []
    for _, pair in pairs_df.iterrows():
        tile_id = str(pair["tile_id"])
        cube_id = str(pair["cube_id"])
        window_start = int(pair["window_start_frame_in_year"])
        context_rel = os.path.join("seasonal_test_split", "context", tile_id, f"context_{cube_id}.npz")
        target_rel = os.path.join("seasonal_test_split", "target", tile_id, f"target_{cube_id}.npz")

        for side, year_col, uid_col, paired_uid_col, valid_col in [
            ("a", "year_index_a", "window_uid_a", "window_uid_b", "mean_valid_ratio_full30_a"),
            ("b", "year_index_b", "window_uid_b", "window_uid_a", "mean_valid_ratio_full30_b"),
        ]:
            year_index = int(pair[year_col])
            start_idx = year_index * 70 + window_start
            in_indices = list(range(start_idx, start_idx + 10))
            out_indices = list(range(start_idx + 10, start_idx + 30))
            meta = {
                "pair_id": str(pair["pair_id"]),
                "sample_side": side,
                "paired_window_uid": str(pair[paired_uid_col]),
                "year_index": year_index,
                "window_start_frame_in_year": window_start,
                "final_rank": _json_scalar(pair.get("final_rank", None)),
                "n_tracks": _json_scalar(pair.get("n_tracks", None)),
                "track_labels": str(pair.get("track_labels", "")),
            }
            rows.append(
                {
                    "path": context_rel,
                    "path_target": target_rel,
                    "sample_id": str(pair[uid_col]),
                    "in_indices": json.dumps(in_indices),
                    "out_indices": json.dumps(out_indices),
                    "in_len": 10,
                    "out_len": 20,
                    "window_idx": len(rows),
                    "in_valid_frac": _json_scalar(pair.get(valid_col, None)),
                    "out_valid_frac": _json_scalar(pair.get(valid_col, None)),
                    "benchmark_meta_json": json.dumps(meta),
                }
            )

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="sample_id", keep="first").reset_index(drop=True)
    df["window_idx"] = np.arange(len(df), dtype=np.int64)
    return df


class SeasonalCSVDataset(Dataset):
    """Dataset that loads EarthNet2021 NPZ files based on a benchmark CSV.

    Each CSV row specifies:
      - path: context NPZ file
      - path_target: target NPZ file
      - in_indices: JSON list of absolute frame indices for context
      - out_indices: JSON list of absolute frame indices for target
      - benchmark_meta_json: JSON string with benchmark metadata

    Returns dicts compatible with Earthformer forward pass (layout THWC).
    """

    def __init__(
        self,
        csv_path: Optional[str],
        pairs_csv: str,
        layout: str = "THWC",
        static_layout: str = "CHW",
        earthnet_root=None,
    ):
        if csv_path is None:
            self.df = build_inference_df_from_pairs(pd.read_csv(pairs_csv))
        else:
            self.df = pd.read_csv(csv_path)
        self.layout = layout
        self.static_layout = static_layout
        self.earthnet_root = earthnet_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        context_path = resolve_earthnet_path(row["path"], self.earthnet_root, "seasonal_test_split")
        target_path = resolve_earthnet_path(row["path_target"], self.earthnet_root, "seasonal_test_split")
        in_indices = np.array(json.loads(row["in_indices"]), dtype=np.int64)
        out_indices = np.array(json.loads(row["out_indices"]), dtype=np.int64)

        # Load NPZ files
        context_npz = np.load(context_path, allow_pickle=False)
        target_npz = np.load(target_path, allow_pickle=False)

        # highresdynamic: (128, 128, 7, T) — concatenate context + target along time axis
        context_hrd = context_npz["highresdynamic"].astype(np.float32)
        target_hrd = target_npz["highresdynamic"].astype(np.float32)
        hrd_all = np.concatenate([context_hrd, target_hrd], axis=-1)  # (128, 128, 7, T_total)

        # Select frames by indices
        all_indices = np.concatenate([in_indices, out_indices])
        hrd_selected = hrd_all[:, :, :, all_indices]  # (128, 128, 7, 30)

        # Clean up
        hrd_selected = np.nan_to_num(hrd_selected, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
        np.clip(hrd_selected, 0.0, 1.0, out=hrd_selected)

        # Static and meso data (load from context NPZ)
        highresstatic = context_npz["highresstatic"].astype(np.float32)
        highresstatic = np.nan_to_num(highresstatic, copy=False, nan=0.0)

        mesodynamic = context_npz["mesodynamic"].astype(np.float32)
        mesodynamic = np.nan_to_num(mesodynamic, copy=False, nan=0.0)

        mesostatic = context_npz["mesostatic"].astype(np.float32)
        mesostatic = np.nan_to_num(mesostatic, copy=False, nan=0.0)

        # Layout conversion: default NPZ layout is HWCT
        hrd_selected = self._change_layout(hrd_selected, "HWCT", self.layout)
        highresstatic = self._change_layout_static(highresstatic, "HWC", self.static_layout)
        mesodynamic = self._change_layout(mesodynamic, "HWCT", self.layout)
        mesostatic = self._change_layout_static(mesostatic, "HWC", self.static_layout)

        return {
            "highresdynamic": hrd_selected,
            "highresstatic": highresstatic,
            "mesodynamic": mesodynamic,
            "mesostatic": mesostatic,
            # Metadata (strings — collated as lists by DataLoader)
            "sample_id": str(row["sample_id"]),
            "window_idx": int(row["window_idx"]),
            "in_len": int(row["in_len"]),
            "out_len": int(row["out_len"]),
            "benchmark_meta_json": str(row["benchmark_meta_json"]),
        }

    @staticmethod
    def _change_layout(data, from_layout, to_layout):
        if from_layout == to_layout:
            return data
        from_str = " ".join(from_layout)
        to_str = " ".join(to_layout)
        from einops import rearrange as _rearrange
        return _rearrange(data, f"{from_str} -> {to_str}")

    @staticmethod
    def _change_layout_static(data, from_layout, to_layout):
        if from_layout == to_layout:
            return data
        from_str = " ".join(from_layout)
        to_str = " ".join(to_layout)
        from einops import rearrange as _rearrange
        return _rearrange(data, f"{from_str} -> {to_str}")


def collate_fn_seasonal(batch):
    """Custom collate that handles mixed tensor + metadata fields."""
    tensor_keys = ["highresdynamic", "highresstatic", "mesodynamic", "mesostatic"]
    meta_keys = ["sample_id", "window_idx", "in_len", "out_len", "benchmark_meta_json"]

    collated = {}
    for key in tensor_keys:
        collated[key] = torch.from_numpy(np.stack([b[key] for b in batch], axis=0))
    for key in meta_keys:
        collated[key] = [b[key] for b in batch]

    return collated


# =====================================================================
# Model Builder (from earthformer_eval_extreme_summer_bench.py)
# =====================================================================

def build_model_from_config(oc):
    model_cfg = OmegaConf.to_object(oc.model)
    num_blocks = len(model_cfg["enc_depth"])

    def _expand_pattern(key):
        val = model_cfg[key]
        if isinstance(val, str):
            return [val] * num_blocks
        return list(val)

    model = CuboidTransformerAuxModel(
        input_shape=model_cfg["input_shape"],
        target_shape=model_cfg["target_shape"],
        base_units=model_cfg["base_units"],
        block_units=model_cfg.get("block_units", None),
        scale_alpha=model_cfg["scale_alpha"],
        enc_depth=model_cfg["enc_depth"],
        dec_depth=model_cfg["dec_depth"],
        enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
        dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
        dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
        downsample=model_cfg["downsample"],
        downsample_type=model_cfg["downsample_type"],
        enc_attn_patterns=_expand_pattern("self_pattern"),
        dec_self_attn_patterns=_expand_pattern("cross_self_pattern"),
        dec_cross_attn_patterns=_expand_pattern("cross_pattern"),
        dec_cross_last_n_frames=model_cfg["dec_cross_last_n_frames"],
        num_heads=model_cfg["num_heads"],
        attn_drop=model_cfg["attn_drop"],
        proj_drop=model_cfg["proj_drop"],
        ffn_drop=model_cfg["ffn_drop"],
        upsample_type=model_cfg["upsample_type"],
        ffn_activation=model_cfg["ffn_activation"],
        gated_ffn=model_cfg["gated_ffn"],
        norm_layer=model_cfg["norm_layer"],
        num_global_vectors=model_cfg["num_global_vectors"],
        use_dec_self_global=model_cfg["use_dec_self_global"],
        dec_self_update_global=model_cfg["dec_self_update_global"],
        use_dec_cross_global=model_cfg["use_dec_cross_global"],
        use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
        use_global_self_attn=model_cfg["use_global_self_attn"],
        separate_global_qkv=model_cfg["separate_global_qkv"],
        global_dim_ratio=model_cfg["global_dim_ratio"],
        initial_downsample_type=model_cfg["initial_downsample_type"],
        initial_downsample_activation=model_cfg["initial_downsample_activation"],
        initial_downsample_stack_conv_num_layers=model_cfg["initial_downsample_stack_conv_num_layers"],
        initial_downsample_stack_conv_dim_list=model_cfg["initial_downsample_stack_conv_dim_list"],
        initial_downsample_stack_conv_downscale_list=model_cfg["initial_downsample_stack_conv_downscale_list"],
        initial_downsample_stack_conv_num_conv_list=model_cfg["initial_downsample_stack_conv_num_conv_list"],
        padding_type=model_cfg["padding_type"],
        checkpoint_level=model_cfg["checkpoint_level"],
        pos_embed_type=model_cfg["pos_embed_type"],
        use_relative_pos=model_cfg["use_relative_pos"],
        self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
        attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
        ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
        conv_init_mode=model_cfg["conv_init_mode"],
        down_up_linear_init_mode=model_cfg["down_up_linear_init_mode"],
        norm_init_mode=model_cfg["norm_init_mode"],
        auxiliary_channels=model_cfg["auxiliary_channels"],
        unet_dec_cross_mode=model_cfg["unet_dec_cross_mode"],
    )
    return model


# =====================================================================
# Forward Pass (from earthformer_eval_extreme_summer_bench.py)
# =====================================================================

def earthformer_forward(model, batch, in_len, out_len, img_height, img_width, data_channels=4, device="cuda"):
    """Run Earthformer forward pass.

    Args:
        batch: dict with "highresdynamic" (N,T,H,W,C), "highresstatic" (N,C,H,W),
               "mesodynamic" (N,T,H,W,C), "mesostatic" (N,C,H,W)

    Returns:
        pred_seq: (N, T_out, H, W, C_data)
        target_seq: (N, T_out, H, W, C_data)
        mask: (N, T_out, H, W, 1) — 1=invalid, 0=valid (Earthformer convention)
        in_seq: (N, T_in, H, W, C_data)
    """
    highresdynamic = batch["highresdynamic"].to(device, torch.float32)
    highresstatic = batch["highresstatic"].to(device, torch.float32)
    mesodynamic = batch["mesodynamic"].to(device, torch.float32)
    mesostatic = batch["mesostatic"].to(device, torch.float32)

    seq = highresdynamic[..., :data_channels]
    mask = highresdynamic[..., data_channels:data_channels + 1][:, in_len:in_len + out_len]

    in_seq = seq[:, :in_len]
    target_seq = seq[:, in_len:in_len + out_len]

    total_t = in_len + out_len
    mesodynamic_interp = rearrange(mesodynamic, "b t h w c -> b c t h w")
    mesodynamic_interp = F.interpolate(mesodynamic_interp, size=(total_t, img_height, img_width), mode="nearest")

    highresstatic_interp = rearrange(highresstatic, "b c h w -> b c 1 h w")
    highresstatic_interp = F.interpolate(highresstatic_interp, size=(total_t, img_height, img_width), mode="nearest")

    mesostatic_interp = rearrange(mesostatic, "b c h w -> b c 1 h w")
    mesostatic_interp = F.interpolate(mesostatic_interp, size=(total_t, img_height, img_width), mode="nearest")

    aux_data = torch.cat((highresstatic_interp, mesodynamic_interp, mesostatic_interp), dim=1)
    aux_data = rearrange(aux_data, "b c t h w -> b t h w c")

    pred_seq = model(in_seq, aux_data[:, :in_len], aux_data[:, in_len:in_len + out_len])
    return pred_seq, target_seq, mask, in_seq


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser("Earthformer Seasonal Matched-Pair Benchmark Eval")
    parser.add_argument("--cfg", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to earthformer .pt state_dict checkpoint")
    parser.add_argument("--earthnet_root", type=str, default=None,
                        help="Path to local EarthNet2021 root. Required when --data_csv is omitted.")
    parser.add_argument("--data_csv", type=str, default=None,
                        help="Optional per-window inference CSV. If omitted, it is generated from --benchmark_pairs_csv.")
    parser.add_argument("--benchmark_pairs_csv", type=str, required=True,
                        help="Path to seasonal_pairs_benchmark.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    if args.data_csv is None and args.earthnet_root is None:
        parser.error("--earthnet_root is required when --data_csv is omitted")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load config ----
    from train_cuboid_earthnet import CuboidEarthNet2021PLModule
    oc = CuboidEarthNet2021PLModule.get_base_config(CuboidEarthNet2021PLModule)

    # Try to recover config from PL checkpoint hyper_parameters if --cfg not given
    raw = torch.load(args.ckpt_path, map_location="cpu")
    if args.cfg is not None:
        oc = OmegaConf.merge(oc, OmegaConf.load(args.cfg))
    elif isinstance(raw, dict) and "hyper_parameters" in raw:
        hp = raw["hyper_parameters"]
        hp_oc = hp if isinstance(hp, OmegaConf) else OmegaConf.create(hp)
        oc = OmegaConf.merge(oc, hp_oc)
        print("[eval] WARNING: --cfg not provided; recovered config from checkpoint hyper_parameters")
    else:
        print("[eval] WARNING: --cfg not provided and checkpoint has no hyper_parameters. "
              "Using base config defaults — this will likely cause a state_dict mismatch!")

    in_len = oc.layout.in_len
    out_len = oc.layout.out_len
    img_height = oc.layout.img_height
    img_width = oc.layout.img_width
    data_channels = oc.model.data_channels

    print(f"[eval] Config: in_len={in_len}, out_len={out_len}, "
          f"img={img_height}x{img_width}, channels={data_channels}")

    # ---- Build model ----
    model = build_model_from_config(oc)
    if isinstance(raw, dict) and "state_dict" in raw:
        prefix = "torch_nn_module."
        state_dict = {k[len(prefix):]: v for k, v in raw["state_dict"].items()
                      if k.startswith(prefix)}
        print(f"[eval] Extracted state_dict from PL checkpoint (epoch={raw.get('epoch', '?')})")
    else:
        state_dict = raw
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"[eval] Model loaded from {args.ckpt_path}")

    # ---- Dataset ----
    dataset_cfg = OmegaConf.to_object(oc.dataset)
    dataset = SeasonalCSVDataset(
        csv_path=args.data_csv,
        pairs_csv=args.benchmark_pairs_csv,
        layout=dataset_cfg.get("layout", "THWC"),
        static_layout=dataset_cfg.get("static_layout", "CHW"),
        earthnet_root=args.earthnet_root,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn_seasonal,
    )
    dataset_source = args.data_csv if args.data_csv is not None else args.benchmark_pairs_csv
    print(f"[eval] Dataset: {len(dataset)} samples from {dataset_source}")

    # ---- Metrics ----
    ens_metric = EarthNet2021ScoreUpdateWithoutCompute(layout="NTHWC", eps=1e-4).to(device)

    # Per-window storage
    window_summaries: Dict[str, WindowSummary] = {}
    per_window_records: List[dict] = []
    sum_pixel_mae = 0.0
    sum_ndvi_mae = 0.0
    pixel_mae_count = 0
    ndvi_mae_count = 0

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    total_count = 0

    # ---- Eval loop ----
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="eval/seasonal_bench")):
            bsz = batch["highresdynamic"].shape[0]

            pred_seq, target_seq, mask_ef, in_seq = earthformer_forward(
                model, batch, in_len, out_len, img_height, img_width,
                data_channels=data_channels, device=device,
            )
            # pred_seq, target_seq: (N, T_out, H, W, C) — NTHWC
            # mask_ef: (N, T_out, H, W, 1) — 1=invalid, 0=valid (Earthformer convention)

            # Feed to EarthNetScore (expects NTHWC, mask=1 for invalid)
            ens_metric.update(pred_seq, target_seq, mask_ef)

            # Move to CPU for WindowSummary computation
            pred_cpu = pred_seq.cpu().float().clamp(0.0, 1.0)
            target_cpu = target_seq.cpu().float()
            mask_ef_cpu = mask_ef.cpu().float()

            for i in range(bsz):
                sample_id = str(batch["sample_id"][i])

                # Convert from THWC to CTHW for compute_window_summary
                pred_i_cthw = pred_cpu[i].permute(3, 0, 1, 2)       # (C, T_out, H, W)
                gt_i_cthw = target_cpu[i].permute(3, 0, 1, 2)       # (C, T_out, H, W)

                # Convert mask: Earthformer 1=invalid → WindowSummary 1=valid
                mask_ef_i = mask_ef_cpu[i, :, :, :, 0]              # (T_out, H, W)
                valid_mask_i = (1.0 - mask_ef_i).unsqueeze(0)       # (1, T_out, H, W)

                # Deterministic model: ensemble of 1
                pred_ensemble_i = pred_i_cthw.unsqueeze(0)  # (1, C, T_out, H, W)

                # Compute window summary
                ws, pred_mean_i = compute_window_summary(
                    pred_ensemble=pred_ensemble_i,
                    gt=gt_i_cthw,
                    valid_mask=valid_mask_i,
                    window_uid=sample_id,
                )
                window_summaries[sample_id] = ws

                # Accumulate per-window quality
                if np.isfinite(ws.pixel_mae):
                    sum_pixel_mae += ws.pixel_mae
                    pixel_mae_count += 1
                if np.isfinite(ws.ndvi_mae):
                    sum_ndvi_mae += ws.ndvi_mae
                    ndvi_mae_count += 1

                # Parse benchmark metadata for per-window CSV
                bmeta = None
                if "benchmark_meta_json" in batch:
                    try:
                        bmeta = parse_seasonal_benchmark_meta(batch["benchmark_meta_json"][i])
                    except Exception:
                        pass

                rec = {
                    "window_uid": sample_id,
                    "pixel_mae": ws.pixel_mae,
                    "ndvi_mae": ws.ndvi_mae,
                    "ndvi_gt_trajectory": json.dumps(ws.ndvi_gt_trajectory.tolist()),
                    "ndvi_pred_trajectory_mean": json.dumps(
                        _nanmean_axis0(ws.ndvi_pred_trajectory).tolist()
                    ),
                }
                if bmeta is not None:
                    rec["pair_id"] = bmeta.get("pair_id", "")
                    rec["sample_side"] = bmeta.get("sample_side", "")
                    rec["track_labels"] = bmeta.get("track_labels", "")
                per_window_records.append(rec)

                total_count += 1

    # ---- Compute results ----
    print(f"\n[eval] === Earthformer Seasonal Matched-Pair Benchmark ===")
    print(f"[eval] total windows processed: {total_count}")
    print(f"[eval] unique window summaries: {len(window_summaries)}")

    # Per-window quality summary
    quality = {"n_windows": total_count}
    if pixel_mae_count > 0:
        quality["pixel_mae"] = sum_pixel_mae / pixel_mae_count
    if ndvi_mae_count > 0:
        quality["ndvi_mae"] = sum_ndvi_mae / ndvi_mae_count
    print(f"[eval] Per-window quality: {quality}")

    # Pair metric computation
    pair_computer = PairMetricComputer(
        benchmark_pairs_csv=args.benchmark_pairs_csv,
        window_summaries=window_summaries,
    )
    overall_metrics, per_pair_records = pair_computer.compute_all()

    # Print summary dashboard.
    print(f"\n[eval] === Paired Metrics (Overall) ===")
    for k in ["DRR_mean", "DHR", "PDC_spearman",
              "per_window_pixel_mae", "per_window_ndvi_mae"]:
        v = overall_metrics.get(k, "N/A")
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # EarthNetScore
    ens_dict = ens_metric.compute()

    print(f"\n[eval] === EarthNetScore ===")
    for k, v in ens_dict.items():
        val = v.item() if isinstance(v, torch.Tensor) else float(v)
        print(f"  {k}: {val:.6f}")

    # Assemble results dict
    results = {
        "model": "earthformer",
        "num_windows": total_count,
        "num_ensemble": 1,
        "benchmark_pairs_csv": args.benchmark_pairs_csv,
        "per_window_quality": quality,
        "paired_metrics": overall_metrics,
        "earthnet_score": {
            k: (v.item() if isinstance(v, torch.Tensor) else float(v))
            for k, v in ens_dict.items()
        },
    }

    # Save metrics.json
    result_path = os.path.join(output_dir, "metrics.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, cls=_NaNSafeEncoder)
    print(f"\n[eval] Metrics saved: {result_path}")

    # Save per_pair_metrics.csv
    if per_pair_records:
        pair_csv_path = os.path.join(output_dir, "per_pair_metrics.csv")
        pd.DataFrame(per_pair_records).to_csv(pair_csv_path, index=False)
        print(f"[eval] Per-pair CSV saved: {pair_csv_path}")

    # Save per_window_metrics.csv (deduplicated by window_uid)
    if per_window_records:
        window_csv_path = os.path.join(output_dir, "per_window_metrics.csv")
        wdf = pd.DataFrame(per_window_records)
        wdf = wdf.drop_duplicates(subset="window_uid", keep="first")
        wdf.to_csv(window_csv_path, index=False)
        print(f"[eval] Per-window CSV saved: {window_csv_path} ({len(wdf)} unique windows)")


if __name__ == "__main__":
    main()
