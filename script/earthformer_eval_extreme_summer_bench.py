#!/usr/bin/env python3
"""
Earthformer evaluation on Extreme Summer Benchmark.

Loads a trained Earthformer checkpoint and evaluates it on the EO-WM Extreme
Summer benchmark CSV.

Metrics computed:
  - EarthNetScore (MAD, OLS, EMD, SSIM) via EarthNet2021ScoreUpdateWithoutCompute
  - Pixel MAE, NDVI MAE, trough NDVI MAE, and drop amplitude error

Usage:
    export EARTHFORMER_REPO=/path/to/earth-forecasting-transformer

    python script/earthformer_eval_extreme_summer_bench.py \
        --cfg cfg.yaml \
        --ckpt_path /path/to/earthformer_earthnet2021.pt \
        --earthnet_root /path/to/EarthNet2021 \
        --data_csv benchmark_csv/extreme_summer_benchmark.csv \
        --output_dir ./eval_extreme_output \
        --batch_size 4
"""

import argparse
import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
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
# Benchmark Metadata Parsing
# =====================================================================

def parse_benchmark_meta(meta_json_str: str) -> dict:
    """Parse benchmark_meta_json string into a dict with typed fields."""
    raw = json.loads(meta_json_str)

    def _float(key, default=None):
        v = raw.get(key, default)
        if v is None:
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def _int(key, default=None):
        v = raw.get(key, default)
        if v is None:
            return default
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return default

    return {
        "sample_id": raw.get("sample_id", "unknown"),
        "drop_amplitude": _float("drop_amplitude", 0.0),
        "baseline_ndvi": _float("baseline_ndvi", 0.0),
        "t_trough_absolute": _int("t_trough_absolute"),
        "slice_start_idx": _int("slice_start_idx", 0),
        "in_len": 10,
    }


# =====================================================================
# ExtremeMetricAccumulator
# =====================================================================

class ExtremeMetricAccumulator:
    """Accumulates per-sample metrics for extreme summer benchmark evaluation."""

    def __init__(self):
        self.per_sample_records: list[dict] = []
        self.sum_pixel_mae = 0.0
        self.pixel_count = 0.0
        self.sum_ndvi_mae = 0.0
        self.ndvi_pixel_count = 0.0
        self.sum_trough_ndvi_mae = 0.0
        self.trough_ndvi_count = 0.0
        self.sum_drop_amp_error = 0.0
        self.drop_amp_count = 0
        self.n_samples = 0

    def update_sample(
        self,
        pred_ensemble: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor,
        benchmark_meta: dict,
        in_s2: torch.Tensor = None,
        in_valid: torch.Tensor = None,
    ):
        """Process one sample's predictions and accumulate metrics.

        Args:
            pred_ensemble: (N, C, T_out, H, W) — N=1 for deterministic model.
            gt: (C, T_out, H, W) ground truth.
            valid_mask: (1, T_out, H, W) validity mask, 1=valid.
            benchmark_meta: parsed benchmark metadata dict.
        """
        self.n_samples += 1
        pred_mean = pred_ensemble.mean(dim=0)

        record = {}
        if benchmark_meta is not None:
            record["sample_id"] = benchmark_meta.get("sample_id", "unknown")
            record["drop_amplitude"] = benchmark_meta.get("drop_amplitude", None)
        else:
            record["sample_id"] = f"sample_{self.n_samples - 1}"
            record["drop_amplitude"] = None

        record.update(self._compute_metrics(pred_mean, gt, valid_mask, benchmark_meta))

        self.per_sample_records.append(record)

    def _compute_metrics(self, pred_mean, gt, valid_mask, benchmark_meta):
        record = {}
        mask = valid_mask
        diff = (pred_mean - gt) * mask
        n_valid = mask.sum() * pred_mean.shape[0]
        if n_valid > 0:
            sample_mae = diff.abs().sum().item() / n_valid.item()
        else:
            sample_mae = float("nan")
        record["pixel_mae"] = sample_mae
        if not math.isnan(sample_mae):
            self.sum_pixel_mae += sample_mae * n_valid.item()
            self.pixel_count += n_valid.item()

        # NDVI MAE
        pred_ndvi = compute_ndvi(pred_mean)
        gt_ndvi = compute_ndvi(gt)
        mask_squeezed = mask.squeeze(0)
        gt_veg_mask = compute_vegetation_mask(gt_ndvi)
        combined_mask = mask_squeezed * gt_veg_mask.float()

        ndvi_diff = (pred_ndvi - gt_ndvi) * combined_mask
        n_ndvi_valid = combined_mask.sum()
        if n_ndvi_valid > 0:
            sample_ndvi_mae = ndvi_diff.abs().sum().item() / n_ndvi_valid.item()
        else:
            sample_ndvi_mae = float("nan")
        record["ndvi_mae"] = sample_ndvi_mae
        if not math.isnan(sample_ndvi_mae):
            self.sum_ndvi_mae += sample_ndvi_mae * n_ndvi_valid.item()
            self.ndvi_pixel_count += n_ndvi_valid.item()

        # Trough NDVI MAE
        trough_ndvi_mae = float("nan")
        if benchmark_meta is not None:
            t_trough_abs = benchmark_meta.get("t_trough_absolute")
            slice_start = benchmark_meta.get("slice_start_idx", 0)
            in_len = benchmark_meta.get("in_len", 10)
            if t_trough_abs is not None:
                t_trough_rel = int(t_trough_abs) - int(slice_start) - int(in_len)
                T_out = pred_ndvi.shape[0]
                if 0 <= t_trough_rel < T_out:
                    trough_mask = combined_mask[t_trough_rel]
                    trough_diff = (pred_ndvi[t_trough_rel] - gt_ndvi[t_trough_rel]) * trough_mask
                    n_trough = trough_mask.sum()
                    if n_trough > 0:
                        trough_ndvi_mae = trough_diff.abs().sum().item() / n_trough.item()
                        self.sum_trough_ndvi_mae += trough_ndvi_mae * n_trough.item()
                        self.trough_ndvi_count += n_trough.item()
        record["trough_ndvi_mae"] = trough_ndvi_mae

        # Drop Amplitude Error
        drop_amp_error = float("nan")
        if benchmark_meta is not None:
            gt_drop = benchmark_meta.get("drop_amplitude", None)
            baseline_ndvi = benchmark_meta.get("baseline_ndvi", None)
            if gt_drop is not None and baseline_ndvi is not None:
                per_frame_ndvi_means = []
                for t_idx in range(pred_ndvi.shape[0]):
                    frame_mask = combined_mask[t_idx]
                    n_pix = frame_mask.sum()
                    if n_pix > 0:
                        mean_ndvi = (pred_ndvi[t_idx] * frame_mask).sum().item() / n_pix.item()
                        per_frame_ndvi_means.append(mean_ndvi)
                if len(per_frame_ndvi_means) > 0:
                    pred_min_target_ndvi = min(per_frame_ndvi_means)
                    pred_drop = baseline_ndvi - pred_min_target_ndvi
                    drop_amp_error = abs(pred_drop - gt_drop)
                    self.sum_drop_amp_error += drop_amp_error
                    self.drop_amp_count += 1
        record["drop_amplitude_error"] = drop_amp_error
        return record

    def compute_summary(self, all_records=None):
        summary = {"n_samples": self.n_samples}
        if self.pixel_count > 0:
            summary["pixel_mae"] = self.sum_pixel_mae / self.pixel_count
        if self.ndvi_pixel_count > 0:
            summary["ndvi_mae"] = self.sum_ndvi_mae / self.ndvi_pixel_count
        if self.trough_ndvi_count > 0:
            summary["trough_ndvi_mae"] = self.sum_trough_ndvi_mae / self.trough_ndvi_count
        if self.drop_amp_count > 0:
            summary["drop_amplitude_error"] = self.sum_drop_amp_error / self.drop_amp_count

        records = all_records if all_records is not None else self.per_sample_records
        summary["by_extreme_bin"] = self._compute_bin_metrics(records)
        return summary

    def _compute_bin_metrics(self, records=None):
        records = records if records is not None else self.per_sample_records
        if not records:
            return {}
        scored = [r for r in records if r.get("drop_amplitude") is not None]
        if not scored:
            return {}
        scores = [r["drop_amplitude"] for r in scored]
        low_thr = np.percentile(scores, 33.3)
        high_thr = np.percentile(scores, 66.7)
        bins = {"low": [], "mid": [], "high": []}
        for r in scored:
            s = r["drop_amplitude"]
            if s <= low_thr:
                bins["low"].append(r)
            elif s <= high_thr:
                bins["mid"].append(r)
            else:
                bins["high"].append(r)
        result = {}
        for bin_name, bin_records in bins.items():
            if not bin_records:
                continue
            bin_summary = {"n_samples": len(bin_records)}
            for metric_key in ["trough_ndvi_mae", "drop_amplitude_error"]:
                vals = [r[metric_key] for r in bin_records
                        if metric_key in r and not (isinstance(r[metric_key], float) and math.isnan(r[metric_key]))]
                if vals:
                    bin_summary[metric_key] = sum(vals) / len(vals)
            result[bin_name] = bin_summary
        return result

    def get_per_sample_records(self):
        return self.per_sample_records


# =====================================================================
# Visualization
# =====================================================================

def _s2_to_rgb(frame_hwc: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """Convert a single (H, W, C>=4) EarthNet frame [B,G,R,NIR,...] to RGB uint8."""
    rgb = frame_hwc[:, :, [2, 1, 0]]  # R, G, B
    rgb = np.clip(rgb / 0.3, 0, 1)
    rgb = np.power(rgb, gamma)
    return (rgb * 255).astype(np.uint8)


def _render_row(frames_thwc: np.ndarray, max_frames: int = 20, gap: int = 2) -> np.ndarray:
    """Render a temporal sequence as a horizontal strip of RGB frames.

    Args:
        frames_thwc: (T, H, W, C) numpy array
        max_frames: show at most this many (evenly sampled)
        gap: pixel gap between frames
    Returns:
        (H, W_total, 3) uint8 image
    """
    T = frames_thwc.shape[0]
    if T <= max_frames:
        indices = np.arange(T)
    else:
        indices = np.linspace(0, T - 1, max_frames, dtype=int)

    rgbs = [_s2_to_rgb(frames_thwc[t]) for t in indices]
    if gap > 0 and len(rgbs) > 1:
        h = rgbs[0].shape[0]
        sep = np.full((h, gap, 3), 24, dtype=np.uint8)
        parts = [rgbs[0]]
        for img in rgbs[1:]:
            parts.extend([sep, img])
        return np.concatenate(parts, axis=1)
    return np.concatenate(rgbs, axis=1)


def save_vis_sample(save_path: str, in_seq: np.ndarray, gt_seq: np.ndarray, pred_seq: np.ndarray):
    """Save a 3-row visualization: Input / GT / Pred.

    Args:
        save_path: output png path
        in_seq: (T_in, H, W, C) input context
        gt_seq: (T_out, H, W, C) ground truth target
        pred_seq: (T_out, H, W, C) model prediction
    """
    row_in = _render_row(in_seq, max_frames=10)
    row_gt = _render_row(gt_seq, max_frames=20)
    row_pred = _render_row(pred_seq, max_frames=20)

    # Pad narrower rows to max width
    max_w = max(row_in.shape[1], row_gt.shape[1], row_pred.shape[1])

    def _pad(img, w):
        if img.shape[1] >= w:
            return img
        pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
        return np.concatenate([img, pad], axis=1)

    row_in = _pad(row_in, max_w)
    row_gt = _pad(row_gt, max_w)
    row_pred = _pad(row_pred, max_w)

    # Add row labels
    label_w = 120
    rows_labeled = []
    for label, row_img in [("Input (10)", row_in), ("GT (20)", row_gt), ("Pred (20)", row_pred)]:
        h = row_img.shape[0]
        canvas = np.zeros((h, label_w + row_img.shape[1], 3), dtype=np.uint8)
        canvas[:, :label_w] = 18
        canvas[:, label_w:] = row_img
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        draw.text((6, max(2, (h - 12) // 2)), label, fill=(235, 235, 235))
        rows_labeled.append(np.array(pil))

    # Divider between rows
    divider = np.full((3, rows_labeled[0].shape[1], 3), 80, dtype=np.uint8)
    parts = [rows_labeled[0], divider, rows_labeled[1], divider, rows_labeled[2]]
    overview = np.concatenate(parts, axis=0)
    Image.fromarray(overview).save(save_path)


# =====================================================================
# Custom Dataset: CSV-driven NPZ loading for Extreme Summer Benchmark
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


class ExtremeCSVDataset(Dataset):
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
        csv_path: str,
        layout: str = "THWC",
        static_layout: str = "CHW",
        earthnet_root=None,
    ):
        self.df = pd.read_csv(csv_path)
        self.layout = layout
        self.static_layout = static_layout
        self.earthnet_root = earthnet_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        context_path = resolve_earthnet_path(row["path"], self.earthnet_root, "extreme_test_split")
        target_path = resolve_earthnet_path(row["path_target"], self.earthnet_root, "extreme_test_split")
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
        # Convert to target layout (default THWC for Earthformer dataset config)
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
        """Change layout using einops-style permutation for HWCT<->THWC etc."""
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


def collate_fn_extreme(batch):
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
# Model Builder
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
# Forward Pass
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
    parser = argparse.ArgumentParser("Earthformer Extreme Summer Benchmark Eval")
    parser.add_argument("--cfg", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to earthformer .pt state_dict checkpoint")
    parser.add_argument("--earthnet_root", type=str, default=None,
                        help="Path to local EarthNet2021 root. Required when CSV paths are relative or from another machine.")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Path to extreme_summer_benchmark CSV")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vis_samples", type=int, default=10,
                        help="Number of samples to save visualization for")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Force CUDA initialization early
    if device.type == "cuda":
        print(f"[eval] Initializing CUDA device {device} ...", flush=True)
        torch.cuda.set_device(device)
        torch.cuda.init()
        _dummy = torch.zeros(1, device=device)
        del _dummy
        print(f"[eval] CUDA initialized, device={torch.cuda.get_device_name(device)}, "
              f"total_mem={torch.cuda.get_device_properties(device).total_memory/1e9:.1f} GB", flush=True)

    # ---- Load config ----
    from train_cuboid_earthnet import CuboidEarthNet2021PLModule
    oc = CuboidEarthNet2021PLModule.get_base_config(CuboidEarthNet2021PLModule)
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(args.cfg)
        oc = OmegaConf.merge(oc, oc_from_file)

    in_len = oc.layout.in_len
    out_len = oc.layout.out_len
    img_height = oc.layout.img_height
    img_width = oc.layout.img_width
    data_channels = oc.model.data_channels

    print(f"[eval] Config: in_len={in_len}, out_len={out_len}, "
          f"img={img_height}x{img_width}, channels={data_channels}", flush=True)

    # ---- Build model ----
    print("[eval] Building model ...", flush=True)
    model = build_model_from_config(oc)
    print("[eval] Loading checkpoint ...", flush=True)
    raw = torch.load(args.ckpt_path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        prefix = "torch_nn_module."
        state_dict = {k[len(prefix):]: v for k, v in raw["state_dict"].items()
                      if k.startswith(prefix)}
        print(f"[eval] Extracted state_dict from PL checkpoint (epoch={raw.get('epoch', '?')})", flush=True)
    else:
        state_dict = raw
    print("[eval] load_state_dict ...", flush=True)
    model.load_state_dict(state_dict)
    del raw, state_dict
    torch.cuda.empty_cache()
    print(f"[eval] model.to({device}) ...", flush=True)
    model = model.to(device).eval()
    print(f"[eval] Model loaded, GPU mem={torch.cuda.memory_allocated(device)/1e9:.2f} GB", flush=True)

    # ---- Dataset ----
    dataset_cfg = OmegaConf.to_object(oc.dataset)
    dataset = ExtremeCSVDataset(
        csv_path=args.data_csv,
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
        collate_fn=collate_fn_extreme,
    )
    print(f"[eval] Dataset: {len(dataset)} samples from {args.data_csv}", flush=True)

    # ---- Metrics ----
    ens_metric = EarthNet2021ScoreUpdateWithoutCompute(layout="NTHWC", eps=1e-4).to(device)
    accumulator = ExtremeMetricAccumulator()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    total_count = 0
    vis_count = 0
    vis_dir = os.path.join(output_dir, "vis")

    # ---- Eval loop ----
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="eval/extreme_summer")):
            bsz = batch["highresdynamic"].shape[0]

            pred_seq, target_seq, mask_ef, in_seq = earthformer_forward(
                model, batch, in_len, out_len, img_height, img_width,
                data_channels=data_channels, device=device,
            )
            # pred_seq, target_seq: (N, T_out, H, W, C) — NTHWC
            # mask_ef: (N, T_out, H, W, 1) — 1=invalid, 0=valid (Earthformer convention)

            # Feed to EarthNetScore (expects NTHWC, mask=1 for invalid)
            ens_metric.update(pred_seq, target_seq, mask_ef)

            # Move to CPU for ExtremeMetricAccumulator
            pred_cpu = pred_seq.cpu().float().clamp(0.0, 1.0)
            target_cpu = target_seq.cpu().float()
            mask_ef_cpu = mask_ef.cpu().float()
            in_cpu = in_seq.cpu().float()

            for i in range(bsz):
                # Convert from THWC to CTHW for metric accumulation.
                pred_i_cthw = pred_cpu[i].permute(3, 0, 1, 2)       # (C, T_out, H, W)
                gt_i_cthw = target_cpu[i].permute(3, 0, 1, 2)       # (C, T_out, H, W)
                in_i_cthw = in_cpu[i].permute(3, 0, 1, 2)           # (C, T_in, H, W)

                # Convert mask: Earthformer 1=invalid → accumulator 1=valid
                mask_ef_i = mask_ef_cpu[i, :, :, :, 0]              # (T_out, H, W)
                valid_mask_i = (1.0 - mask_ef_i).unsqueeze(0)       # (1, T_out, H, W)

                # Deterministic model: ensemble of 1
                pred_ensemble_i = pred_i_cthw.unsqueeze(0)  # (1, C, T_out, H, W)

                # Parse benchmark metadata
                bmeta = None
                if "benchmark_meta_json" in batch:
                    bmeta = parse_benchmark_meta(batch["benchmark_meta_json"][i])

                accumulator.update_sample(
                    pred_ensemble=pred_ensemble_i,
                    gt=gt_i_cthw,
                    valid_mask=valid_mask_i,
                    benchmark_meta=bmeta,
                    in_s2=in_i_cthw,
                )
                total_count += 1

                # Visualization (first N samples)
                if vis_count < args.vis_samples:
                    os.makedirs(vis_dir, exist_ok=True)
                    save_vis_sample(
                        os.path.join(vis_dir, f"sample_{vis_count:04d}.png"),
                        in_seq=in_cpu[i].clamp(0, 1).numpy(),      # (T_in, H, W, C)
                        gt_seq=target_cpu[i].clamp(0, 1).numpy(),   # (T_out, H, W, C)
                        pred_seq=pred_cpu[i].numpy(),                # (T_out, H, W, C)
                    )
                    vis_count += 1

    # ---- Compute results ----
    extreme_summary = accumulator.compute_summary()
    ens_dict = ens_metric.compute()

    results = {
        "model": "earthformer",
        "num_samples": total_count,
        "num_ensemble": 1,
    }
    for k, v in ens_dict.items():
        results[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
    results["extreme_metrics"] = extreme_summary

    # Print
    print(f"\n[eval] === Earthformer Extreme Summer Benchmark Results ===")
    print(f"[eval] samples={total_count}")
    print("[eval] EarthNetScore metrics:")
    for k, v in ens_dict.items():
        val = v.item() if isinstance(v, torch.Tensor) else float(v)
        print(f"  {k}: {val:.6f}")
    print("[eval] Extreme benchmark metrics:")
    for k, v in extreme_summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    # Save
    result_path = os.path.join(output_dir, "metrics.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[eval] Metrics saved: {result_path}")
    if vis_count > 0:
        print(f"[eval] Visualizations saved: {vis_dir}/ ({vis_count} samples)")

    per_sample_records = accumulator.get_per_sample_records()
    if per_sample_records:
        csv_path = os.path.join(output_dir, "per_sample_metrics.csv")
        df = pd.DataFrame(per_sample_records)
        df.to_csv(csv_path, index=False)
        print(f"[eval] Per-sample CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
