"""
CUDA_VISIBLE_DEVICES=0 python scripts/analyze_voxel_purity.py \
  --dataset-root datasets/waymo \
  --output-dir output_voxel_purity \
  --camera-name FRONT \
  --voxel-sizes 0.1 0.2 0.4 0.8 \
  --max-scenes 4 \
  --max-views-per-scene 20 \
  --model-id depth-anything/DA3NESTED-GIANT-LARGE \
  --device cuda \
  --feat-mode last2_avg \
  --feat-dim-out 64 \
  --feature-chunk-size 10000 \
  --feature-device cpu \
  --round-digits 4
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

from depth_anything_3.api import DepthAnything3
from depth_anything_3.sparse_voxelizer import SparseVoxelizer



def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)





def round_floats(obj: Any, ndigits: int = 4) -> Any:
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, ndigits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(round_floats(v, ndigits) for v in obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return v
        return round(v, ndigits)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return obj
        return round(obj, ndigits)
    return obj


def format_float_str(v: Any, ndigits: int = 4) -> str:
    if v is None:
        return ""
    try:
        v = float(v)
    except Exception:
        return str(v)
    if math.isnan(v) or math.isinf(v):
        return str(v)
    return f"{v:.{ndigits}f}"


def aggregate_by_voxel_size(rows: List[Dict[str, Any]], ndigits: int = 4) -> List[Dict[str, Any]]:
    if len(rows) == 0:
        return []
    by_size: Dict[float, List[Dict[str, Any]]] = {}
    for row in rows:
        vs = float(row['voxel_size'])
        by_size.setdefault(vs, []).append(row)

    out: List[Dict[str, Any]] = []
    for vs in sorted(by_size.keys()):
        group = by_size[vs]
        item: Dict[str, Any] = {
            'voxel_size': vs,
            'num_scenes': len(group),
        }
        numeric_keys = sorted({k for r in group for k, v in r.items() if isinstance(v, (int, float, np.integer, np.floating)) and k not in ('voxel_size',)})
        for k in numeric_keys:
            vals = []
            for r in group:
                if k not in r:
                    continue
                v = r[k]
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isnan(fv) or math.isinf(fv):
                    continue
                vals.append(fv)
            if len(vals) == 0:
                continue
            item[f'avg_{k}'] = round(sum(vals) / len(vals), ndigits)
        out.append(item)
    return out
def discover_waymo_scenes(
    dataset_root: str,
    camera_name: str = "FRONT",
    exts=(".jpg", ".jpeg", ".png"),
) -> List[Dict[str, Any]]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")
    if not root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {dataset_root}")

    scenes: List[Dict[str, Any]] = []
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        cam_dir = scene_dir / camera_name
        if not cam_dir.exists() or not cam_dir.is_dir():
            continue

        image_paths: List[str] = []
        for ext in exts:
            image_paths.extend(sorted(str(p) for p in cam_dir.glob(f"*{ext}")))
        image_paths = sorted(image_paths)
        if len(image_paths) == 0:
            continue

        scenes.append(
            {
                "scene_name": scene_dir.name,
                "camera_name": camera_name,
                "image_dir": str(cam_dir),
                "image_paths": image_paths,
            }
        )

    if len(scenes) == 0:
        raise FileNotFoundError(
            f"No scenes found under {dataset_root} with camera folder '{camera_name}'"
        )
    return scenes



def _as_homogeneous_batch(E: torch.Tensor) -> torch.Tensor:
    if E.shape[-2:] == (4, 4):
        return E
    if E.shape[-2:] != (3, 4):
        raise ValueError(f"Unsupported extrinsics shape: {tuple(E.shape)}")
    n = E.shape[0]
    bottom = torch.tensor([0, 0, 0, 1], device=E.device, dtype=E.dtype).view(1, 1, 4).expand(n, 1, 4)
    return torch.cat([E, bottom], dim=1)



def _choose_feat_tokens(raw_feats: Any, feat_mode: str, device: torch.device) -> torch.Tensor:
    if feat_mode == "last":
        feat_tokens = raw_feats[3][0]
    elif feat_mode == "last2_avg":
        feat_tokens = 0.5 * (raw_feats[2][0] + raw_feats[3][0])
    elif feat_mode == "all4_avg":
        feat_tokens = sum(raw_feats[i][0] for i in range(4)) / 4.0
    else:
        raise ValueError(f"Unknown feat_mode: {feat_mode}")

    # original voxelizer expects [1, V, Ntok, C] then uses [0]
    # after that it wants [V, Ntok, C]
    feat_tokens = feat_tokens[0].to(device)
    return feat_tokens



def _unproject_vectorized(
    depth: torch.Tensor,
    K: torch.Tensor,
    E: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, h, w = depth.shape
    device = depth.device

    valid_coords = torch.nonzero(mask, as_tuple=False)
    if valid_coords.numel() == 0:
        empty_xyz = torch.empty((0, 3), device=device, dtype=depth.dtype)
        empty_idx = torch.empty((0,), device=device, dtype=torch.long)
        return empty_xyz, empty_idx, empty_idx, empty_idx

    view_ids = valid_coords[:, 0]
    ys = valid_coords[:, 1]
    xs = valid_coords[:, 2]

    z = depth[view_ids, ys, xs]
    homo_coords = torch.stack([xs.float(), ys.float(), torch.ones_like(xs, dtype=torch.float32)], dim=-1)

    inv_K = torch.inverse(K)
    inv_K_sel = inv_K[view_ids]
    points_cam = torch.bmm(inv_K_sel, homo_coords.unsqueeze(-1)).squeeze(-1)
    points_cam = points_cam * z.unsqueeze(-1)

    E_h = _as_homogeneous_batch(E)
    c2w = torch.inverse(E_h)
    c2w_sel = c2w[view_ids]
    R = c2w_sel[:, :3, :3]
    t = c2w_sel[:, :3, 3]
    points_world = torch.bmm(R, points_cam.unsqueeze(-1)).squeeze(-1) + t
    return points_world, view_ids, ys, xs



def _quantile_dict(x: torch.Tensor, name_prefix: str) -> Dict[str, float]:
    if x.numel() == 0:
        return {
            f"{name_prefix}_mean": float("nan"),
            f"{name_prefix}_median": float("nan"),
            f"{name_prefix}_p90": float("nan"),
            f"{name_prefix}_p95": float("nan"),
            f"{name_prefix}_max": float("nan"),
        }
    x = x.float()
    return {
        f"{name_prefix}_mean": float(x.mean().item()),
        f"{name_prefix}_median": float(x.median().item()),
        f"{name_prefix}_p90": float(torch.quantile(x, 0.90).item()),
        f"{name_prefix}_p95": float(torch.quantile(x, 0.95).item()),
        f"{name_prefix}_max": float(x.max().item()),
    }



def _histogram_png(values: np.ndarray, title: str, path: str, bins: int = 80) -> None:
    if not HAS_MPL or values.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()



def analyze_prediction_voxel_purity(
    prediction: Any,
    voxel_size: float,
    max_depth: float,
    conf_percentile: float,
    feat_mode: str,
    patch_size: int,
    feat_dim_out: Optional[int],
    device: torch.device,
    feature_chunk_size: int = 50000,
    feature_device: str = "cpu",
    skip_feature_metrics: bool = False,
) -> Dict[str, Any]:
    depth = torch.from_numpy(prediction.depth).to(device)
    intrinsics = torch.from_numpy(prediction.intrinsics).to(device)
    extrinsics = torch.from_numpy(prediction.extrinsics).to(device)
    images = torch.from_numpy(prediction.processed_images).to(device) if prediction.processed_images is not None else None
    conf = torch.from_numpy(prediction.conf).to(device) if prediction.conf is not None else None
    raw_feats = getattr(prediction, "raw_feats", None)
    if skip_feature_metrics:
        raw_feats = None

    mask = torch.isfinite(depth) & (depth > 0) & (depth < max_depth)
    if conf is not None:
        conf_thresh = torch.nanquantile(conf, conf_percentile / 100.0)
        mask &= torch.isfinite(conf)
        mask &= conf >= conf_thresh

    world_points, view_ids, ys, xs = _unproject_vectorized(depth, intrinsics, extrinsics, mask)
    if world_points.shape[0] == 0:
        return {
            "num_points": 0,
            "num_voxels": 0,
            "voxel_size": voxel_size,
        }

    voxel_coords = torch.floor(world_points / voxel_size).long()
    unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
    num_voxels = unique_voxels.shape[0]
    num_points = world_points.shape[0]

    one_long = torch.ones(num_points, device=device, dtype=torch.long)
    one_float = torch.ones(num_points, device=device, dtype=torch.float32)

    voxel_point_counts = torch.zeros(num_voxels, device=device, dtype=torch.long)
    voxel_point_counts.index_add_(0, inverse_indices, one_long)

    point_depth = depth[view_ids, ys, xs].float()
    depth_min = torch.full((num_voxels,), float("inf"), device=device, dtype=torch.float32)
    depth_max = torch.full((num_voxels,), float("-inf"), device=device, dtype=torch.float32)
    depth_min.scatter_reduce_(0, inverse_indices, point_depth, reduce="amin", include_self=True)
    depth_max.scatter_reduce_(0, inverse_indices, point_depth, reduce="amax", include_self=True)
    voxel_depth_range = depth_max - depth_min

    point_conf = conf[view_ids, ys, xs].float() if conf is not None else None
    voxel_confidence = None
    if point_conf is not None:
        voxel_conf_sum = torch.zeros(num_voxels, device=device, dtype=torch.float32)
        voxel_conf_sum.index_add_(0, inverse_indices, point_conf)
        voxel_confidence = voxel_conf_sum / voxel_point_counts.clamp_min(1).float()

    voxel_view_pairs = torch.stack([inverse_indices, view_ids], dim=1)
    unique_voxel_view_pairs = torch.unique(voxel_view_pairs, dim=0)
    voxel_view_counts = torch.zeros(num_voxels, device=device, dtype=torch.long)
    voxel_view_counts.index_add_(0, unique_voxel_view_pairs[:, 0], torch.ones(unique_voxel_view_pairs.shape[0], device=device, dtype=torch.long))

    voxel_color_std = None
    voxel_color_l2_mean = None
    if images is not None:
        pixel_colors = images.reshape(-1, 3)[mask.reshape(-1)].float()
        if pixel_colors.max() > 1.0:
            pixel_colors = pixel_colors / 255.0

        color_sum = torch.zeros((num_voxels, 3), device=device, dtype=torch.float32)
        color_sq_sum = torch.zeros((num_voxels, 3), device=device, dtype=torch.float32)
        color_sum.index_add_(0, inverse_indices, pixel_colors)
        color_sq_sum.index_add_(0, inverse_indices, pixel_colors * pixel_colors)
        voxel_color_mean = color_sum / voxel_point_counts.unsqueeze(-1).clamp_min(1)
        voxel_color_var = color_sq_sum / voxel_point_counts.unsqueeze(-1).clamp_min(1) - voxel_color_mean * voxel_color_mean
        voxel_color_var = torch.clamp(voxel_color_var, min=0.0)
        voxel_color_std = torch.sqrt(voxel_color_var.mean(dim=1))

        point_color_centered = pixel_colors - voxel_color_mean[inverse_indices]
        point_color_dist = torch.linalg.norm(point_color_centered, dim=1)
        color_dist_sum = torch.zeros(num_voxels, device=device, dtype=torch.float32)
        color_dist_sum.index_add_(0, inverse_indices, point_color_dist)
        voxel_color_l2_mean = color_dist_sum / voxel_point_counts.clamp_min(1).float()

    voxel_feat_l2_mean = None
    voxel_feat_cos_mean = None
    feat_dim = None
    if raw_feats is not None:
        feat_device_t = torch.device(feature_device)
        feat_tokens = _choose_feat_tokens(raw_feats, feat_mode=feat_mode, device=device)
        _, ntok, c = feat_tokens.shape
        if feat_dim_out is not None and feat_dim_out < c:
            feat_tokens = feat_tokens[..., :feat_dim_out]
        feat_tokens = feat_tokens.to(feat_device_t, dtype=torch.float16)
        feat_dim = int(feat_tokens.shape[-1])

        h, w = depth.shape[-2:]
        hf, wf = h // patch_size, w // patch_size
        expected_ntok = hf * wf
        if ntok != expected_ntok:
            raise ValueError(f"Ntok={ntok}, expected {expected_ntok} for image size {(h, w)} and patch={patch_size}")

        feat_sum = torch.zeros((num_voxels, feat_dim), device=device, dtype=torch.float32)
        for start in range(0, num_points, feature_chunk_size):
            end = min(start + feature_chunk_size, num_points)
            patch_y = (ys[start:end] // patch_size).to(feat_device_t)
            patch_x = (xs[start:end] // patch_size).to(feat_device_t)
            token_idx = patch_y * wf + patch_x
            view_chunk = view_ids[start:end].to(feat_device_t)
            point_feat_chunk = feat_tokens[view_chunk, token_idx].to(device=device, dtype=torch.float32)
            feat_sum.index_add_(0, inverse_indices[start:end], point_feat_chunk)
            del patch_y, patch_x, token_idx, view_chunk, point_feat_chunk
        feat_mean = feat_sum / voxel_point_counts.unsqueeze(-1).clamp_min(1).to(torch.float32)

        feat_l2_sum = torch.zeros(num_voxels, device=device, dtype=torch.float32)
        feat_cos_sum = torch.zeros(num_voxels, device=device, dtype=torch.float32)
        feat_mean_norm = F_normalize(feat_mean)
        for start in range(0, num_points, feature_chunk_size):
            end = min(start + feature_chunk_size, num_points)
            patch_y = (ys[start:end] // patch_size).to(feat_device_t)
            patch_x = (xs[start:end] // patch_size).to(feat_device_t)
            token_idx = patch_y * wf + patch_x
            view_chunk = view_ids[start:end].to(feat_device_t)
            point_feat_chunk = feat_tokens[view_chunk, token_idx].to(device=device, dtype=torch.float32)
            mean_chunk = feat_mean[inverse_indices[start:end]]
            mean_chunk_norm = feat_mean_norm[inverse_indices[start:end]]

            diff = point_feat_chunk - mean_chunk
            l2 = torch.linalg.norm(diff, dim=1)
            cos = (F_normalize(point_feat_chunk) * mean_chunk_norm).sum(dim=1)

            feat_l2_sum.index_add_(0, inverse_indices[start:end], l2)
            feat_cos_sum.index_add_(0, inverse_indices[start:end], cos)
            del patch_y, patch_x, token_idx, view_chunk, point_feat_chunk, mean_chunk, mean_chunk_norm, diff, l2, cos

        voxel_feat_l2_mean = feat_l2_sum / voxel_point_counts.clamp_min(1).float()
        voxel_feat_cos_mean = feat_cos_sum / voxel_point_counts.clamp_min(1).float()

    harmful_score = None
    components = []
    if voxel_color_std is not None:
        components.append(safe_zscore(voxel_color_std))
    if voxel_feat_l2_mean is not None:
        components.append(safe_zscore(voxel_feat_l2_mean))
    components.append(safe_zscore(voxel_depth_range))
    if len(components) > 0:
        harmful_score = torch.stack(components, dim=0).mean(dim=0)

    worst_k = min(20, num_voxels)
    if harmful_score is not None and worst_k > 0:
        worst_idx = torch.topk(harmful_score, k=worst_k, largest=True).indices
    else:
        worst_idx = torch.arange(min(20, num_voxels), device=device)

    summary: Dict[str, Any] = {
        "voxel_size": float(voxel_size),
        "num_points": int(num_points),
        "num_voxels": int(num_voxels),
        "avg_points_per_voxel": float(voxel_point_counts.float().mean().item()),
        "median_points_per_voxel": float(voxel_point_counts.float().median().item()),
        "max_points_per_voxel": int(voxel_point_counts.max().item()),
        "avg_views_per_voxel": float(voxel_view_counts.float().mean().item()),
        "median_views_per_voxel": float(voxel_view_counts.float().median().item()),
        "max_views_per_voxel": int(voxel_view_counts.max().item()),
        "feature_dim": feat_dim,
    }
    summary.update(_quantile_dict(voxel_depth_range, "depth_range"))
    if voxel_confidence is not None:
        summary.update(_quantile_dict(voxel_confidence, "voxel_confidence"))
    if voxel_color_std is not None:
        summary.update(_quantile_dict(voxel_color_std, "color_std"))
    if voxel_color_l2_mean is not None:
        summary.update(_quantile_dict(voxel_color_l2_mean, "color_l2_mean"))
    if voxel_feat_l2_mean is not None:
        summary.update(_quantile_dict(voxel_feat_l2_mean, "feat_l2_mean"))
    if voxel_feat_cos_mean is not None:
        summary.update(_quantile_dict(voxel_feat_cos_mean, "feat_cos_mean"))
    if harmful_score is not None:
        summary.update(_quantile_dict(harmful_score, "harmful_score"))

    worst_voxels: List[Dict[str, Any]] = []
    for idx in worst_idx.tolist():
        item: Dict[str, Any] = {
            "voxel_coord": unique_voxels[idx].detach().cpu().tolist(),
            "point_count": int(voxel_point_counts[idx].item()),
            "view_count": int(voxel_view_counts[idx].item()),
            "depth_range": float(voxel_depth_range[idx].item()),
        }
        if voxel_confidence is not None:
            item["confidence"] = float(voxel_confidence[idx].item())
        if voxel_color_std is not None:
            item["color_std"] = float(voxel_color_std[idx].item())
        if voxel_color_l2_mean is not None:
            item["color_l2_mean"] = float(voxel_color_l2_mean[idx].item())
        if voxel_feat_l2_mean is not None:
            item["feat_l2_mean"] = float(voxel_feat_l2_mean[idx].item())
        if voxel_feat_cos_mean is not None:
            item["feat_cos_mean"] = float(voxel_feat_cos_mean[idx].item())
        if harmful_score is not None:
            item["harmful_score"] = float(harmful_score[idx].item())
        worst_voxels.append(item)

    arrays: Dict[str, np.ndarray] = {
        "voxel_coords": unique_voxels.detach().cpu().numpy(),
        "point_counts": voxel_point_counts.detach().cpu().numpy(),
        "view_counts": voxel_view_counts.detach().cpu().numpy(),
        "depth_range": voxel_depth_range.detach().cpu().numpy(),
    }
    if voxel_confidence is not None:
        arrays["voxel_confidence"] = voxel_confidence.detach().cpu().numpy()
    if voxel_color_std is not None:
        arrays["color_std"] = voxel_color_std.detach().cpu().numpy()
    if voxel_color_l2_mean is not None:
        arrays["color_l2_mean"] = voxel_color_l2_mean.detach().cpu().numpy()
    if voxel_feat_l2_mean is not None:
        arrays["feat_l2_mean"] = voxel_feat_l2_mean.detach().cpu().numpy()
    if voxel_feat_cos_mean is not None:
        arrays["feat_cos_mean"] = voxel_feat_cos_mean.detach().cpu().numpy()
    if harmful_score is not None:
        arrays["harmful_score"] = harmful_score.detach().cpu().numpy()

    return {
        "summary": summary,
        "worst_voxels": worst_voxels,
        "arrays": arrays,
    }



def F_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)



def safe_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    std = x.std(unbiased=False)
    if torch.isnan(std) or std < eps:
        return torch.zeros_like(x)
    return (x - x.mean()) / std.clamp_min(eps)



def analyze_scene(
    model: DepthAnything3,
    scene: Dict[str, Any],
    voxel_sizes: Sequence[float],
    output_root: str,
    device: torch.device,
    max_depth: float,
    conf_percentile: float,
    feat_mode: str,
    patch_size: int,
    feat_dim_out: Optional[int],
    max_views_per_scene: Optional[int],
    export_dir: Optional[str] = None,
    feature_chunk_size: int = 50000,
    feature_device: str = "cpu",
    skip_feature_metrics: bool = False,
) -> List[Dict[str, Any]]:
    scene_name = scene["scene_name"]
    image_paths = scene["image_paths"]
    if max_views_per_scene is not None and len(image_paths) > max_views_per_scene:
        image_paths = image_paths[:max_views_per_scene]

    print(f"[INFO] Inference scene={scene_name} num_views={len(image_paths)}")
    with torch.no_grad():
        prediction = model.inference(
            image_paths,
            export_dir=export_dir,
            export_format="none",
        )

    scene_out_dir = os.path.join(output_root, scene_name)
    ensure_dir(scene_out_dir)

    scene_summaries: List[Dict[str, Any]] = []
    for voxel_size in voxel_sizes:
        print(f"  [INFO] Analyze voxel_size={voxel_size}")
        result = analyze_prediction_voxel_purity(
            prediction=prediction,
            voxel_size=voxel_size,
            max_depth=max_depth,
            conf_percentile=conf_percentile,
            feat_mode=feat_mode,
            patch_size=patch_size,
            feat_dim_out=feat_dim_out,
            device=device,
            feature_chunk_size=feature_chunk_size,
            feature_device=feature_device,
            skip_feature_metrics=skip_feature_metrics,
        )
        if "summary" not in result:
            summary = {
                "scene_name": scene_name,
                "voxel_size": voxel_size,
                **result,
            }
            scene_summaries.append(round_floats(summary, 4))
            continue

        tag = f"voxel_{str(voxel_size).replace('.', 'p')}"
        size_dir = os.path.join(scene_out_dir, tag)
        ensure_dir(size_dir)

        arrays = result.pop("arrays")
        summary = result["summary"]
        summary["scene_name"] = scene_name
        summary = round_floats(summary, 4)
        worst_voxels = round_floats(result["worst_voxels"], 4)
        save_json({"summary": summary, "worst_voxels": worst_voxels}, os.path.join(size_dir, "summary.json"))
        np.savez_compressed(os.path.join(size_dir, "per_voxel_metrics.npz"), **arrays)

        if "color_std" in arrays:
            _histogram_png(arrays["color_std"], f"{scene_name} color_std vs={voxel_size}", os.path.join(size_dir, "hist_color_std.png"))
        if "feat_l2_mean" in arrays:
            _histogram_png(arrays["feat_l2_mean"], f"{scene_name} feat_l2_mean vs={voxel_size}", os.path.join(size_dir, "hist_feat_l2.png"))
        if "feat_cos_mean" in arrays:
            _histogram_png(arrays["feat_cos_mean"], f"{scene_name} feat_cos_mean vs={voxel_size}", os.path.join(size_dir, "hist_feat_cos.png"))
        if "depth_range" in arrays:
            _histogram_png(arrays["depth_range"], f"{scene_name} depth_range vs={voxel_size}", os.path.join(size_dir, "hist_depth_range.png"))
        if "harmful_score" in arrays:
            _histogram_png(arrays["harmful_score"], f"{scene_name} harmful_score vs={voxel_size}", os.path.join(size_dir, "hist_harmful_score.png"))

        scene_summaries.append(summary)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return scene_summaries



def write_global_csv(rows: List[Dict[str, Any]], path: str, ndigits: int = 4) -> None:
    import csv

    if len(rows) == 0:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            cooked = {}
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, (float, np.floating)):
                    cooked[k] = format_float_str(v, ndigits)
                else:
                    cooked[k] = v
            writer.writerow(cooked)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--camera-name", type=str, default="FRONT")
    parser.add_argument("--model-id", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--voxel-sizes", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--max-views-per-scene", type=int, default=None)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--conf-percentile", type=float, default=30.0)
    parser.add_argument("--feat-mode", type=str, default="last2_avg", choices=["last", "last2_avg", "all4_avg"])
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--feat-dim-out", type=int, default=None)
    parser.add_argument("--feature-chunk-size", type=int, default=50000)
    parser.add_argument("--feature-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-feature-metrics", action="store_true")
    parser.add_argument("--export-dir", type=str, default=None)
    parser.add_argument("--round-digits", type=int, default=4)
    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    scenes = discover_waymo_scenes(args.dataset_root, camera_name=args.camera_name)
    if args.scene_names:
        wanted = set(args.scene_names)
        scenes = [s for s in scenes if s["scene_name"] in wanted]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    if len(scenes) == 0:
        raise RuntimeError("No scenes selected for analysis.")

    print(f"[INFO] Selected {len(scenes)} scenes")
    model = DepthAnything3.from_pretrained(args.model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    all_rows: List[Dict[str, Any]] = []
    for scene in scenes:
        rows = analyze_scene(
            model=model,
            scene=scene,
            voxel_sizes=args.voxel_sizes,
            output_root=args.output_dir,
            device=device,
            max_depth=args.max_depth,
            conf_percentile=args.conf_percentile,
            feat_mode=args.feat_mode,
            patch_size=args.patch_size,
            feat_dim_out=args.feat_dim_out,
            max_views_per_scene=args.max_views_per_scene,
            export_dir=args.export_dir,
            feature_chunk_size=args.feature_chunk_size,
            feature_device=args.feature_device,
            skip_feature_metrics=args.skip_feature_metrics,
        )
        all_rows.extend(rows)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_rows = round_floats(all_rows, args.round_digits)
    avg_rows = aggregate_by_voxel_size(all_rows, ndigits=args.round_digits)

    save_json({"rows": all_rows}, os.path.join(args.output_dir, "all_scene_summary.json"))
    write_global_csv(all_rows, os.path.join(args.output_dir, "all_scene_summary.csv"), ndigits=args.round_digits)

    save_json({"rows": avg_rows}, os.path.join(args.output_dir, "average_by_voxel_size.json"))
    write_global_csv(avg_rows, os.path.join(args.output_dir, "average_by_voxel_size.csv"), ndigits=args.round_digits)

    print(f"[DONE] Wrote summary to {os.path.join(args.output_dir, 'all_scene_summary.csv')}")
    print(f"[DONE] Wrote voxel-size averages to {os.path.join(args.output_dir, 'average_by_voxel_size.csv')}")



if __name__ == "__main__":
    main()
