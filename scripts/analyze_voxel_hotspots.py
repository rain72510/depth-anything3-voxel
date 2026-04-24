"""

CUDA_VISIBLE_DEVICES=0 python scripts/analyze_voxel_hotspots.py \
  --dataset-root datasets/waymo \
  --output-dir output_voxel_hotspots_glb \
  --camera-name FRONT \
  --voxel-sizes 0.2 0.4 \
  --max-scenes 4 \
  --max-views-per-scene 20 \
  --model-id depth-anything/DA3NESTED-GIANT-LARGE \
  --device cuda \
  --top-percent 1.0 \
  --round-digits 4

CUDA_VISIBLE_DEVICES=0 python scripts/analyze_voxel_hotspots.py \
  --dataset-root datasets/waymo \
  --output-dir output_voxel_hotspots_glb \
  --scene-names 10017090168044687777_6380_000_6400_000 \
  --voxel-sizes 0.2 0.4 \
  --max-views-per-scene 20 \
  --model-id depth-anything/DA3NESTED-GIANT-LARGE \
  --device cuda

"""


import os
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh

from depth_anything_3.api import DepthAnything3

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


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


def aggregate_by_voxel_size(rows: List[Dict[str, Any]], ndigits: int = 4) -> List[Dict[str, Any]]:
    if len(rows) == 0:
        return []
    by_size: Dict[float, List[Dict[str, Any]]] = {}
    for row in rows:
        vs = float(row["voxel_size"])
        by_size.setdefault(vs, []).append(row)

    out: List[Dict[str, Any]] = []
    for vs in sorted(by_size.keys()):
        group = by_size[vs]
        item: Dict[str, Any] = {
            "voxel_size": vs,
            "num_scenes": len(group),
        }
        numeric_keys = sorted(
            {
                k
                for r in group
                for k, v in r.items()
                if isinstance(v, (int, float, np.integer, np.floating)) and k not in ("voxel_size",)
            }
        )
        for k in numeric_keys:
            vals = []
            for r in group:
                if k not in r:
                    continue
                try:
                    fv = float(r[k])
                except Exception:
                    continue
                if math.isnan(fv) or math.isinf(fv):
                    continue
                vals.append(fv)
            if vals:
                item[f"avg_{k}"] = round(sum(vals) / len(vals), ndigits)
        out.append(item)
    return out


def discover_waymo_scenes(
    dataset_root: str,
    camera_name: str = "FRONT",
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
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


def _unproject_vectorized(
    depth: torch.Tensor,
    K: torch.Tensor,
    E: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _compute_color_metrics(
    num_voxels: int,
    inverse_indices: torch.Tensor,
    pixel_colors: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    device = pixel_colors.device
    counts = torch.zeros(num_voxels, device=device, dtype=torch.float32)
    counts.index_add_(0, inverse_indices, torch.ones(pixel_colors.shape[0], device=device, dtype=torch.float32))

    color_sum = torch.zeros((num_voxels, 3), device=device, dtype=torch.float32)
    color_sum.index_add_(0, inverse_indices, pixel_colors)
    color_mean = color_sum / counts.unsqueeze(-1).clamp_min(1.0)

    diff = pixel_colors - color_mean[inverse_indices]
    diff_sq = diff * diff
    sq_sum = torch.zeros((num_voxels, 3), device=device, dtype=torch.float32)
    sq_sum.index_add_(0, inverse_indices, diff_sq)
    color_var = sq_sum / counts.unsqueeze(-1).clamp_min(1.0)
    color_std_per_channel = torch.sqrt(torch.clamp(color_var, min=0.0))
    color_std = color_std_per_channel.mean(dim=1)

    color_l2 = torch.linalg.norm(diff, dim=1)
    color_l2_sum = torch.zeros(num_voxels, device=device, dtype=torch.float32)
    color_l2_sum.index_add_(0, inverse_indices, color_l2)
    color_l2_mean = color_l2_sum / counts.clamp_min(1.0)

    return {
        "counts": counts,
        "color_mean": color_mean,
        "color_std": color_std,
        "color_l2_mean": color_l2_mean,
    }


def _compute_depth_range(
    num_voxels: int,
    inverse_indices: torch.Tensor,
    point_depth: torch.Tensor,
) -> torch.Tensor:
    dev = point_depth.device
    depth_min = torch.full((num_voxels,), float("inf"), device=dev, dtype=torch.float32)
    depth_max = torch.full((num_voxels,), float("-inf"), device=dev, dtype=torch.float32)
    depth_min.scatter_reduce_(0, inverse_indices, point_depth.float(), reduce="amin", include_self=True)
    depth_max.scatter_reduce_(0, inverse_indices, point_depth.float(), reduce="amax", include_self=True)
    return torch.where(torch.isfinite(depth_min) & torch.isfinite(depth_max), depth_max - depth_min, torch.zeros_like(depth_min))


def _quantile_dict(x: torch.Tensor, name_prefix: str) -> Dict[str, float]:
    if x.numel() == 0:
        return {
            f"{name_prefix}_mean": float("nan"),
            f"{name_prefix}_median": float("nan"),
            f"{name_prefix}_p90": float("nan"),
            f"{name_prefix}_p95": float("nan"),
            f"{name_prefix}_max": float("nan"),
        }
    xf = x.float()
    return {
        f"{name_prefix}_mean": float(xf.mean().item()),
        f"{name_prefix}_median": float(torch.quantile(xf, 0.5).item()),
        f"{name_prefix}_p90": float(torch.quantile(xf, 0.9).item()),
        f"{name_prefix}_p95": float(torch.quantile(xf, 0.95).item()),
        f"{name_prefix}_max": float(xf.max().item()),
    }


def transform_points_for_glb(points_xyz: np.ndarray, mode: str = "zup_to_yup") -> np.ndarray:
    """
    Convert point cloud coordinates for GLB viewers.

    Common case here:
    - internal/world coordinates behave like Z-up
    - many GLB viewers expect Y-up

    mode:
      - "none": no transform
      - "zup_to_yup": rotate -90 deg around X, (x, y, z) -> (x, z, -y)
      - "zup_to_yup_flip": rotate +90 deg around X, (x, y, z) -> (x, -z, y)
    """
    if mode == "none":
        return points_xyz

    xyz = np.asarray(points_xyz, dtype=np.float32)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    if mode == "zup_to_yup":
        return np.stack([x, z, -y], axis=1)
    if mode == "zup_to_yup_flip":
        return np.stack([x, -z, y], axis=1)

    raise ValueError(f"Unknown GLB transform mode: {mode}")


def export_point_cloud(
    points_xyz: np.ndarray,
    colors_rgba: np.ndarray,
    out_path: str,
    glb_transform: str = "zup_to_yup",
) -> None:
    verts = points_xyz
    if out_path.lower().endswith(".glb"):
        verts = transform_points_for_glb(points_xyz, mode=glb_transform)
    pc = trimesh.points.PointCloud(vertices=verts, colors=colors_rgba)
    pc.export(out_path)


def make_overlay_colors(
    point_colors_rgb: np.ndarray,
    inverse_indices: np.ndarray,
    top_std_mask: np.ndarray,
    top_l2_mask: np.ndarray,
    intersection_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    n = point_colors_rgb.shape[0]
    base = np.concatenate([point_colors_rgb.copy(), np.full((n, 1), 255, dtype=np.uint8)], axis=1)

    std_overlay = base.copy()
    std_overlay[top_std_mask[inverse_indices]] = np.array([255, 0, 0, 255], dtype=np.uint8)

    l2_overlay = base.copy()
    l2_overlay[top_l2_mask[inverse_indices]] = np.array([0, 80, 255, 255], dtype=np.uint8)

    inter_overlay = base.copy()
    inter_overlay[intersection_mask[inverse_indices]] = np.array([255, 220, 0, 255], dtype=np.uint8)

    combined = base.copy()
    combined[top_std_mask[inverse_indices]] = np.array([255, 0, 0, 255], dtype=np.uint8)
    combined[top_l2_mask[inverse_indices]] = np.array([0, 80, 255, 255], dtype=np.uint8)
    combined[intersection_mask[inverse_indices]] = np.array([255, 220, 0, 255], dtype=np.uint8)
    return {
        "std": std_overlay,
        "l2": l2_overlay,
        "intersection": inter_overlay,
        "combined": combined,
    }


def save_hotspot_plot(
    voxel_centers: np.ndarray,
    top_std_mask: np.ndarray,
    top_l2_mask: np.ndarray,
    out_path: str,
    max_bg_points: int = 120000,
) -> None:
    if not HAS_MPL:
        return
    n = voxel_centers.shape[0]
    bg_idx = np.arange(n)
    if n > max_bg_points:
        bg_idx = np.random.default_rng(0).choice(n, size=max_bg_points, replace=False)

    inter = top_std_mask & top_l2_mask
    std_only = top_std_mask & (~inter)
    l2_only = top_l2_mask & (~inter)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    bg = voxel_centers[bg_idx]
    ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=0.2, c="lightgray", alpha=0.15)

    for mask, color, label, size in [
        (std_only, "red", "top color_std", 3.5),
        (l2_only, "royalblue", "top color_l2_mean", 3.5),
        (inter, "gold", "intersection", 5.0),
    ]:
        pts = voxel_centers[mask]
        if len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=color, alpha=0.95, label=label)

    ax.legend(loc="upper right")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def analyze_scene(
    model: DepthAnything3,
    scene: Dict[str, Any],
    output_root: str,
    voxel_sizes: List[float],
    max_views_per_scene: Optional[int],
    max_depth: float,
    conf_percentile: float,
    top_percent: float,
    round_digits: int,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    image_paths = scene["image_paths"]
    if max_views_per_scene is not None and len(image_paths) > max_views_per_scene:
        image_paths = image_paths[:max_views_per_scene]

    print(f"[INFO] Inference scene={scene['scene_name']} num_views={len(image_paths)}")
    with torch.no_grad():
        prediction = model.inference(image_paths, export_format="none")

    depth = torch.from_numpy(prediction.depth).to(device)
    intrinsics = torch.from_numpy(prediction.intrinsics).to(device)
    extrinsics = torch.from_numpy(prediction.extrinsics).to(device)
    images = torch.from_numpy(prediction.processed_images).to(device) if prediction.processed_images is not None else None
    conf = torch.from_numpy(prediction.conf).to(device) if prediction.conf is not None else None

    mask = torch.isfinite(depth) & (depth > 0) & (depth < max_depth)
    if conf is not None:
        conf_thresh = torch.nanquantile(conf, conf_percentile / 100.0)
        mask &= torch.isfinite(conf)
        mask &= conf >= conf_thresh

    world_points, view_ids, ys, xs = _unproject_vectorized(depth, intrinsics, extrinsics, mask)
    if world_points.shape[0] == 0:
        print(f"[WARN] scene={scene['scene_name']} has zero valid points")
        return []

    point_colors = images.reshape(-1, 3)[mask.reshape(-1)].float()
    if point_colors.max() > 1.0:
        point_colors = point_colors / 255.0
    point_depth = depth[view_ids, ys, xs].float()

    rows: List[Dict[str, Any]] = []
    scene_root = os.path.join(output_root, scene["scene_name"])
    ensure_dir(scene_root)

    world_points_np = world_points.detach().cpu().numpy().astype(np.float32)
    point_colors_np = (point_colors.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)

    for voxel_size in voxel_sizes:
        print(f"  [INFO] Export hotspots voxel_size={voxel_size}")
        voxel_coords = torch.floor(world_points / voxel_size).long()
        unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)
        num_voxels = int(unique_voxels.shape[0])
        if num_voxels == 0:
            continue

        counts = torch.zeros(num_voxels, device=device, dtype=torch.float32)
        counts.index_add_(0, inverse_indices, torch.ones(world_points.shape[0], device=device, dtype=torch.float32))
        point_sum = torch.zeros((num_voxels, 3), device=device, dtype=torch.float32)
        point_sum.index_add_(0, inverse_indices, world_points.float())
        voxel_centers = point_sum / counts.unsqueeze(-1).clamp_min(1.0)

        metrics = _compute_color_metrics(num_voxels, inverse_indices, point_colors)
        color_std = metrics["color_std"]
        color_l2_mean = metrics["color_l2_mean"]
        color_mean = metrics["color_mean"]
        depth_range = _compute_depth_range(num_voxels, inverse_indices, point_depth)

        k = max(1, int(math.ceil(num_voxels * (top_percent / 100.0))))
        top_std_idx = torch.topk(color_std, k=k, largest=True).indices
        top_l2_idx = torch.topk(color_l2_mean, k=k, largest=True).indices
        top_std_mask = torch.zeros(num_voxels, device=device, dtype=torch.bool)
        top_l2_mask = torch.zeros(num_voxels, device=device, dtype=torch.bool)
        top_std_mask[top_std_idx] = True
        top_l2_mask[top_l2_idx] = True
        intersection_mask = top_std_mask & top_l2_mask

        voxel_dir = os.path.join(scene_root, f"voxel_{str(voxel_size).replace('.', 'p')}")
        ensure_dir(voxel_dir)

        voxel_centers_np = voxel_centers.detach().cpu().numpy().astype(np.float32)
        color_mean_np = (color_mean.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
        color_mean_rgba = np.concatenate([color_mean_np, np.full((num_voxels, 1), 255, dtype=np.uint8)], axis=1)

        top_std_mask_np = top_std_mask.detach().cpu().numpy()
        top_l2_mask_np = top_l2_mask.detach().cpu().numpy()
        inter_mask_np = intersection_mask.detach().cpu().numpy()
        inverse_np = inverse_indices.detach().cpu().numpy()

        overlay_colors = make_overlay_colors(point_colors_np, inverse_np, top_std_mask_np, top_l2_mask_np, inter_mask_np)

        # original point cloud
        export_point_cloud(world_points_np, np.concatenate([point_colors_np, np.full((point_colors_np.shape[0], 1), 255, dtype=np.uint8)], axis=1), os.path.join(voxel_dir, "original_points.ply"), glb_transform=args.glb_transform)
        export_point_cloud(world_points_np, np.concatenate([point_colors_np, np.full((point_colors_np.shape[0], 1), 255, dtype=np.uint8)], axis=1), os.path.join(voxel_dir, "original_points.glb"), glb_transform=args.glb_transform)

        # point cloud overlays at point level
        export_point_cloud(world_points_np, overlay_colors["std"], os.path.join(voxel_dir, "original_points_top_color_std_overlay.ply"), glb_transform=args.glb_transform)
        export_point_cloud(world_points_np, overlay_colors["std"], os.path.join(voxel_dir, "original_points_top_color_std_overlay.glb"), glb_transform=args.glb_transform)
        export_point_cloud(world_points_np, overlay_colors["l2"], os.path.join(voxel_dir, "original_points_top_color_l2_overlay.ply"), glb_transform=args.glb_transform)
        export_point_cloud(world_points_np, overlay_colors["l2"], os.path.join(voxel_dir, "original_points_top_color_l2_overlay.glb"), glb_transform=args.glb_transform)
        export_point_cloud(world_points_np, overlay_colors["combined"], os.path.join(voxel_dir, "original_points_hotspots_overlay.ply"), glb_transform=args.glb_transform)
        export_point_cloud(world_points_np, overlay_colors["combined"], os.path.join(voxel_dir, "original_points_hotspots_overlay.glb"), glb_transform=args.glb_transform)

        # voxel-center clouds
        export_point_cloud(voxel_centers_np, color_mean_rgba, os.path.join(voxel_dir, "voxel_centers_mean_color.ply"), glb_transform=args.glb_transform)
        export_point_cloud(voxel_centers_np, color_mean_rgba, os.path.join(voxel_dir, "voxel_centers_mean_color.glb"), glb_transform=args.glb_transform)

        if top_std_mask_np.any():
            export_point_cloud(voxel_centers_np[top_std_mask_np], np.tile(np.array([[255, 0, 0, 255]], dtype=np.uint8), (int(top_std_mask_np.sum()), 1)), os.path.join(voxel_dir, "top_color_std_centers.ply"), glb_transform=args.glb_transform)
            export_point_cloud(voxel_centers_np[top_std_mask_np], np.tile(np.array([[255, 0, 0, 255]], dtype=np.uint8), (int(top_std_mask_np.sum()), 1)), os.path.join(voxel_dir, "top_color_std_centers.glb"), glb_transform=args.glb_transform)
        if top_l2_mask_np.any():
            export_point_cloud(voxel_centers_np[top_l2_mask_np], np.tile(np.array([[0, 80, 255, 255]], dtype=np.uint8), (int(top_l2_mask_np.sum()), 1)), os.path.join(voxel_dir, "top_color_l2_centers.ply"), glb_transform=args.glb_transform)
            export_point_cloud(voxel_centers_np[top_l2_mask_np], np.tile(np.array([[0, 80, 255, 255]], dtype=np.uint8), (int(top_l2_mask_np.sum()), 1)), os.path.join(voxel_dir, "top_color_l2_centers.glb"), glb_transform=args.glb_transform)
        if inter_mask_np.any():
            export_point_cloud(voxel_centers_np[inter_mask_np], np.tile(np.array([[255, 220, 0, 255]], dtype=np.uint8), (int(inter_mask_np.sum()), 1)), os.path.join(voxel_dir, "top_intersection_centers.ply"), glb_transform=args.glb_transform)
            export_point_cloud(voxel_centers_np[inter_mask_np], np.tile(np.array([[255, 220, 0, 255]], dtype=np.uint8), (int(inter_mask_np.sum()), 1)), os.path.join(voxel_dir, "top_intersection_centers.glb"), glb_transform=args.glb_transform)

        save_hotspot_plot(
            voxel_centers=voxel_centers_np,
            top_std_mask=top_std_mask_np,
            top_l2_mask=top_l2_mask_np,
            out_path=os.path.join(voxel_dir, "top_hotspots_3d.png"),
        )

        np.savez_compressed(
            os.path.join(voxel_dir, "per_voxel_metrics.npz"),
            voxel_centers=voxel_centers_np,
            voxel_mean_color=color_mean_np,
            counts=counts.detach().cpu().numpy(),
            color_std=color_std.detach().cpu().numpy(),
            color_l2_mean=color_l2_mean.detach().cpu().numpy(),
            depth_range=depth_range.detach().cpu().numpy(),
            top_std_mask=top_std_mask_np,
            top_l2_mask=top_l2_mask_np,
            intersection_mask=inter_mask_np,
        )

        row = {
            "scene_name": scene["scene_name"],
            "voxel_size": float(voxel_size),
            "num_points": int(world_points.shape[0]),
            "num_voxels": num_voxels,
            "avg_points_per_voxel": float(counts.mean().item()),
            "median_points_per_voxel": float(torch.quantile(counts, 0.5).item()),
            "top_k_voxels": int(k),
            "num_intersection_voxels": int(intersection_mask.sum().item()),
            **_quantile_dict(color_std, "color_std"),
            **_quantile_dict(color_l2_mean, "color_l2_mean"),
            **_quantile_dict(depth_range, "depth_range"),
        }
        save_json(round_floats(row, round_digits), os.path.join(voxel_dir, "summary.json"))
        rows.append(row)

        del voxel_coords, unique_voxels, inverse_indices, counts, point_sum, voxel_centers
        del metrics, color_std, color_l2_mean, color_mean, depth_range
        del top_std_idx, top_l2_idx, top_std_mask, top_l2_mask, intersection_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--camera-name", type=str, default="FRONT")
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-views-per-scene", type=int, default=20)
    parser.add_argument("--model-id", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--voxel-sizes", nargs="+", type=float, default=[0.2, 0.4])
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--conf-percentile", type=float, default=30.0)
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--round-digits", type=int, default=4)
    parser.add_argument(
        "--glb-transform",
        type=str,
        default="zup_to_yup",
        choices=["none", "zup_to_yup", "zup_to_yup_flip"],
        help="Coordinate transform applied only when exporting .glb files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    scenes = discover_waymo_scenes(args.dataset_root, camera_name=args.camera_name)
    if args.scene_names:
        wanted = set(args.scene_names)
        scenes = [s for s in scenes if s["scene_name"] in wanted]
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    print(f"[INFO] Selected {len(scenes)} scenes")

    model = DepthAnything3.from_pretrained(args.model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    all_rows: List[Dict[str, Any]] = []
    for scene in scenes:
        scene_rows = analyze_scene(
            model=model,
            scene=scene,
            output_root=args.output_dir,
            voxel_sizes=args.voxel_sizes,
            max_views_per_scene=args.max_views_per_scene,
            max_depth=args.max_depth,
            conf_percentile=args.conf_percentile,
            top_percent=args.top_percent,
            round_digits=args.round_digits,
            device=device,
            args=args,
        )
        all_rows.extend(scene_rows)

    write_global_csv(all_rows, os.path.join(args.output_dir, "all_scene_summary.csv"), ndigits=args.round_digits)
    save_json(round_floats({"rows": all_rows}, args.round_digits), os.path.join(args.output_dir, "all_scene_summary.json"))

    by_size = aggregate_by_voxel_size(all_rows, ndigits=args.round_digits)
    write_global_csv(by_size, os.path.join(args.output_dir, "average_by_voxel_size.csv"), ndigits=args.round_digits)
    save_json(round_floats({"rows": by_size}, args.round_digits), os.path.join(args.output_dir, "average_by_voxel_size.json"))
    print("[INFO] Done")


if __name__ == "__main__":
    main()
