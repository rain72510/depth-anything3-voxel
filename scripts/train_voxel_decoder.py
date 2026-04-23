import os
import glob
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import psutil
import wandb
import traceback
import time

from depth_anything_3.api import DepthAnything3

# currently no wrapper
# 你自己的 voxelizer 與 decoder
from depth_anything_3.sparse_voxelizer import SparseVoxelizer
from depth_anything_3.model.voxel_gaussian_decoder import VoxelGaussianDecoder

from types import SimpleNamespace
from PIL import Image

from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode, render_3dgs
from depth_anything_3.specs import Gaussians
from depth_anything_3.utils.gsply_helpers import export_ply
from depth_anything_3.utils.loss_utils import ssim

from PIL import Image
import math

def tensor_to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """
    x: [3,H,W], float in [0,1]
    return: uint8 [H,W,3]
    """
    x = x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return x
def save_tensor_image(x: torch.Tensor, path: str):
    Image.fromarray(tensor_to_uint8_image(x)).save(path)
def save_diff_image(pred: torch.Tensor, gt: torch.Tensor, path: str, amplify: float = 4.0):
    """
    pred, gt: [3,H,W], float in [0,1]
    """
    diff = (pred - gt).abs().mean(dim=0, keepdim=True)  # [1,H,W]
    diff = (diff * amplify).clamp(0, 1).repeat(3, 1, 1)
    save_tensor_image(diff, path)
def select_valid_view_indices(num_views: int, requested: list[int]) -> list[int]:
    out = []
    for idx in requested:
        if 0 <= idx < num_views:
            out.append(idx)
    if len(out) == 0 and num_views > 0:
        out = [0]
    return out


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def get_cpu_mem_stats():
    vm = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    return {
        "system_ram_used_gb": vm.used / (1024 ** 3),
        "system_ram_available_gb": vm.available / (1024 ** 3),
        "system_ram_percent": vm.percent,
        "process_ram_gb": process.memory_info().rss / (1024 ** 3),
    }

def get_gpu_mem_stats(device: torch.device):
    stats = {}
    if torch.cuda.is_available() and device.type == "cuda":
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        stats = {
            "gpu_mem_allocated_gb": torch.cuda.memory_allocated(dev_idx) / (1024 ** 3),
            "gpu_mem_reserved_gb": torch.cuda.memory_reserved(dev_idx) / (1024 ** 3),
            "gpu_mem_max_allocated_gb": torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 3),
            "gpu_mem_max_reserved_gb": torch.cuda.max_memory_reserved(dev_idx) / (1024 ** 3),
        }
    return stats

def make_wandb_image_triplet(pred: torch.Tensor, gt: torch.Tensor, caption: str = ""):
    """
    pred, gt: [3,H,W], float in [0,1]
    """
    diff = (pred - gt).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1).clamp(0, 1)

    pred_np = tensor_to_uint8_image(pred)
    gt_np = tensor_to_uint8_image(gt)
    diff_np = tensor_to_uint8_image(diff)

    return [
        wandb.Image(gt_np, caption=f"{caption} | gt"),
        wandb.Image(pred_np, caption=f"{caption} | pred"),
        wandb.Image(diff_np, caption=f"{caption} | diff"),
    ]

def make_timestamped_output_dir(base_output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(final_dir, exist_ok=True)
    return final_dir

def save_checkpoint(
    decoder,
    optimizer,
    epoch,
    output_dir,
    keep_last_k=5,
):
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pth")

    torch.save({
        "epoch": epoch,
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, ckpt_path)

    print(f"[INFO] Saved checkpoint: {ckpt_path}")

    # 刪舊的
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_epoch_*.pth")))
    if len(ckpts) > keep_last_k:
        for old_ckpt in ckpts[:-keep_last_k]:
            os.remove(old_ckpt)

def discover_waymo_scenes(
    dataset_root: str,
    camera_name: str = "FRONT",
    exts=(".jpg", ".jpeg", ".png"),
):
    """
    Scan dataset structure like:

    dataset_root/
      scene_001/FRONT/*.jpg
      scene_002/FRONT/*.jpg
      ...

    Returns:
        scenes: list[dict]
            [
                {
                    "scene_name": "...",
                    "camera_name": "FRONT",
                    "image_dir": ".../scene_xxx/FRONT",
                    "image_paths": [...]
                },
                ...
            ]
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")
    if not root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {dataset_root}")

    scenes = []

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue

        cam_dir = scene_dir / camera_name
        if not cam_dir.exists() or not cam_dir.is_dir():
            continue

        image_paths = []
        for ext in exts:
            image_paths.extend(sorted(cam_dir.glob(f"*{ext}")))

        image_paths = sorted(str(p) for p in image_paths)

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
            f"No scenes found under {dataset_root} with camera folder '{camera_name}' "
            f"and extensions {exts}"
        )

    return scenes


def build_decoder_inputs(voxel_dict: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Minimal decoder inputs:
      - anchor_xyz      <- voxel_mean_points
      - dino_feat       <- voxel_features
      - confidence      <- voxel_confidence if exists, else fallback
      - cov_diag        <- voxel_var_points

    Fallback strategy if voxel_confidence does not exist:
      use normalized voxel_view_counts
    """
    anchor_xyz = voxel_dict["voxel_mean_points"].to(device).float()
    dino_feat = voxel_dict["voxel_features"].to(device).float()
    cov_diag = voxel_dict["voxel_var_points"].to(device).float()

    # if dino_feat is None:
    #     raise ValueError("voxel_features is None. Please ensure raw_feats are available and voxel feature aggregation is enabled.")

    if "voxel_confidence" in voxel_dict and voxel_dict["voxel_confidence"] is not None:
        confidence = voxel_dict["voxel_confidence"].to(device).float()
    else:
        # fallback: normalize view counts into [0,1]
        v = voxel_dict["voxel_view_counts"].to(device).float()
        confidence = v / v.max().clamp_min(1.0)

    voxel_colors = None
    if "voxel_colors" in voxel_dict and voxel_dict["voxel_colors"] is not None:
        voxel_colors = voxel_dict["voxel_colors"].to(device).float()

        if voxel_colors.max() > 1.0:
            voxel_colors = voxel_colors / 255.0

    return {
        "anchor_xyz": anchor_xyz,   # [K, 3]
        "dino_feat": dino_feat,     # [K, C]
        "confidence": confidence,   # [K]
        "cov_diag": cov_diag,       # [K, 3]
        "voxel_colors": voxel_colors,
    }


def flatten_gaussians(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert [N, K, ...] -> [N*K, ...]
    """
    flat = {
        "means3D": out["centers"].reshape(-1, 3),
        "scales": out["scales"].reshape(-1, 3),
        "rotations": out["quaternions"].reshape(-1, 4),
        "opacity": out["opacity"].reshape(-1, 1),
        "colors": out["colors"].reshape(-1, 3),
    }
    if "gaussian_feature" in out:
        flat["gaussian_feature"] = out["gaussian_feature"].reshape(-1, out["gaussian_feature"].shape[-1])
    return flat


def save_gaussian_scene_npz(scene: Dict[str, torch.Tensor], path: str):
    np.savez_compressed(
        path,
        means3D=scene["means3D"].detach().cpu().numpy(),
        scales=scene["scales"].detach().cpu().numpy(),
        rotations=scene["rotations"].detach().cpu().numpy(),
        opacity=scene["opacity"].detach().cpu().numpy(),
        colors=scene["colors"].detach().cpu().numpy(),
    )

def masked_l1_loss(pred, gt, valid_mask, eps=1e-8):
    # pred, gt: [B,3,H,W]
    # valid_mask: [B,H,W] or [B,1,H,W], True=keep
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    valid_mask = valid_mask.float()

    diff = (pred - gt).abs() * valid_mask
    denom = valid_mask.sum() * pred.shape[1]
    return diff.sum() / denom.clamp_min(eps)

def compute_photometric_loss(
    decoder_out: Dict[str, torch.Tensor],
    voxel_dict: Dict[str, Any],
    rendered_rgb: torch.Tensor,   # [v,3,H,W]
    gt_rgb: torch.Tensor,         # [v,3,H,W]
    valid_mask=None,   # [v,H,W], True=non-sky
    lambda_photo: float = 1.0,
    lambda_color: float = 0.05,
    lambda_offset: float = 1e-2,
    lambda_scale: float = 1e-1,
    lambda_opacity: float = 1e-2,
    lambda_anchor: float = 1e-4,
    lambda_disp: float = 1e-3,
    # lambda_dssim: float = 0.02,
    lambda_dssim: float = 0,
    lambda_scale_vol: float = 1e-2,
) -> Dict[str, torch.Tensor]:
    losses = {}

    # 1. photometric supervision
    # print(f"rendered_rgb: shape={rendered_rgb.shape}, dtype={rendered_rgb.dtype}, device={rendered_rgb.device}")
    # print(f"gt_rgb: shape={gt_rgb.shape}, dtype={gt_rgb.dtype}, device={gt_rgb.device}")
    # print(f"rendered_rgb range: [{rendered_rgb.min().item():.4f}, {rendered_rgb.max().item():.4f}]"
    #       f", gt_rgb range: [{gt_rgb.min().item():.4f}, {gt_rgb.max().item():.4f}]")

    # print dim of rendered_rgb and gt_rgb for debugging
    # print(f"rendered_rgb: shape={rendered_rgb.shape}, dtype={rendered_rgb.dtype}, device={rendered_rgb.device}")
    # print(f"gt_rgb: shape={gt_rgb.shape}, dtype={gt_rgb.dtype}, device={gt_rgb.device}")

    # print the first few pixel values of rendered_rgb and gt_rgb for debugging
    # print(f"rendered_rgb sample: {rendered_rgb.view(3, -1)[:, :5]}")
    # print(f"gt_rgb sample: {gt_rgb.view(3, -1)[:, :5]}")

    # print shape, dtype, device, min, max of rendered_rgb and gt_rgb for debugging
    # print(f"rendered_rgb: shape={rendered_rgb.shape}, dtype={rendered_rgb.dtype}, device={rendered_rgb.device}, "
    #       f"min={rendered_rgb.min().item():.4f}, max={rendered_rgb.max().item():.4f}")
    # print(f"gt_rgb: shape={gt_rgb.shape}, dtype={gt_rgb.dtype}, device={gt_rgb.device}, "
    #       f"min={gt_rgb.min().item():.4f}, max={gt_rgb.max().item():.4f}")

    if valid_mask is None:
        losses["photo"] = lambda_photo * F.l1_loss(rendered_rgb, gt_rgb)
    else:
        losses["photo"] = lambda_photo * masked_l1_loss(rendered_rgb, gt_rgb, valid_mask)
    # losses["photo"] = lambda_photo * F.l1_loss(rendered_rgb, gt_rgb)

    # for k, v in decoder_out.items():
    #     print(f"decoder_out[{k}]: shape={v.shape}, dtype={v.dtype}, device={v.device}, "
    #           f"min={v.min().item():.4f}, max={v.max().item():.4f}")
    
    # decoder_out["colors"] shape
    # print(f"decoder_out['colors']: shape={decoder_out['colors'].shape}, dtype={decoder_out['colors'].dtype}, device={decoder_out['colors'].device}, "
    #       f"min={decoder_out['colors'].min().item():.4f}, max={decoder_out['colors'].max().item():.4f}")

    # # 2. optional voxel color prior
    if voxel_dict.get("voxel_colors", None) is not None:
        target_color = voxel_dict["voxel_colors"].to(decoder_out["colors"].device).float()/255.0
        pred_color = decoder_out["colors"].mean(dim=1)
        # print dim, dtype, device, min, max of pred_color and target_color for debugging
        # print(f"pred_color: shape={pred_color.shape}, dtype={pred_color.dtype}, device={pred_color.device}, "
        #       f"min={pred_color.min().item():.4f}, max={pred_color.max().item():.4f}")
        # print(f"target_color: shape={target_color.shape}, dtype={target_color.dtype}, device={target_color.device}, "
        #       f"min={target_color.min().item():.4f}, max={target_color.max().item():.4f}")

        losses["color"] = lambda_color * F.l1_loss(pred_color, target_color)
    else:
        losses["color"] = lambda_color * torch.tensor(0.0, device=decoder_out["colors"].device)

    # # 3. geometry / regularization
    losses["offset_reg"] = lambda_offset * decoder_out["offsets"].pow(2).mean()
    # print(f"offset_reg: {losses['offset_reg'].item():.6f}")
    losses["scale_reg"] = lambda_scale * decoder_out["scales"].pow(2).mean()
    losses["scale_vol_reg"] = lambda_scale_vol * decoder_out["scales"].prod(dim=1).mean()
    losses["opacity_reg"] = lambda_opacity * decoder_out["opacity"].mean()
    # losses["anchor_scale_reg"] = decoder_out["anchor_scale"].pow(2).mean()

    # # actual displacement regularization
    # disp = decoder_out["offsets"] * decoder_out["anchor_scale"].unsqueeze(-1)
    # losses["disp_reg"] = disp.pow(2).mean()
    

    ssim_val = ssim(rendered_rgb, gt_rgb)
    dssim = (1.0 - ssim_val) / 2.0
    losses["dssim"] = lambda_dssim * dssim

    # print all loss components for debugging
    # for k, v in losses.items():
    #     print(f"{k} loss: {v.item():.6f}")

    losses["total"] = (
        losses["photo"]
        + losses["color"]
        + losses["offset_reg"]
        + losses["scale_reg"]
        + losses["scale_vol_reg"]
        + losses["opacity_reg"]
        # + losses["anchor_scale_reg"]
        # + losses["disp_reg"]
        + losses["dssim"]

    )
    return losses

def build_renderer_gaussians(flat_scene: Dict[str, torch.Tensor]):
    """
    flat_scene:
        means3D   [M,3]
        scales    [M,3]
        rotations [M,4]
        opacity   [M,1] or [M]
        colors    [M,3]
    """

    means = flat_scene["means3D"]
    scales = flat_scene["scales"]
    rotations = flat_scene["rotations"]
    opacities = flat_scene["opacity"]
    colors = flat_scene["colors"]

    if opacities.ndim == 2 and opacities.shape[-1] == 1:
        opacities = opacities.squeeze(-1)   # -> [M]

    # M = means.shape[0]
    # device = means.device
    # dtype = means.dtype
    
    # harmonics: [1, M, 3, 9]
    # use RGB as SH DC term only

    # harmonics = torch.zeros((1, M, 3, 9), device=device, dtype=dtype)
    # harmonics[0, :, :, 0] = colors

    gaussian = SimpleNamespace(
        means=means.unsqueeze(0),          # [1, M, 3]
        scales=scales.unsqueeze(0),        # [1, M, 3]
        rotations=rotations.unsqueeze(0),  # [1, M, 4]
        opacities=opacities.unsqueeze(0),  # [1, M]
        # harmonics=harmonics,               # [1, M, 3, 9]
        harmonics=colors.unsqueeze(0),              # [1, M, 3]
    )

    # for k, v in gaussian.__dict__.items():
    #     print(f"gaussian.{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}, "
    #           f"min={v.min().item():.4f}, max={v.max().item():.4f}")
    return gaussian

def save_flat_scene_as_ply(
    flat_scene: dict,
    save_path: str,
    save_sh_dc_only: bool = True,
    shift_and_scale: bool = False,
):
    """
    flat_scene:
        means3D   [M,3]
        scales    [M,3]
        rotations [M,4]
        opacity   [M,1] or [M]
        colors    [M,3]   in [0,1]
    """
    means = flat_scene["means3D"].detach()
    scales = flat_scene["scales"].detach().clamp_min(1e-8)
    rotations = flat_scene["rotations"].detach()
    opacities = flat_scene["opacity"].detach().reshape(-1).clamp(1e-6, 1 - 1e-6)
    colors = flat_scene["colors"].detach().clamp(0.0, 1.0)

    # SH degree 0 only: [M,3,1]
    harmonics = colors.unsqueeze(-1)

    export_ply(
        means=means,
        scales=scales,
        rotations=rotations,
        harmonics=harmonics,
        opacities=opacities,
        path=Path(save_path),
        shift_and_scale=shift_and_scale,
        save_sh_dc_only=save_sh_dc_only,
        match_3dgs_mcmc_dev=False,
    )

def save_debug_ply_pair(
    flat_scene: dict,
    voxel_mean_points,  # [K, 3] tensor or None
    output_dir: str,
    global_step: int,
    scene_name: str,
    keep_last_k: int = 20,
):
    """Save Gaussian PLY + voxel XYZ PLY for a logged render_samples step."""
    import glob as _glob
    debug_dir = os.path.join(output_dir, "debug_ply")
    os.makedirs(debug_dir, exist_ok=True)

    prefix = f"step_{global_step:07d}_{scene_name}"

    # 3DGS PLY
    gs_path = os.path.join(debug_dir, prefix + "_gaussians.ply")
    save_flat_scene_as_ply(flat_scene, gs_path)

    # Voxel XYZ PLY
    if voxel_mean_points is not None:
        from plyfile import PlyData, PlyElement
        import numpy as np
        pts = voxel_mean_points.float().numpy()
        el = PlyElement.describe(
            np.array([(pts[i, 0], pts[i, 1], pts[i, 2]) for i in range(len(pts))],
                     dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]),
            "vertex",
        )
        vox_path = os.path.join(debug_dir, prefix + "_voxels.ply")
        PlyData([el]).write(vox_path)

    # Keep only last K pairs
    all_gs = sorted(_glob.glob(os.path.join(debug_dir, "step_*_gaussians.ply")), key=os.path.getmtime)
    if len(all_gs) > keep_last_k:
        for old_path in all_gs[:-keep_last_k]:
            os.remove(old_path)
            vox_old = old_path.replace("_gaussians.ply", "_voxels.ply")
            if os.path.exists(vox_old):
                os.remove(vox_old)


def save_recent_training_ply(
    flat_scene: dict,
    output_dir: str,
    epoch: int,
    scene_name: str,
    keep_last_k: int = 10,
):
    ply_root = os.path.join(output_dir, "train_ply")
    os.makedirs(ply_root, exist_ok=True)

    save_path = os.path.join(
        ply_root,
        f"epoch_{epoch:04d}_{scene_name}.ply"
    )
    save_flat_scene_as_ply(flat_scene, save_path)

    all_ply = sorted(
        glob.glob(os.path.join(ply_root, "epoch_*.ply")),
        key=os.path.getmtime
    )
    if len(all_ply) > keep_last_k:
        for old_path in all_ply[:-keep_last_k]:
            os.remove(old_path)
            print(f"[INFO] Removed old ply: {old_path}")

def render_views_from_decoder_output(
    decoder_out: Dict[str, torch.Tensor],
    intrinsics: torch.Tensor,   # [V,3,3]
    extrinsics: torch.Tensor,   # [V,4,4] or [V,3,4]
    image_hw: tuple[int, int],
    view_indices: torch.Tensor,
    chunk_size: int = 2,
):
    
    def ensure_homogeneous_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
        """
        extrinsics:
            [V,4,4] or [V,3,4]
        returns:
            [V,4,4]
        """
        if extrinsics.shape[-2:] == (4, 4):
            return extrinsics

        if extrinsics.shape[-2:] == (3, 4):
            V = extrinsics.shape[0]
            bottom = torch.zeros(V, 1, 4, device=extrinsics.device, dtype=extrinsics.dtype)
            bottom[:, 0, 3] = 1.0
            extrinsics = torch.cat([extrinsics, bottom], dim=1)
            return extrinsics

        raise ValueError(f"Unsupported extrinsics shape: {extrinsics.shape}")

    def normalize_intrinsics(intrinsics: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        intrinsics: [V,3,3] in pixel coordinates
        returns:    [V,3,3] normalized by image width/height
        """
        K = intrinsics.clone()
        K[:, 0, 0] = K[:, 0, 0] / W   # fx
        K[:, 1, 1] = K[:, 1, 1] / H   # fy
        K[:, 0, 2] = K[:, 0, 2] / W   # cx
        K[:, 1, 2] = K[:, 1, 2] / H   # cy
        return K

    flat_scene = flatten_gaussians(decoder_out)
    gs_world = build_renderer_gaussians(flat_scene)
    # print("[DEBUG] built gaussian.means.shape      =", gs_world.means.shape)
    # print("[DEBUG] built gaussian.scales.shape     =", gs_world.scales.shape)
    # print("[DEBUG] built gaussian.rotations.shape  =", gs_world.rotations.shape)
    # print("[DEBUG] built gaussian.opacities.shape  =", gs_world.opacities.shape)
    # print("[DEBUG] built gaussian.harmonics.shape  =", gs_world.harmonics.shape)
    # print("[DEBUG] built gaussian.means.dtype      =", gs_world.means.dtype)
    # print("[DEBUG] built gaussian.means.device     =", gs_world.means.device)

    H, W = image_hw

    extrinsics = ensure_homogeneous_extrinsics(extrinsics)
    intrinsics = normalize_intrinsics(intrinsics, H=H, W=W)
    view_indices = torch.as_tensor(view_indices, device=extrinsics.device, dtype=torch.long).reshape(-1)
    if len(view_indices) == 1:
        extr_sel = extrinsics.index_select(0, view_indices)
        intr_sel = intrinsics.index_select(0, view_indices)
        color, depth = render_3dgs(
            gaussian=gs_world,
            extrinsics=extr_sel,
            intrinsics=intr_sel,
            image_shape=image_hw,
            chunk_size=chunk_size,
            trj_mode="original",
            # use_sh=True,
            use_sh=False,
            color_mode="RGB+ED",
            enable_tqdm=False,
        )
    else:
        extr_sel = extrinsics.index_select(0, view_indices).unsqueeze(0)
        intr_sel = intrinsics.index_select(0, view_indices).unsqueeze(0)   # [v,3,3]
        color, depth = run_renderer_in_chunk_w_trj_mode(
            gaussians=gs_world,
            extrinsics=extr_sel,
            intrinsics=intr_sel,
            image_shape=image_hw,
            chunk_size=chunk_size,
            trj_mode="original",
            use_sh=True,
            color_mode="RGB+ED",
            enable_tqdm=False,
        )

    # print(f"extr_sel: shape={extr_sel.shape}, dtype={extr_sel.dtype}, device={extr_sel.device}")
    # print(f"intr_sel: shape={intr_sel.shape}, dtype={intr_sel.dtype}, device={intr_sel.device}")


    # DA3 export code uses color[idx] as a video tensor, so color is batched at dim 0.
    # print color.shape, depth.shape
    # print(f"Rendered color: shape={color.shape}, dtype={color.dtype}, device={color.device}")
    # print(f"Rendered depth: shape={depth.shape}, dtype={depth.dtype}, device={depth.device}")
    rendered_rgb = color[0]   # [v,3,H,W]
    rendered_depth = depth[0] if depth.ndim >= 4 else depth
    # print(f"Extracted rendered_rgb: shape={rendered_rgb.shape}, dtype={rendered_rgb.dtype}, device={rendered_rgb.device}")
    # print(f"Extracted rendered_depth: shape={rendered_depth.shape}, dtype={rendered_depth.dtype}, device={rendered_depth.device}")

    return rendered_rgb, rendered_depth, flat_scene

def train_one_step_on_scene(
    decoder: VoxelGaussianDecoder,
    optimizer: torch.optim.Optimizer,
    scene_cache,
    device: torch.device,
    views_per_step: int = 2,
    render_chunk_size: int = 2,
):
    decoder.train()
    optimizer.zero_grad()

    dec_in = scene_cache["decoder_inputs"]
    voxel_dict = scene_cache["voxel_dict"]
    gt_images = scene_cache["images"]         # [V,3,H,W]
    intrinsics = scene_cache["intrinsics"]    # [V,3,3]
    extrinsics = scene_cache["extrinsics"]    # [V,4,4] or [V,3,4]
    camera_xyz = scene_cache["camera_xyz"]      # [V,3]
    sky_mask_all = scene_cache["sky_mask"]   # [V,H,W]


    gt_images = (
        torch.from_numpy(gt_images)
        .permute(0,3,1,2)
        .contiguous()
        .float() / 255.0
    )
    if gt_images.max() > 1.0:
        gt_images = gt_images / 255.0

    V, C, H, W = gt_images.shape
    if V == 0:
        raise RuntimeError("No supervision views available in scene_cache.")

    supervision_global = scene_cache.get("supervision_indices", None)
    input_global = scene_cache.get("input_indices", None)

    if supervision_global is not None and input_global is not None:
        input_global_set = set(input_global)

        seen_local = [i for i, g in enumerate(supervision_global) if g in input_global_set]
        novel_local = [i for i, g in enumerate(supervision_global) if g not in input_global_set]

        target_seen = views_per_step // 2
        target_novel = views_per_step - target_seen

        sampled = []

        if len(seen_local) > 0:
            num_seen = min(target_seen, len(seen_local))
            seen_perm = torch.randperm(len(seen_local), device=device)[:num_seen]
            sampled.extend([seen_local[i] for i in seen_perm.cpu().tolist()])

        if len(novel_local) > 0:
            num_novel = min(target_novel, len(novel_local))
            novel_perm = torch.randperm(len(novel_local), device=device)[:num_novel]
            sampled.extend([novel_local[i] for i in novel_perm.cpu().tolist()])

        # 如果某一邊不夠，就從另一邊補足
        if len(sampled) < min(views_per_step, V):
            remaining_pool = [i for i in range(V) if i not in sampled]
            need = min(views_per_step, V) - len(sampled)
            if len(remaining_pool) > 0:
                extra_perm = torch.randperm(len(remaining_pool), device=device)[:need]
                sampled.extend([remaining_pool[i] for i in extra_perm.cpu().tolist()])

        view_indices = torch.tensor(sampled, device=device, dtype=torch.long)
        num_views = len(sampled)
    else:
        num_views = min(views_per_step, V)
        view_indices = torch.randperm(V, device=device)[:num_views]

    num_seen_sampled = 0
    num_novel_sampled = 0
    if supervision_global is not None and input_global is not None:
        input_global_set = set(input_global)
        sampled_global = [supervision_global[i] for i in view_indices.cpu().tolist()]
        num_seen_sampled = sum([g in input_global_set for g in sampled_global])
        num_novel_sampled = len(sampled_global) - num_seen_sampled

    # Now we need to select the camera_xyz to input into the decoder.

    decoder_outs = []
    rendered_rgbs = []

    total_loss = 0.0
    loss_photo_sum = 0.0
    loss_offset_reg_sum = 0.0
    loss_scale_reg_sum = 0.0
    loss_opacity_reg_sum = 0.0
    loss_dssim_sum = 0.0
    loss_color_sum = 0.0
    loss_scale_vol_reg_sum = 0.0
    delta_color_abs_mean_sum = 0.0
    rendered_rgbs_to_log = []
    gt_rgbs_to_log = []
    sky_masks_to_log = []

    flat_scene_stats = None
    
    for i, view in enumerate(view_indices):
        # start = time.time()
        decoder_out = decoder(
            anchor_xyz=dec_in["anchor_xyz"],
            camera_xyz=camera_xyz[view:view+1],  # select one view's camera_xyz at a time, shape [1,3]
            dino_feat=dec_in["dino_feat"],
            confidence=dec_in["confidence"],
            cov_diag=dec_in["cov_diag"],
            voxel_colors=dec_in["voxel_colors"],
        )
        # print(f"Decoder forward pass done. Time: {time.time() - start:.2f} seconds.")

        # start = time.time()
        rendered_rgb, rendered_depth, flat_scene = render_views_from_decoder_output(
            decoder_out=decoder_out,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_hw=(H, W),
            view_indices=[view],
            # view_indices=view_indices,
            chunk_size=render_chunk_size,
        )
        # print(f"Render views from decoder output done. Time: {time.time() - start:.2f} seconds.")
        # decoder_outs.append(decoder_out)
        # rendered_rgbs.append(rendered_rgb)
        gt_rgb = gt_images[view:view+1]
        gt_rgb = gt_rgb.to(rendered_rgb.device)
        sky_mask = sky_mask_all[view:view+1].to(rendered_rgb.device)   # [1,H,W]
        valid_mask = ~sky_mask

        
        if rendered_rgb.ndim == 3:
            rendered_rgb = rendered_rgb.unsqueeze(0) 

        # start = time.time()
        losses = compute_photometric_loss(
            decoder_out=decoder_out,
            voxel_dict=voxel_dict,
            rendered_rgb=rendered_rgb,
            gt_rgb=gt_rgb,
            valid_mask=valid_mask,
        )
        # print(f"Compute photometric loss done. Time: {time.time() - start:.2f} seconds.")

        total_loss = total_loss + losses["total"]
        loss_photo_sum = loss_photo_sum + losses["photo"].item()
        loss_offset_reg_sum = loss_offset_reg_sum + losses["offset_reg"].item()
        loss_scale_reg_sum = loss_scale_reg_sum + losses["scale_reg"].item()
        loss_dssim_sum = loss_dssim_sum + losses["dssim"].item()
        loss_opacity_reg_sum = loss_opacity_reg_sum + losses["opacity_reg"].item()
        loss_color_sum = loss_color_sum + losses["color"].item()
        loss_scale_vol_reg_sum = loss_scale_vol_reg_sum + losses["scale_vol_reg"].item()
        delta_color = decoder_out.get("delta_color", None)
        if delta_color is not None:
            dc = delta_color.detach()
            delta_color_abs_mean_sum += dc.abs().mean().item()

        if i == 0:
            rendered_rgbs_to_log.append(rendered_rgb.detach().cpu())
            gt_rgbs_to_log.append(gt_rgb.detach().cpu())
            sky_masks_to_log = sky_mask.detach().cpu()
            flat_scene_stats = {
                "num_gaussians": int(flat_scene["means3D"].shape[0]),
                "mean_opacity": float(flat_scene["opacity"].mean().item()),
                "mean_scale": float(flat_scene["scales"].mean().item()),
                "mean_abs_center": float(flat_scene["means3D"].abs().mean().item()),
            }
            flat_scene_to_save = {
                "means3D": flat_scene["means3D"].detach().cpu(),
                "scales": flat_scene["scales"].detach().cpu(),
                "rotations": flat_scene["rotations"].detach().cpu(),
                "opacity": flat_scene["opacity"].detach().cpu(),
                "colors": flat_scene["colors"].detach().cpu(),
            }

        del decoder_out, rendered_rgb, rendered_depth, flat_scene, losses

    total_loss = total_loss / num_views
    # start = time.time()
    total_loss.backward()
    optimizer.step()
    # print(f"Backward and optimizer step done. Time: {time.time() - start:.2f} seconds.")

    delta_color_stats = None
    if delta_color_abs_mean_sum > 0:
        delta_color_stats = {
            "abs_mean": delta_color_abs_mean_sum / num_views,
        }


    return {
        "losses": {
            "total": total_loss.detach(),
            "photo": loss_photo_sum / num_views,
            "offset_reg": loss_offset_reg_sum / num_views,
            "scale_reg": loss_scale_reg_sum / num_views,
            "dssim": loss_dssim_sum / num_views,
            "opacity_reg": loss_opacity_reg_sum / num_views,
            "color": loss_color_sum / num_views,
            "scale_vol_reg": loss_scale_vol_reg_sum / num_views,
        },
        "rendered_rgb": rendered_rgbs_to_log[0],
        "gt_rgb": gt_rgbs_to_log[0],
        "view_indices": view_indices.detach().cpu(),
        "scene_stats": flat_scene_stats,
        "flat_scene_to_save": flat_scene_to_save,
        "sky_mask": sky_masks_to_log[0],
        "delta_color_stats": delta_color_stats,
        "sampling_stats": {
            "num_seen": num_seen_sampled,
            "num_novel": num_novel_sampled,
        },
    }

    # return {
    #     "losses": losses,
    #     "flat_scene": flat_scene,
    #     "rendered_rgb": rendered_rgbs.detach().cpu(),
    #     "gt_rgb": gt_rgb.detach().cpu(),
    #     "view_indices": view_indices.detach().cpu(),
    #     "scene_stats": flat_scene_stats,
    # }

def train_one_group(
    decoder,
    optimizer,
    group_scenes,
    model,
    voxelizer,
    cache_root,
    views_per_step,
    steps_per_group,
    device,
    sequence_length,
    supervision_mode,
):
    scene_names = [s["scene_name"] for s in group_scenes]
    scene_map = {s["scene_name"]: s for s in group_scenes}
    step_logs = []

    for step in range(steps_per_group):
        scene_name = scene_names[step % len(scene_names)]
        scene = scene_map[scene_name]
        all_image_paths = scene["image_paths"]
        num_scene_views = len(all_image_paths)

        if num_scene_views < 3:
            print(f"[WARN] Skip scene {scene_name}: only {num_scene_views} view(s)")
            continue

        if supervision_mode == "even_input_mixed_supervision":
            effective_sequence_length = min(sequence_length, len(scene["image_paths"]))

            split = sample_even_odd_window(
                scene["image_paths"],
                sequence_length=effective_sequence_length,
            )
            try:
                scene_cache = prepare_scene_cache(
                    model=model,
                    voxelizer=voxelizer,
                    input_image_paths=split["input_paths"],
                    supervision_image_paths=[scene["image_paths"][i] for i in split["window_indices"]],
                    output_dir=os.path.join(cache_root, scene_name, f"{split['start']:04d}_{split['end']:04d}"),
                    scene_cache_dir=os.path.join(cache_root, scene_name),
                    full_scene_num_views=len(scene["image_paths"]),
                    input_indices=split["input_indices"],
                    supervision_indices=split["window_indices"],
                    device=device,
                )
            except Exception as e:
                print(f"[WARN] Skip scene {scene_name} at step {step}: {e}")
                continue
        else:
            scene_cache = prepare_scene_cache(
                model=model,
                voxelizer=voxelizer,
                input_image_paths=scene["image_paths"],
                output_dir=os.path.join(cache_root, scene_name),
                scene_cache_dir=os.path.join(cache_root, scene_name),
                full_scene_num_views=len(scene["image_paths"]),
                input_indices=list(range(len(scene["image_paths"]))),
                supervision_indices=list(range(len(scene["image_paths"]))),
                device=device,
            )

        info = train_one_step_on_scene(
            decoder=decoder,
            optimizer=optimizer,
            scene_cache=scene_cache,
            views_per_step=views_per_step,
            device=device,
        )

        step_logs.append({
            "scene_name": scene_name,
            "total": info["losses"]["total"].item(),
            "losses": info["losses"],
            "rendered_rgb": info["rendered_rgb"],
            "gt_rgb": info["gt_rgb"],
            "view_indices": info["view_indices"],
            "scene_stats": info.get("scene_stats", {}),
            "flat_scene_to_save": info.get("flat_scene_to_save", None),
            "voxel_mean_points": scene_cache["voxel_dict"]["voxel_mean_points"].detach().cpu() if scene_cache.get("voxel_dict") else None,
            "sky_mask": info.get("sky_mask", None),
            "delta_color_stats": info.get("delta_color_stats", None),
            "sampling_stats": info.get("sampling_stats", None),
        })
        # break  # for debugging, remove this in actual training

    return step_logs

def chunk_list(items, chunk_size):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def build_sky_mask_from_gt(
    images_u8: np.ndarray,   # [V,H,W,3], uint8
    depth: np.ndarray,       # [V,H,W]
    conf: np.ndarray,        # [V,H,W]
):
    V, H, W, _ = images_u8.shape

    # 上方區域 prior
    ys = np.arange(H)[None, :, None]
    top_mask = ys < int(0.45 * H)
    top_mask = np.broadcast_to(top_mask, (V, H, W))

    # 顏色條件：偏亮、低飽和
    img = images_u8.astype(np.float32) / 255.0
    rgb_max = img.max(axis=-1)
    rgb_min = img.min(axis=-1)
    sat = rgb_max - rgb_min
    bright_low_sat = (rgb_max > 0.6) & (sat < 0.18)

    # 幾何條件：遠 depth 或低 conf
    valid_depth = np.isfinite(depth) & (depth > 0)
    if valid_depth.any():
        depth_thr = np.percentile(depth[valid_depth], 90)
    else:
        depth_thr = np.inf
    far_mask = depth >= depth_thr

    valid_conf = np.isfinite(conf)
    if valid_conf.any():
        conf_thr = np.percentile(conf[valid_conf], 25)
    else:
        conf_thr = -np.inf
    low_conf = conf <= conf_thr

    sky_mask = top_mask & bright_low_sat & (far_mask | low_conf)
    return sky_mask.astype(np.bool_)

def mask_to_uint8_image(mask: torch.Tensor) -> np.ndarray:
    """
    mask: [H,W] bool or float
    return: [H,W,3] uint8
    """
    if mask.dtype == torch.bool:
        x = mask.float()
    else:
        x = mask
    x = x.detach().clamp(0, 1).cpu().numpy()
    x = (x * 255).astype(np.uint8)
    x = np.stack([x, x, x], axis=-1)
    return x

def make_wandb_image_triplet_with_mask(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    caption: str = "",
):
    diff = (pred - gt).abs().mean(dim=0, keepdim=True).repeat(3, 1, 1).clamp(0, 1)

    # print shape, dtype, device, min, max of pred, gt, diff, mask for debugging
    # print(f"pred: shape={pred.shape}, dtype={pred.dtype}, device={pred.device}, min={pred.min().item():.4f}, max={pred.max().item():.4f}")
    # print(f"gt: shape={gt.shape}, dtype={gt.dtype}, device={gt.device}, min={gt.min().item():.4f}, max={gt.max().item():.4f}")
    # print(f"diff: shape={diff.shape}, dtype={diff.dtype}, device={diff.device}, min={diff.min().item():.4f}, max={diff.max().item():.4f}")
    # print(f"mask: shape={mask.shape}, dtype={mask.dtype}, device={mask.device}, min={mask.min().item():.4f}, max={mask.max().item():.4f}")
    pred_np = tensor_to_uint8_image(pred)
    gt_np = tensor_to_uint8_image(gt)
    diff_np = tensor_to_uint8_image(diff)
    mask_np = mask_to_uint8_image(mask)

    return [
        wandb.Image(gt_np, caption=f"{caption} | gt"),
        wandb.Image(pred_np, caption=f"{caption} | pred"),
        wandb.Image(diff_np, caption=f"{caption} | diff"),
        wandb.Image(mask_np, caption=f"{caption} | sky_mask"),
    ]

def load_precomputed_sky_mask(cache_dir: str, images_u8: np.ndarray) -> np.ndarray:
    sky_mask_path = os.path.join(cache_dir, "sky_mask.npz")

    if not os.path.exists(sky_mask_path):
        raise FileNotFoundError(
            f"Precomputed sky mask not found: {sky_mask_path}\n"
            f"Please run precompute_sky_mask.py first."
        )

    data = np.load(sky_mask_path)
    sky_mask = data["sky_mask"].astype(np.bool_)

    expected_shape = images_u8.shape[:3]   # [V,H,W]
    if tuple(sky_mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"Sky mask shape mismatch: got {sky_mask.shape}, expected {expected_shape}"
        )

    return sky_mask

def load_precomputed_sky_mask_subset(
    scene_cache_dir: str,
    full_num_views: int,
    selected_indices: list[int],
):
    sky_mask_path = os.path.join(scene_cache_dir, "sky_mask.npz")

    if not os.path.exists(sky_mask_path):
        raise FileNotFoundError(
            f"Precomputed sky mask not found: {sky_mask_path}\n"
            f"Please run precompute_sky_mask.py first."
        )

    data = np.load(sky_mask_path)
    sky_mask = data["sky_mask"].astype(np.bool_)   # [V,H,W]

    cached_num_views = sky_mask.shape[0]

    if cached_num_views != full_num_views:
        print(
            f"[WARN] sky mask num_views mismatch in {scene_cache_dir}: "
            f"cache={cached_num_views}, current_scene={full_num_views}"
        )

    if len(selected_indices) == 0:
        raise ValueError("selected_indices is empty.")

    max_idx = max(selected_indices)
    min_idx = min(selected_indices)

    if min_idx < 0 or max_idx >= cached_num_views:
        raise ValueError(
            f"Sky mask index out of range: min={min_idx}, max={max_idx}, "
            f"cached_num_views={cached_num_views}"
        )

    return sky_mask[selected_indices]

def sample_even_odd_window(image_paths, sequence_length, stride=1):
    num_imgs = len(image_paths)
    if num_imgs < sequence_length:
        start = 0
        end = num_imgs
    else:
        max_start = num_imgs - sequence_length
        start = np.random.randint(0, max_start + 1)
        end = start + sequence_length

    window_indices = list(range(start, end, stride))
    input_indices = window_indices[::2]   # 偶數位置
    target_indices = window_indices[1::2] # 奇數位置

    if len(input_indices) == 0:
        input_indices = [window_indices[0]]
    if len(target_indices) == 0:
        target_indices = [window_indices[-1]]

    input_paths = [image_paths[i] for i in input_indices]
    target_global_indices = target_indices  # 對原 scene 的 index

    return {
        "start": start,
        "end": end,
        "window_indices": window_indices,
        "input_indices": input_indices,
        "target_indices": target_global_indices,
        "input_paths": input_paths,
    }

def load_scene_name_list(txt_path: str) -> list[str]:
    if txt_path is None:
        return []

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"scene list file not found: {txt_path}")

    scene_names = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if len(name) == 0:
                continue
            scene_names.append(name)

    return scene_names


def filter_scenes_by_name(
    scenes: list[dict],
    selected_scene_names: list[str],
    mode: str = "restrict",
) -> list[dict]:
    selected_set = set(selected_scene_names)

    if mode == "restrict":
        filtered = [s for s in scenes if s["scene_name"] in selected_set]
    elif mode == "exclude":
        filtered = [s for s in scenes if s["scene_name"] not in selected_set]
    else:
        raise ValueError(f"Unsupported scene filter mode: {mode}")

    return filtered

def report_missing_scene_names(discovered_scenes: list[dict], requested_scene_names: list[str]):
    discovered = {s["scene_name"] for s in discovered_scenes}
    missing = sorted(set(requested_scene_names) - discovered)
    if len(missing) > 0:
        print(f"[WARN] {len(missing)} scene(s) from list file were not found in dataset:")
        for name in missing:
            print(f"  - {name}")

def prepare_scene_cache(
    model: DepthAnything3,
    voxelizer: SparseVoxelizer,
    input_image_paths,
    output_dir: str,
    device: torch.device,
    supervision_image_paths=None,
    scene_cache_dir=None,
    full_scene_num_views=None,
    input_indices=None,
    supervision_indices=None,
):
    ensure_dir(output_dir)

    # print(f"preparing scene: image_paths={image_paths}")

    with torch.no_grad():
        prediction = model.inference(
            input_image_paths,
            export_dir=output_dir,
            export_format="none",
        )

    # start = time.time()
    voxel_dict = voxelizer.voxelize_prediction(prediction)
    # print(f"Voxelization done. Time: {time.time() - start:.2f} seconds. Num voxels: {voxel_dict['num_voxels']}")

    if voxel_dict["num_voxels"] == 0:
        raise RuntimeError("Voxelization returned zero voxels.")

    decoder_inputs = build_decoder_inputs(voxel_dict, device=device)
    images = prediction.processed_images

    intrinsics = torch.from_numpy(prediction.intrinsics).float().to(device)  # [V,3,3]
    extrinsics = torch.from_numpy(prediction.extrinsics).float().to(device)  # likely [V,4,4] or [V,3,4]
    # extrract from extrinsics if possible, otherwise raise error
    # print(f"extrinsics shape: {extrinsics.shape}")
    if extrinsics.shape[-2:] == (4, 4):
        camera_xyz = extrinsics[:, :3, 3]  # [V,3]
    elif extrinsics.shape[-2:] == (3, 4):
        camera_xyz = extrinsics[:, :3, 3]  # [V,3]
    else:
        raise ValueError(f"Unsupported extrinsics shape: {extrinsics.shape}")
    
    if supervision_image_paths is None:
        supervision_prediction = prediction
        supervision_images = supervision_prediction.processed_images
        supervision_intrinsics = torch.from_numpy(supervision_prediction.intrinsics).float().to(device)
        supervision_extrinsics = torch.from_numpy(supervision_prediction.extrinsics).float().to(device)
        supervision_camera_xyz = supervision_extrinsics[:, :3, 3]

        if scene_cache_dir is None:
            raise ValueError("scene_cache_dir is required to load precomputed sky mask.")

        if full_scene_num_views is None:
            full_scene_num_views = len(input_image_paths)

        if supervision_indices is None:
            supervision_indices = list(range(len(supervision_images)))

        try:
            supervision_sky_mask = load_precomputed_sky_mask_subset(
                scene_cache_dir=scene_cache_dir,
                full_num_views=full_scene_num_views,
                selected_indices=supervision_indices,
            )
        except Exception as e:
            print(f"[WARN] Failed to load sky mask subset for {scene_cache_dir}: {e}")
            H, W = supervision_images.shape[1:3]
            supervision_sky_mask = np.zeros(
                (len(supervision_images), H, W),
                dtype=np.bool_,
            )
    else:
        with torch.no_grad():
            supervision_prediction = model.inference(
                supervision_image_paths,
                export_format="none",
            )

        supervision_images = supervision_prediction.processed_images
        supervision_intrinsics = torch.from_numpy(supervision_prediction.intrinsics).float().to(device)
        supervision_extrinsics = torch.from_numpy(supervision_prediction.extrinsics).float().to(device)
        supervision_camera_xyz = supervision_extrinsics[:, :3, 3]

        if scene_cache_dir is None:
            raise ValueError("scene_cache_dir is required to load precomputed sky mask.")
        if full_scene_num_views is None:
            raise ValueError("full_scene_num_views is required for supervision subset mode.")
        if supervision_indices is None:
            raise ValueError("supervision_indices is required for supervision subset mode.")

        try:
            supervision_sky_mask = load_precomputed_sky_mask_subset(
                scene_cache_dir=scene_cache_dir,
                full_num_views=full_scene_num_views,
                selected_indices=supervision_indices,
            )
        except Exception as e:
            print(f"[WARN] Failed to load sky mask subset for {scene_cache_dir}: {e}")
            H, W = supervision_images.shape[1:3]
            supervision_sky_mask = np.zeros(
                (len(supervision_images), H, W),
                dtype=np.bool_,
            )

    # return {
    #     "prediction": prediction,
    #     "voxel_dict": voxel_dict,
    #     "decoder_inputs": decoder_inputs,
    #     "images": images,
    #     "intrinsics": intrinsics,
    #     "extrinsics": extrinsics,
    #     "image_paths": image_paths,
    #     "camera_xyz": camera_xyz,
    #     "sky_mask": torch.from_numpy(sky_mask),   # [V,H,W], bool
    # }

    return {
        "prediction": prediction,
        "voxel_dict": voxel_dict,
        "decoder_inputs": decoder_inputs,

        # encoder/input branch
        "input_image_paths": input_image_paths,
        "input_indices": input_indices,
        "supervision_indices": supervision_indices,
        "input_intrinsics": intrinsics,
        "input_extrinsics": extrinsics,
        "input_camera_xyz": camera_xyz,

        # supervision branch
        "images": supervision_images,
        "intrinsics": supervision_intrinsics,
        "extrinsics": supervision_extrinsics,
        "camera_xyz": supervision_camera_xyz,
        "sky_mask": torch.from_numpy(supervision_sky_mask),
    }

@torch.no_grad()
def verify_scene_no_cache(
    decoder,
    model,
    voxelizer,
    image_paths,
    scene_name,
    output_dir,
    cache_root,
    epoch,
    device,
    view_indices=(0,10,20),
):
    decoder.eval()

    print(f"[VERIFY] preparing scene {scene_name}")

    prediction = model.inference(image_paths)

    # sky_mask_np = build_sky_mask_from_gt(
    #     prediction.processed_images,
    #     prediction.depth,
    #     prediction.conf,
    # )

    sky_mask_np = load_precomputed_sky_mask(
        cache_dir=os.path.join(cache_root, scene_name),
        images_u8=prediction.processed_images,
    )

    gt_images = (
        torch.from_numpy(prediction.processed_images)
        .permute(0,3,1,2)
        .contiguous()
        .float() / 255.0
    )

    intrinsics = torch.from_numpy(prediction.intrinsics).float().to(device)
    extrinsics = torch.from_numpy(prediction.extrinsics).float().to(device)

    V, _, H, W = gt_images.shape
    valid_view_indices = select_valid_view_indices(V, list(view_indices))
    view_tensor = torch.tensor(valid_view_indices, device=device, dtype=torch.long)

    voxel_dict = voxelizer.voxelize_prediction(prediction)
    decoder_inputs = build_decoder_inputs(voxel_dict, device=device)
    camera_xyz = extrinsics[:, :3, 3].to(device)

    rendered_rgbs = []
    flat_scene = None

    for view in valid_view_indices:
        decoder_out = decoder(
            anchor_xyz=decoder_inputs["anchor_xyz"],
            camera_xyz=camera_xyz[view:view+1],
            dino_feat=decoder_inputs["dino_feat"],
            confidence=decoder_inputs["confidence"],
            cov_diag=decoder_inputs["cov_diag"],
            voxel_colors=decoder_inputs["voxel_colors"],
        )

        pred_rgb, pred_depth, flat_scene = render_views_from_decoder_output(
            decoder_out=decoder_out,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_hw=(H, W),
            view_indices=[view],
        )

        rendered_rgbs.append(pred_rgb)   # [3,H,W]

    rendered_rgbs = torch.stack(rendered_rgbs, dim=0)   # [v,3,H,W]

    gt_images = gt_images.to(device)
    gt_rgb = gt_images[view_tensor]                     # [v,3,H,W]

    sky_mask = torch.from_numpy(sky_mask_np).to(device) # [V,H,W]
    sky_mask = sky_mask[view_tensor]                    # [v,H,W]
    valid_mask = ~sky_mask

    l1 = masked_l1_loss(rendered_rgbs, gt_rgb, valid_mask).item()

    scene_dir = os.path.join(output_dir, "verification", f"epoch_{epoch:04d}", scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    # print gt_rgb, rendered_rgbs
    # print(f"[VERIFY] gt_rgb: shape={gt_rgb.shape}, dtype={gt_rgb.dtype}, device={gt_rgb.device}, "
    #       f"min={gt_rgb.min().item():.4f}, max={gt_rgb.max().item():.4f}")
    # print(f"[VERIFY] rendered_rgbs: shape={rendered_rgbs.shape}, dtype={rendered_rgbs.dtype}, device={rendered_rgbs.device}, "
    #       f"min={rendered_rgbs.min().item():.4f}, max={rendered_rgbs.max().item():.4f}")
    # print(f"[VERIFY] sky_mask: shape={sky_mask.shape}, dtype={sky_mask.dtype}, device={sky_mask.device}, "
    #       f"num_sky_pixels={sky_mask.sum().item()}, num_valid_pixels={valid_mask.sum().item()}")

    for i, view_idx in enumerate(valid_view_indices):
        save_tensor_image(gt_rgb[i], f"{scene_dir}/view_{view_idx}_gt.png")
        save_tensor_image(rendered_rgbs[i], f"{scene_dir}/view_{view_idx}_pred.png")
        save_diff_image(rendered_rgbs[i], gt_rgb[i], f"{scene_dir}/view_{view_idx}_diff.png")

    save_flat_scene_as_ply(
        flat_scene,
        os.path.join(scene_dir, "gaussian_scene.ply"),
    )

    print(f"[VERIFY] {scene_name} L1 = {l1:.6f}")

    return {
        "scene_name": scene_name,
        "l1": l1,
        "pred_rgb": rendered_rgbs.detach().cpu(),
        "gt_rgb": gt_rgb.detach().cpu(),
        "sky_mask": sky_mask.detach().cpu(),
        "view_indices": valid_view_indices,
        "num_voxels": int(voxel_dict["num_voxels"]),
        "num_gaussians": int(flat_scene["means3D"].shape[0]),
    }

def safe_verify_scene_no_cache(*args, **kwargs):
    try:
        out = verify_scene_no_cache(*args, **kwargs)
        return {
            "ok": True,
            "result": out,
            "error": None,
        }
    except Exception as e:
        print(f"[WARN] verify_scene_no_cache failed: {e}")
        traceback.print_exc()

        # 嘗試同步，讓錯誤更完整地暴露
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        return {
            "ok": False,
            "result": None,
            "error": repr(e),
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str,
        default="None", help="Folder containing input jpg images")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="None",
        help="Dataset root containing scene folders, e.g. datasets/waymo/",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="FRONT",
        help="Camera subfolder inside each scene folder",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--steps-per-group", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--views-per-step", type=int, default=2)
    parser.add_argument("--render-chunk-size", type=int, default=2)
    parser.add_argument("--lambda-photo", type=float, default=1.0)
    parser.add_argument("--lambda-color", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--val-scene-names", nargs="*", default=[])
    parser.add_argument("--val-max-scenes", type=int, default=4)
    parser.add_argument("--val-view-indices", nargs="*", type=int, default=[0, 10, 20])
    parser.add_argument("--verify-render-chunk-size", type=int, default=2)

    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument(
        "--supervision-mode",
        type=str,
        default="even_input_mixed_supervision",
        choices=[
            "random_views",
            "even_input_mixed_supervision",
        ],
    )

    parser.add_argument(
        "--scene-list-file",
        type=str,
        default=None,
        help="Path to a txt file containing one scene name per line",
    )
    parser.add_argument(
        "--scene-filter-mode",
        type=str,
        default="restrict",
        choices=["restrict", "exclude"],
        help="How to apply --scene-list-file to discovered scenes",
    )
    parser.add_argument(
        "--train-only-scene-list",
        action="store_true",
        help="If set, apply scene-list filtering only to training scenes, not validation scenes",
    )

    # voxelizer params
    parser.add_argument("--voxel-size", type=float, default=0.4)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--truncation-band", type=float, default=0.5)
    parser.add_argument("--feat-mode", type=str, default="last2_avg")
    parser.add_argument("--feat-dim-out", type=int, default=128)
    parser.add_argument("--neighbor-patch-radius", type=int, default=0,
                        help="Neighboring patch radius for voxel feature aggregation (0=center only, 1=3x3, 2=5x5)")
    parser.add_argument("--perview-conf", action="store_true",
                        help="Compute confidence threshold per view instead of globally")

    # sky mask
    parser.add_argument(
        "--cache-root",
        type=str,
        default="output_train_voxel_decoder/cache",
        help="Fixed cache root for precomputed scene cache such as sky masks",
    )

    # decoder params
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-gaussians", type=int, default=4)

    # save ckpt
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--keep-last-k", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)

    # wandb params
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="voxel-anchored-3dgs")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-log-train-images-every", type=int, default=1)
    parser.add_argument("--wandb-log-val-images", action="store_true")
    parser.add_argument("--wandb-watch-model", action="store_true")

    args = parser.parse_args()

    args.output_dir = make_timestamped_output_dir(args.output_dir)
    print(f"[INFO] Output directory: {args.output_dir}")

    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    # wandb setup
    run = None
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
            dir=args.output_dir,
        )

        if args.wandb_watch_model:
            wandb.watch(decoder if "decoder" in locals() else None, log="all", log_freq=100)

    scenes = discover_waymo_scenes(
    dataset_root=args.dataset_root,
    camera_name=args.camera_name,
    )
    print(f"[INFO] Found {len(scenes)} scenes before filtering")

    scene_list_names = []
    if args.scene_list_file is not None:
        scene_list_names = load_scene_name_list(args.scene_list_file)
        print(f"[INFO] Loaded {len(scene_list_names)} scene names from {args.scene_list_file}")

    # split first
    if len(args.val_scene_names) > 0:
        val_scene_names = set(args.val_scene_names)
        val_scenes = [s for s in scenes if s["scene_name"] in val_scene_names]
        train_scenes = [s for s in scenes if s["scene_name"] not in val_scene_names]
    else:
        val_scenes = scenes[-args.val_max_scenes:]
        train_scenes = scenes[:-args.val_max_scenes] if len(scenes) > args.val_max_scenes else scenes

    # then filter by scene list
    if len(scene_list_names) > 0:
        if args.train_only_scene_list:
            train_scenes = filter_scenes_by_name(
                train_scenes,
                scene_list_names,
                mode=args.scene_filter_mode,
            )
        else:
            train_scenes = filter_scenes_by_name(
                train_scenes,
                scene_list_names,
                mode=args.scene_filter_mode,
            )
            val_scenes = filter_scenes_by_name(
                val_scenes,
                scene_list_names,
                mode=args.scene_filter_mode,
            )

    print(f"[INFO] Final train scenes: {len(train_scenes)}")
    print(f"[INFO] Final val scenes: {len(val_scenes)}")

    if len(train_scenes) == 0:
        raise RuntimeError("No training scenes left after filtering.")

    if len(val_scenes) == 0:
        print("[WARN] No validation scenes left after filtering.")

    report_missing_scene_names(scenes, scene_list_names)
    # scenes = scenes[:1]
    # val_scenes = scenes[:1]

    model = DepthAnything3.from_pretrained(args.model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    voxelizer = SparseVoxelizer(
        max_depth=args.max_depth,
        voxel_size=args.voxel_size,
        conf_percentile=args.conf_percentile,
        truncation_band=args.truncation_band,
        feat_mode=args.feat_mode,
        neighbor_patch_radius=args.neighbor_patch_radius,
        perview_conf=args.perview_conf,
        # feat_dim_out=args.feat_dim_out,
    )

    # 用第一個 scene warmup，拿 dino_dim
    first_scene = scenes[0]
    with torch.no_grad():
        pred = model.inference(
            first_scene["image_paths"],
            export_dir=args.output_dir,
            export_format="none",
        )
        voxel_dict = voxelizer.voxelize_prediction(pred)

    if voxel_dict["voxel_features"] is None:
        raise RuntimeError("voxel_features is None.")

    dino_dim = voxel_dict["voxel_features"].shape[1]

    decoder = VoxelGaussianDecoder(
        dino_dim=dino_dim,
        hidden_dim=args.hidden_dim,
        num_gaussians=args.num_gaussians,
        voxel_size=args.voxel_size,
    ).to(device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    start_epoch = 1

    if args.resume is not None:
        print(f"[INFO] Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])

        start_epoch = ckpt["epoch"] + 1
        print(f"[INFO] Resume from epoch {start_epoch}")

    # scene_groups = chunk_list(scenes, args.group_size)
    scene_groups = chunk_list(train_scenes, args.group_size)

    if args.use_wandb and args.wandb_watch_model:
        wandb.watch(decoder, log="all", log_freq=100)

    best_val = float("inf")

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        last_train_log = None
        epoch_logs = []
        print(f"\n[INFO] Epoch {epoch:04d}/{args.epochs}")
        np.random.shuffle(scene_groups)

        for group_idx, group in enumerate(scene_groups):
            print(f"[INFO] Loading group {group_idx+1}/{len(scene_groups)} with {len(group)} scenes")

            logs = train_one_group(
                decoder=decoder,
                optimizer=optimizer,
                group_scenes=group,
                model=model,
                voxelizer=voxelizer,
                cache_root=args.cache_root,
                views_per_step=args.views_per_step,
                steps_per_group=args.steps_per_group,
                device=device,
                sequence_length=args.sequence_length,
                supervision_mode=args.supervision_mode,
            )

            if len(logs) == 0:
                print("[WARN] No valid logs in this group.")
                continue
            if len(logs) > 0:
                last_train_log = logs[-1]
                epoch_logs.extend(logs)

            avg_total = np.mean([x["total"] for x in logs])
            print(f"[INFO] Group {group_idx+1} avg_total={avg_total:.6f}")
            if args.use_wandb:
                for local_step, log_item in enumerate(logs):
                    global_step += 1

                    wandb_log = {
                        "train/epoch": epoch,
                        "train/group_idx": group_idx + 1,
                        "train/global_step": global_step,
                        "train/loss_total": log_item["total"],
                    }

                    delta_color_stats = log_item.get("delta_color_stats", None)
                    if delta_color_stats is not None:
                        wandb_log["train/delta_color_abs_mean"] = delta_color_stats["abs_mean"]

                    # save all loss in log_item that are not None
                    for k, v in log_item["losses"].items():
                        # print(f"[DEBUG] loss {k} = {v}")
                        if v is not None:
                            wandb_log[f"train/loss_{k}"] = v
                    # if log_item["photo"] is not None:
                    #     wandb_log["train/loss_photo"] = log_item["photo"]
                    # if log_item["offset_reg"] is not None:
                    #     wandb_log["train/loss_offset_reg"] = log_item["offset_reg"]

                    wandb_log.update({
                        f"train/{k}": v for k, v in log_item.get("scene_stats", {}).items()
                    })

                    wandb_log.update({
                        f"train/memory/{k}": v for k, v in get_cpu_mem_stats().items()
                    })
                    wandb_log.update({
                        f"train/memory/{k}": v for k, v in get_gpu_mem_stats(device).items()
                    })
                    sampling_stats = log_item.get("sampling_stats", None)
                    if sampling_stats is not None:
                        wandb_log["train/num_seen_supervision"] = sampling_stats["num_seen"]
                        wandb_log["train/num_novel_supervision"] = sampling_stats["num_novel"]

                    if global_step % args.wandb_log_train_images_every == 0:
                        # print log_item["rendered_rgb"].shape, log_item["gt_rgb"].shape
                        # print(f"[DEBUG] log_item['rendered_rgb'] shape={log_item['rendered_rgb'].shape}, dtype={log_item['rendered_rgb'].dtype}, device={log_item['rendered_rgb'].device}")
                        # print(f"[DEBUG] log_item['gt_rgb'] shape={log_item['gt_rgb'].shape}, dtype={log_item['gt_rgb'].dtype}, device={log_item['gt_rgb'].device}")
                        # pred0 = log_item["rendered_rgb"][0].detach().cpu()
                        pred0 = log_item["rendered_rgb"][0].detach().cpu()
                        gt0 = log_item["gt_rgb"][0].detach().cpu()
                        # print(f"[DEBUG] log_item['sky_mask'] shape={log_item['sky_mask'].shape}, dtype={log_item['sky_mask'].dtype}, device={log_item['sky_mask'].device}")
                        mask0 = log_item["sky_mask"].detach().cpu()
                        imgs = make_wandb_image_triplet_with_mask(
                            pred=pred0,
                            gt=gt0,
                            mask=mask0,
                            caption=f"train step={global_step} scene={log_item['scene_name']}",
                        )
                        wandb_log["train/render_samples"] = imgs

                        # save debug PLY pair for this render_samples step
                        if log_item.get("flat_scene_to_save") is not None:
                            save_debug_ply_pair(
                                flat_scene=log_item["flat_scene_to_save"],
                                voxel_mean_points=log_item.get("voxel_mean_points"),
                                output_dir=args.output_dir,
                                global_step=global_step,
                                scene_name=log_item["scene_name"],
                                keep_last_k=10,
                            )

                    wandb.log(wandb_log, step=global_step)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if epoch % args.save_every == 0:
            save_checkpoint(
                decoder=decoder,
                optimizer=optimizer,
                epoch=epoch,
                output_dir=args.output_dir,
                keep_last_k=args.keep_last_k,
            )
        
        if epoch % args.val_every == 0:
            val_l1_list = []
            print(f"\n[INFO] Validation for epoch {epoch:04d} on {len(val_scenes)} scenes")
            for scene in val_scenes:
                verify_out = safe_verify_scene_no_cache(
                    decoder=decoder,
                    model=model,
                    voxelizer=voxelizer,
                    image_paths=scene["image_paths"],
                    scene_name=scene["scene_name"],
                    output_dir=args.output_dir,
                    cache_root=args.cache_root,
                    epoch=epoch,
                    device=device,
                )
                if verify_out["ok"]:
                    val_info = verify_out["result"]
                    # 正常 log / save
                    val_l1_list.append(val_info["l1"])
                else:
                    print(f"[WARN] skip this verification because of error: {verify_out['error']}")
                    continue

                if args.use_wandb:
                    val_log = {
                        "val/epoch": epoch,
                        "val/l1": val_info["l1"],
                        "val/num_voxels": val_info["num_voxels"],
                        "val/num_gaussians": val_info["num_gaussians"],
                        **{f"val/memory/{k}": v for k, v in get_cpu_mem_stats().items()},
                        **{f"val/memory/{k}": v for k, v in get_gpu_mem_stats(device).items()},
                    }

                    if args.wandb_log_val_images:
                        # print pred0, gt0, mask0
                        # print(f"[DEBUG] val_info['pred_rgb'] shape={val_info['pred_rgb'].shape}, dtype={val_info['pred_rgb'].dtype}, device={val_info['pred_rgb'].device}")
                        # print(f"[DEBUG] val_info['gt_rgb'] shape={val_info['gt_rgb'].shape}, dtype={val_info['gt_rgb'].dtype}, device={val_info['gt_rgb'].device}")
                        # print(f"[DEBUG] val_info['sky_mask'] shape={val_info['sky_mask'].shape}, dtype={val_info['sky_mask'].dtype}, device={val_info['sky_mask'].device}")

                        pred0 = val_info["pred_rgb"][0]
                        gt0 = val_info["gt_rgb"][0]
                        mask0 = val_info["sky_mask"][0]
                        val_log[f"val/{val_info['scene_name']}_samples"] = make_wandb_image_triplet_with_mask(
                            pred=pred0,
                            gt=gt0,
                            mask=mask0,
                            caption=f"val epoch={epoch} scene={val_info['scene_name']}",
                        )

                    wandb.log(val_log, step=global_step)
            if len(val_l1_list) > 0:
                mean_val = np.mean(val_l1_list)

                if mean_val < best_val:
                    best_val = mean_val

                    torch.save({
                        "epoch": epoch,
                        "decoder": decoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val": best_val,
                    }, os.path.join(args.output_dir, "best.pth"))

                    print(f"[INFO] Saved BEST checkpoint (val={best_val:.6f})")
            
            if args.use_wandb and len(val_l1_list) > 0:
                wandb.log({
                    "val/epoch_mean_l1": float(np.mean(val_l1_list)),
                    "val/epoch": epoch,
                }, step=global_step)

if __name__ == "__main__":
    main()