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

from depth_anything_3.api import DepthAnything3

# currently no wrapper
# 你自己的 voxelizer 與 decoder
from depth_anything_3.sparse_voxelizer import SparseVoxelizer
from depth_anything_3.model.voxel_gaussian_decoder import VoxelGaussianDecoder

from types import SimpleNamespace
from PIL import Image

from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode, render_3dgs
from depth_anything_3.specs import Gaussians

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

    return {
        "anchor_xyz": anchor_xyz,   # [K, 3]
        "dino_feat": dino_feat,     # [K, C]
        "confidence": confidence,   # [K]
        "cov_diag": cov_diag,       # [K, 3]
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


def compute_photometric_loss(
    decoder_out: Dict[str, torch.Tensor],
    voxel_dict: Dict[str, Any],
    rendered_rgb: torch.Tensor,   # [v,3,H,W]
    gt_rgb: torch.Tensor,         # [v,3,H,W]
    lambda_photo: float = 1.0,
    lambda_color: float = 0.1,
    lambda_offset: float = 1e-1,
    lambda_scale: float = 1e-3,
    lambda_opacity: float = 1e-4,
    lambda_anchor: float = 1e-4,
    lambda_disp: float = 1e-3,
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

    losses["photo"] = F.l1_loss(rendered_rgb, gt_rgb)

    # for k, v in decoder_out.items():
    #     print(f"decoder_out[{k}]: shape={v.shape}, dtype={v.dtype}, device={v.device}, "
    #           f"min={v.min().item():.4f}, max={v.max().item():.4f}")


    # # 2. optional voxel color prior
    # if voxel_dict.get("voxel_colors", None) is not None:
    #     target_color = voxel_dict["voxel_colors"].to(decoder_out["colors"].device).float()
    #     pred_color = decoder_out["colors"].mean(dim=1)
    #     losses["color"] = F.l1_loss(pred_color, target_color)
    # else:
    #     losses["color"] = torch.tensor(0.0, device=decoder_out["colors"].device)

    # # 3. geometry / regularization
    losses["offset_reg"] = decoder_out["offsets"].pow(2).mean()
    # print(f"offset_reg: {losses['offset_reg'].item():.6f}")
    # losses["scale_reg"] = torch.log(decoder_out["scales"] + 1e-8).pow(2).mean()
    # losses["opacity_reg"] = decoder_out["opacity"].mean()
    # losses["anchor_scale_reg"] = decoder_out["anchor_scale"].pow(2).mean()

    # # actual displacement regularization
    # disp = decoder_out["offsets"] * decoder_out["anchor_scale"].unsqueeze(-1)
    # losses["disp_reg"] = disp.pow(2).mean()
    
    # print all loss components for debugging
    # for k, v in losses.items():
    #     print(f"{k} loss: {v.item():.6f}")

    losses["total"] = (
        lambda_photo * losses["photo"]
        # + lambda_color * losses["color"]
        + lambda_offset * losses["offset_reg"]
        # + lambda_scale * losses["scale_reg"]
        # + lambda_opacity * losses["opacity_reg"]
        # + lambda_anchor * losses["anchor_scale_reg"]
        # + lambda_disp * losses["disp_reg"]
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

    M = means.shape[0]
    device = means.device
    dtype = means.dtype
    # harmonics: [1, M, 3, 9]
    # use RGB as SH DC term only
    harmonics = torch.zeros((1, M, 3, 9), device=device, dtype=dtype)
    harmonics[0, :, :, 0] = colors

    gaussian = SimpleNamespace(
        means=means.unsqueeze(0),          # [1, M, 3]
        scales=scales.unsqueeze(0),        # [1, M, 3]
        rotations=rotations.unsqueeze(0),  # [1, M, 4]
        opacities=opacities.unsqueeze(0),  # [1, M]
        harmonics=harmonics,               # [1, M, 3, 9]
    )
    return gaussian

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
            use_sh=True,
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


    gt_images = (
        torch.from_numpy(gt_images)
        .permute(0,3,1,2)
        .contiguous()
        .float() / 255.0
    )
    if gt_images.max() > 1.0:
        gt_images = gt_images / 255.0
    V, C, H, W = gt_images.shape
    num_views = min(views_per_step, V)
    view_indices = torch.randperm(V, device=device)[:num_views]

    # Now we need to select the camera_xyz to input into the decoder.

    decoder_outs = []
    rendered_rgbs = []

    total_loss = 0.0
    photo_loss_sum = 0.0
    offset_reg_sum = 0.0
    rendered_rgbs_to_log = []
    gt_rgbs_to_log = []

    flat_scene_stats = None
    
    for i, view in enumerate(view_indices):
        decoder_out = decoder(
            anchor_xyz=dec_in["anchor_xyz"],
            camera_xyz=camera_xyz[view:view+1],  # select one view's camera_xyz at a time, shape [1,3]
            dino_feat=dec_in["dino_feat"],
            confidence=dec_in["confidence"],
            cov_diag=dec_in["cov_diag"],
        )
        rendered_rgb, rendered_depth, flat_scene = render_views_from_decoder_output(
            decoder_out=decoder_out,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_hw=(H, W),
            view_indices=[view],
            # view_indices=view_indices,
            chunk_size=render_chunk_size,
        )
        # decoder_outs.append(decoder_out)
        # rendered_rgbs.append(rendered_rgb)
        gt_rgb = gt_images[view:view+1]
        gt_rgb = gt_rgb.to(rendered_rgb.device)

        losses = compute_photometric_loss(
            decoder_out=decoder_out,
            voxel_dict=voxel_dict,
            rendered_rgb=rendered_rgb,
            gt_rgb=gt_rgb,
        )

        total_loss = total_loss + losses["total"]
        photo_loss_sum += losses["photo"].detach()
        offset_reg_sum += losses["offset_reg"].detach()

        if i == 0:
            rendered_rgbs_to_log.append(rendered_rgb.detach().cpu())
            gt_rgbs_to_log.append(gt_rgb.detach().cpu())
            flat_scene_stats = {
                "num_gaussians": int(flat_scene["means3D"].shape[0]),
                "mean_opacity": float(flat_scene["opacity"].mean().item()),
                "mean_scale": float(flat_scene["scales"].mean().item()),
                "mean_abs_center": float(flat_scene["means3D"].abs().mean().item()),
            }

        del decoder_out, rendered_rgb, rendered_depth, flat_scene, losses

    total_loss = total_loss / num_views
    total_loss.backward()
    optimizer.step()

    # decoder_outs = torch.cat(decoder_outs, dim=0)  # [v, ...]
    # for each value in decoder_out, we stack them along dim 0 corresponding to views, so we can compute loss against gt_images[view_indices]

    # for k in decoder_outs[0].keys():
    #     decoder_outs[0][k] = torch.cat([d[k] for d in decoder_outs], dim=0)  # now decoder_outs[0][k] has shape [v, ...]
    # del decoder_outs[1:]  # free memory
    # decoder_outs = decoder_outs[0]  # we only need one dict since they are now concatenated
    # rendered_rgbs = torch.stack(rendered_rgbs, dim=0)  # [v,3,H,W]
    
    # gt_images = gt_images.to(device)
    # gt_rgb = gt_images[view_indices]

    # losses = compute_photometric_loss(
    #     decoder_out=decoder_outs,
    #     voxel_dict=voxel_dict,
    #     rendered_rgb=rendered_rgbs,
    #     gt_rgb=gt_rgb,
    # )

    # losses["total"].backward()
    # optimizer.step()

    # flat_scene_stats = {
    #     "num_gaussians": int(flat_scene["means3D"].shape[0]),
    #     "mean_opacity": float(flat_scene["opacity"].mean().item()),
    #     "mean_scale": float(flat_scene["scales"].mean().item()),
    #     "mean_abs_center": float(flat_scene["means3D"].abs().mean().item()),
    # }

    return {
        "losses": {
            "total": total_loss.detach(),
            "photo": photo_loss_sum / num_views,
            "offset_reg": offset_reg_sum / num_views,
        },
        "rendered_rgb": rendered_rgbs_to_log[0],
        "gt_rgb": gt_rgbs_to_log[0],
        "view_indices": view_indices.detach().cpu(),
        "scene_stats": flat_scene_stats,
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
    decoder: VoxelGaussianDecoder,
    optimizer: torch.optim.Optimizer,
    group_scene_caches: dict,
    views_per_step: int,
    steps_per_group: int,
    device: torch.device,
):
    scene_names = list(group_scene_caches.keys())
    step_logs = []

    for step in range(steps_per_group):
        scene_name = scene_names[step % len(scene_names)]
        scene_cache = group_scene_caches[scene_name]

        info = train_one_step_on_scene(
            decoder=decoder,
            optimizer=optimizer,
            scene_cache=scene_cache,
            views_per_step=views_per_step,
            device=device,
        )

        losses = info["losses"]

        step_logs.append({
            "scene_name": scene_name,
            "total": losses["total"].item(),
            "photo": float(losses["photo"].item()) if "photo" in losses else None,
            "offset_reg": float(losses["offset_reg"].item()) if "offset_reg" in losses else None,
            "rendered_rgb": info["rendered_rgb"],
            "gt_rgb": info["gt_rgb"],
            "view_indices": info["view_indices"],
            "scene_stats": info.get("scene_stats", {}),
        })
        # break  # for debugging, remove this in actual training

    return step_logs

@torch.no_grad()
def verify_scene(
    decoder,
    scene_name: str,
    scene_cache: Dict[str, Any],
    output_dir: str,
    epoch: int,
    view_indices: list[int],
    render_chunk_size: int = 2,
):
    decoder.eval()

    dec_in = scene_cache["decoder_inputs"]
    gt_images = scene_cache["images"]       # [V,3,H,W]
    intrinsics = scene_cache["intrinsics"]
    extrinsics = scene_cache["extrinsics"]
    camera_xyz = scene_cache["camera_xyz"]

    V, _, H, W = gt_images.shape
    view_indices = select_valid_view_indices(V, view_indices)
    view_tensor = torch.tensor(view_indices, device=gt_images.device, dtype=torch.long)

    decoder_out = decoder(
        anchor_xyz=dec_in["anchor_xyz"],
        dino_feat=dec_in["dino_feat"],
        confidence=dec_in["confidence"],
        cov_diag=dec_in["cov_diag"],
    )

    pred_rgb, pred_depth, flat_scene = render_views_from_decoder_output(
        decoder_out=decoder_out,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_hw=(H, W),
        view_indices=view_tensor,
        chunk_size=render_chunk_size,
    )

    gt_rgb = gt_images[view_tensor]

    l1 = F.l1_loss(pred_rgb, gt_rgb).item()

    scene_dir = os.path.join(output_dir, "verification", f"epoch_{epoch:04d}", scene_name)
    ensure_dir(scene_dir)

    save_gaussian_scene_npz(
        flat_scene,
        os.path.join(scene_dir, f"gaussian_scene_epoch_{epoch:04d}.npz"),
    )

    for local_i, view_idx in enumerate(view_indices):
        save_tensor_image(gt_rgb[local_i], os.path.join(scene_dir, f"view_{view_idx:04d}_gt.png"))
        save_tensor_image(pred_rgb[local_i], os.path.join(scene_dir, f"view_{view_idx:04d}_pred.png"))
        save_diff_image(
            pred_rgb[local_i],
            gt_rgb[local_i],
            os.path.join(scene_dir, f"view_{view_idx:04d}_diff.png"),
        )

    metrics = {
        "scene_name": scene_name,
        "epoch": epoch,
        "num_views": len(view_indices),
        "l1": l1,
    }
    save_json(metrics, os.path.join(scene_dir, "metrics.json"))
    return metrics

@torch.no_grad()
def verify_scenes(
    decoder,
    scene_caches: Dict[str, Dict[str, Any]],
    output_dir: str,
    epoch: int,
    view_indices: list[int],
    render_chunk_size: int = 2,
):
    all_metrics = []

    for scene_name, scene_cache in scene_caches.items():
        try:
            metrics = verify_scene(
                decoder=decoder,
                scene_name=scene_name,
                scene_cache=scene_cache,
                output_dir=output_dir,
                epoch=epoch,
                view_indices=view_indices,
                render_chunk_size=render_chunk_size,
            )
            all_metrics.append(metrics)
            print(f"[VERIFY] scene={scene_name} l1={metrics['l1']:.6f}")
        except Exception as e:
            print(f"[VERIFY][WARN] scene={scene_name} failed: {e}")

    summary = {
        "epoch": epoch,
        "num_scenes": len(all_metrics),
        "mean_l1": float(np.mean([m["l1"] for m in all_metrics])) if len(all_metrics) > 0 else None,
        "scenes": all_metrics,
    }

    summary_dir = os.path.join(output_dir, "verification", f"epoch_{epoch:04d}")
    ensure_dir(summary_dir)
    save_json(summary, os.path.join(summary_dir, "summary.json"))

    if summary["mean_l1"] is not None:
        print(f"[VERIFY] epoch={epoch:04d} mean_l1={summary['mean_l1']:.6f}")

    return summary

def chunk_list(items, chunk_size):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

def prepare_scene_cache(
    model: DepthAnything3,
    voxelizer: SparseVoxelizer,
    image_paths,
    output_dir: str,
    device: torch.device,
):
    ensure_dir(output_dir)

    with torch.no_grad():
        prediction = model.inference(
            image_paths,
            export_dir=output_dir,
            export_format="none",
        )

    voxel_dict = voxelizer.voxelize_prediction(prediction)

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

    return {
        "prediction": prediction,
        "voxel_dict": voxel_dict,
        "decoder_inputs": decoder_inputs,
        "images": images,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "image_paths": image_paths,
        "camera_xyz": camera_xyz,
    }

@torch.no_grad()
def verify_scene_no_cache(
    decoder,
    model,
    voxelizer,
    image_paths,
    scene_name,
    output_dir,
    epoch,
    device,
    view_indices=(0,10,20),
):
    decoder.eval()

    print(f"[VERIFY] preparing scene {scene_name}")

    # 1 DAv3 inference
    prediction = model.inference(
        image_paths,
    )

    # gt_images = prediction.processed_images.permute(0,3,1,2)   # V,3,H,W
    
    gt_images = (
        torch.from_numpy(prediction.processed_images)
        .permute(0,3,1,2)
        .contiguous()
        .float() / 255.0
    )
    intrinsics = torch.from_numpy(prediction.intrinsics).float().to(device)  # [V,3,3]
    extrinsics = torch.from_numpy(prediction.extrinsics).float().to(device)  # likely [V,4,

    V, _, H, W = gt_images.shape

    # 2 voxelize
    voxel_dict = voxelizer.voxelize_prediction(prediction)

    # 3 build decoder input
    decoder_inputs = build_decoder_inputs(voxel_dict, device=device)
    camera_xyz = extrinsics[:, :3, 3].to(device)  # [V,3]

    # decoder_outs = []
    rendered_rgbs = []

    for view in view_indices:
        decoder_out = decoder(
            anchor_xyz=decoder_inputs["anchor_xyz"],
            camera_xyz=camera_xyz[view:view+1],   # [1, 3]
            dino_feat=decoder_inputs["dino_feat"],
            confidence=decoder_inputs["confidence"],
            cov_diag=decoder_inputs["cov_diag"],
        )

        pred_rgb, pred_depth, flat_scene = render_views_from_decoder_output(
            decoder_out=decoder_out,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_hw=(H, W),
            view_indices=[view],   # 只 render 這個 view
        )

        # decoder_outs.append(decoder_out)
        rendered_rgbs.append(pred_rgb)

    # for k in decoder_outs[0].keys():
    #     decoder_outs[0][k] = torch.cat([d[k] for d in decoder_outs], dim=0)  # now decoder_outs[0][k] has shape [v, ...]
    # del decoder_outs[1:]  # free memory
    # decoder_outs = decoder_outs[0]  # we only need one dict since they are now concatenated
    rendered_rgbs = torch.stack(rendered_rgbs, dim=0)  # [v,3,H,W]
    
    view_tensor = torch.tensor(view_indices, device=device)

    gt_images = gt_images.to(device)
    gt_rgb = gt_images[view_tensor]

    # 6 metric
    l1 = F.l1_loss(rendered_rgbs, gt_rgb).item()

    # 7 save
    scene_dir = os.path.join(output_dir, "verification", f"epoch_{epoch:04d}", scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    for i, view_idx in enumerate(view_indices):
        save_tensor_image(gt_rgb[i], f"{scene_dir}/view_{view_idx}_gt.png")
        save_tensor_image(rendered_rgbs[i], f"{scene_dir}/view_{view_idx}_pred.png")
        save_diff_image(rendered_rgbs[i], gt_rgb[i], f"{scene_dir}/view_{view_idx}_diff.png")

    save_gaussian_scene_npz(
        flat_scene,
        os.path.join(scene_dir, "gaussian_scene.npz")
    )

    # valid_view_indices = select_valid_view_indices(V, list(view_indices))
    # view_tensor = torch.tensor(valid_view_indices, device=device)

    print(f"[VERIFY] {scene_name} L1 = {l1:.6f}")

    return {
        "scene_name": scene_name,
        "l1": l1,
        "pred_rgb": pred_rgb.detach().cpu(),
        "gt_rgb": gt_rgb.detach().cpu(),
        "view_indices": list(view_indices),
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

    # voxelizer params
    parser.add_argument("--voxel-size", type=float, default=0.4)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--truncation-band", type=float, default=0.5)
    parser.add_argument("--feat-mode", type=str, default="last2_avg")
    parser.add_argument("--feat-dim-out", type=int, default=128)

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
    print(f"[INFO] Found {len(scenes)} scenes")

    all_scene_names = [s["scene_name"] for s in scenes]

    if len(args.val_scene_names) > 0:
        val_scene_names = set(args.val_scene_names)
        val_scenes = [s for s in scenes if s["scene_name"] in val_scene_names]
        train_scenes = [s for s in scenes if s["scene_name"] not in val_scene_names]
    else:
        # 預設最後幾個 scene 當 val
        val_scenes = scenes[-args.val_max_scenes:]
        train_scenes = scenes[:-args.val_max_scenes] if len(scenes) > args.val_max_scenes else scenes
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
        feat_dim_out=args.feat_dim_out,
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
        print(f"\n[INFO] Epoch {epoch:04d}/{args.epochs}")
        np.random.shuffle(scene_groups)

        for group_idx, group in enumerate(scene_groups):
            print(f"[INFO] Loading group {group_idx+1}/{len(scene_groups)} with {len(group)} scenes")

            group_cache = {}
            for scene in group:
                scene_name = scene["scene_name"]
                scene_out_dir = os.path.join(args.output_dir, "cache", scene_name)
                try:
                    group_cache[scene_name] = prepare_scene_cache(
                        model=model,
                        voxelizer=voxelizer,
                        image_paths=scene["image_paths"],
                        output_dir=scene_out_dir,
                        device=device,
                    )
                except Exception as e:
                    print(f"[WARN] Skip scene {scene_name}: {e}")

            if len(group_cache) == 0:
                print("[WARN] No valid scenes in this group.")
                continue

            logs = train_one_group(
                decoder=decoder,
                optimizer=optimizer,
                group_scene_caches=group_cache,
                steps_per_group=args.steps_per_group,
                views_per_step=args.views_per_step,
                device=device,
            )

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

                    if log_item["photo"] is not None:
                        wandb_log["train/loss_photo"] = log_item["photo"]
                    if log_item["offset_reg"] is not None:
                        wandb_log["train/loss_offset_reg"] = log_item["offset_reg"]

                    wandb_log.update({
                        f"train/{k}": v for k, v in log_item.get("scene_stats", {}).items()
                    })

                    wandb_log.update({
                        f"train/memory/{k}": v for k, v in get_cpu_mem_stats().items()
                    })
                    wandb_log.update({
                        f"train/memory/{k}": v for k, v in get_gpu_mem_stats(device).items()
                    })

                    if global_step % args.wandb_log_train_images_every == 0:
                        # print log_item["rendered_rgb"].shape, log_item["gt_rgb"].shape
                        # print(f"[DEBUG] log_item['rendered_rgb'] shape={log_item['rendered_rgb'].shape}, dtype={log_item['rendered_rgb'].dtype}, device={log_item['rendered_rgb'].device}")
                        # print(f"[DEBUG] log_item['gt_rgb'] shape={log_item['gt_rgb'].shape}, dtype={log_item['gt_rgb'].dtype}, device={log_item['gt_rgb'].device}")
                        # pred0 = log_item["rendered_rgb"][0].detach().cpu()
                        pred0 = log_item["rendered_rgb"].detach().cpu()
                        gt0 = log_item["gt_rgb"][0].detach().cpu()
                        imgs = make_wandb_image_triplet(
                            pred=pred0,
                            gt=gt0,
                            caption=f"train step={global_step} scene={log_item['scene_name']}",
                        )
                        wandb_log["train/render_samples"] = imgs

                    wandb.log(wandb_log, step=global_step)

            del group_cache
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
                    epoch=epoch,
                    device=device,
                )
                if verify_out["ok"]:
                    val_info = verify_out["result"]
                    # 正常 log / save
                else:
                    print(f"[WARN] skip this verification because of error: {verify_out['error']}")

                val_l1_list.append(val_info["l1"])

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
                        # print(f"[DEBUG] log_item['pred_rgb'] shape={val_info['pred_rgb'].shape}, dtype={val_info['pred_rgb'].dtype}, device={val_info['pred_rgb'].device}")
                        # print(f"[DEBUG] log_item['gt_rgb'] shape={val_info['gt_rgb'].shape}, dtype={val_info['gt_rgb'].dtype}, device={val_info['gt_rgb'].device}")
                        # pred0 = val_info["pred_rgb"][0]
                        # gt rgb is now [3, 3, 336, 504], check it!

                        pred0 = val_info["pred_rgb"]
                        gt0 = val_info["gt_rgb"][0]
                        val_log[f"val/{val_info['scene_name']}_samples"] = make_wandb_image_triplet(
                            pred=pred0,
                            gt=gt0,
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