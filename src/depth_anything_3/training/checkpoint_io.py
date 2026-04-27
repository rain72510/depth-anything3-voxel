# Auto-extracted from scripts/train_voxel_decoder.py
# Module: checkpoint_io

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
from depth_anything_3.sparse_voxelizer import SparseVoxelizer
from depth_anything_3.model.voxel_gaussian_decoder import VoxelGaussianDecoder
from depth_anything_3.model.sky_mlp import SkyMLP, compute_ray_dirs_world
from types import SimpleNamespace
from PIL import Image
import lpips
from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode, render_3dgs
from depth_anything_3.specs import Gaussians
from depth_anything_3.utils.gsply_helpers import export_ply
from depth_anything_3.utils.loss_utils import ssim
from PIL import Image
import lpips
import math

# sibling imports (auto-generated)
from depth_anything_3.training.utils import ensure_dir

def save_gaussian_scene_npz(scene: Dict[str, torch.Tensor], path: str):
    np.savez_compressed(
        path,
        means3D=scene["means3D"].detach().cpu().numpy(),
        scales=scene["scales"].detach().cpu().numpy(),
        rotations=scene["rotations"].detach().cpu().numpy(),
        opacity=scene["opacity"].detach().cpu().numpy(),
        colors=scene["colors"].detach().cpu().numpy(),
    )

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
