# Auto-extracted from scripts/train_voxel_decoder.py
# Module: wandb_utils

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
from depth_anything_3.training.utils import mask_to_uint8_image, tensor_to_uint8_image

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
