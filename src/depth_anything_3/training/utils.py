# Auto-extracted from scripts/train_voxel_decoder.py
# Module: utils

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

def make_timestamped_output_dir(base_output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(final_dir, exist_ok=True)
    return final_dir

def chunk_list(items, chunk_size):
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

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
