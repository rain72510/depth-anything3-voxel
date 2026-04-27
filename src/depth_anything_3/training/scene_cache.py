# Auto-extracted from scripts/train_voxel_decoder.py
# Module: scene_cache

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
from depth_anything_3.training.sky_mask import load_precomputed_sky_mask_subset
from depth_anything_3.training.scene_discovery import sample_even_odd_window
from depth_anything_3.training.data_prep import build_decoder_inputs
from depth_anything_3.training.utils import ensure_dir

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
