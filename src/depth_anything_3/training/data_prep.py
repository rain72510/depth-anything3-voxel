# Auto-extracted from scripts/train_voxel_decoder.py
# Module: data_prep

from typing import Dict, Any
import torch
from types import SimpleNamespace

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
