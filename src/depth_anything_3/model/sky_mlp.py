"""Directional sky color MLP for driving-scene 3DGS.

A small MLP that takes (positionally-encoded ray direction) + (global scene
feature) and predicts per-pixel sky color. Used to render the sky region while
3D Gaussians handle the rest of the scene.

Pattern inspired by Street Gaussians (ECCV 2024) and SUDS (CVPR 2023), but
conditioned on a feed-forward global scene feature so it generalizes
cross-scene.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def positional_encoding(x: torch.Tensor, num_freq: int = 4) -> torch.Tensor:
    """Sinusoidal positional encoding for unit vectors.

    Args:
        x: [..., 3] unit direction vectors.
        num_freq: number of frequency bands.

    Returns:
        [..., 3 * (1 + 2*num_freq)] encoded.
    """
    if num_freq <= 0:
        return x
    freqs = 2.0 ** torch.arange(num_freq, device=x.device, dtype=x.dtype)
    xb = x.unsqueeze(-1) * freqs
    sin = torch.sin(xb).flatten(-2)
    cos = torch.cos(xb).flatten(-2)
    return torch.cat([x, sin, cos], dim=-1)


class SkyMLP(nn.Module):
    """Ray-direction + scene-feature conditioned sky color MLP."""

    def __init__(
        self,
        global_feat_dim: int,
        hidden_dim: int = 64,
        num_freq: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_freq = num_freq
        dir_enc_dim = 3 * (1 + 2 * num_freq) if num_freq > 0 else 3
        in_dim = dir_enc_dim + global_feat_dim

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(max(0, num_layers - 2)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, 3), nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, ray_dirs: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """Predict sky RGB per ray.

        Args:
            ray_dirs: [..., 3] world-space unit direction vectors.
            global_feat: [D] (one scene) or [B, D] (batched scenes) global descriptor.

        Returns:
            [..., 3] RGB in [0, 1].
        """
        enc = positional_encoding(ray_dirs, self.num_freq)
        leading = enc.shape[:-1]

        if global_feat.dim() == 1:
            g = global_feat.view(*([1] * len(leading)), -1).expand(*leading, -1)
        else:
            # Assume first dim is batch, matches first dim of leading
            extra = len(leading) - 1
            g = global_feat.view(global_feat.shape[0], *([1] * extra), -1).expand(*leading, -1)

        mlp_in = torch.cat([enc, g], dim=-1)
        return self.mlp(mlp_in)


def compute_ray_dirs_world(
    H: int,
    W: int,
    intrinsics: torch.Tensor,   # [3, 3] pixel-coord
    extrinsics_c2w: torch.Tensor,  # [4, 4] camera-to-world
    device: torch.device | None = None,
) -> torch.Tensor:
    """Per-pixel world-space ray directions.

    Args:
        H, W: image dimensions in pixels.
        intrinsics: [3, 3] in pixel coordinates (fx,fy,cx,cy on diagonal/offsets).
        extrinsics_c2w: [4, 4] camera-to-world. If you have world-to-camera, invert first.
        device: output device (default: intrinsics.device).

    Returns:
        [H, W, 3] unit vectors in world space.
    """
    if device is None:
        device = intrinsics.device
    dtype = intrinsics.dtype
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")

    dirs_cam = torch.stack(
        [(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], dim=-1
    )  # [H, W, 3], camera-space (non-normalized)
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)

    R = extrinsics_c2w[:3, :3]
    # d_world[i] = R @ d_cam[i]; in batched einsum: (R, dirs) -> dirs_world
    dirs_world = torch.einsum("ij,hwj->hwi", R, dirs_cam)
    return dirs_world


def composite_sky(
    rendered_rgb: torch.Tensor,  # [3, H, W] or [H, W, 3]
    sky_rgb: torch.Tensor,       # matching shape
    sky_mask: torch.Tensor,      # [H, W] bool — True where sky
) -> torch.Tensor:
    """Blend Gaussian render (non-sky) with MLP sky prediction (sky)."""
    if rendered_rgb.dim() == 3 and rendered_rgb.shape[0] == 3:
        # channels-first [3, H, W]
        m = sky_mask.to(rendered_rgb.dtype).unsqueeze(0)  # [1, H, W]
        return rendered_rgb * (1.0 - m) + sky_rgb * m
    else:
        # channels-last [H, W, 3]
        m = sky_mask.to(rendered_rgb.dtype).unsqueeze(-1)  # [H, W, 1]
        return rendered_rgb * (1.0 - m) + sky_rgb * m
