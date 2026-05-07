import torch
import torch.nn as nn
from typing import Optional, Dict


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


class GeometryHead(nn.Module):
    """v2 geometry path. Consumes the compact (truncated) voxel feature plus
    view direction + distance, outputs centers/scales/rotations only.

    Distance-adaptive scale clamp and scale_init_mult are kept from v1 since
    Run 5/6 showed they matter for far-away anchors."""

    def __init__(
        self,
        feat_dim_in: int,
        hidden_dim: int = 128,
        encoder_dim: int = 32,
        num_gaussians: int = 4,
        voxel_size: float = 0.4,
        scale_clamp_mult: float = 0.5,
        scale_clamp_distance_ref: float = 0.0,
        scale_init_mult: float = 0.1,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.voxel_size = voxel_size
        self.scale_clamp_mult = scale_clamp_mult
        self.scale_clamp_distance_ref = scale_clamp_distance_ref
        self.scale_init_mult = scale_init_mult

        self.encoder = nn.Sequential(
            nn.Linear(feat_dim_in, encoder_dim),
            nn.ReLU(inplace=True),
        )
        self.geo_mlp = nn.Sequential(
            nn.Linear(encoder_dim + 3 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Linear(hidden_dim, num_gaussians * 3)
        self.scaling_head = nn.Linear(hidden_dim, 3)
        self.cov_head = nn.Linear(hidden_dim, num_gaussians * 7)

        self.offset_template = nn.Parameter(0.05 * torch.randn(num_gaussians, 3))

    def forward(
        self,
        voxel_features: torch.Tensor,   # [N, feat_dim_in]
        anchor_xyz: torch.Tensor,        # [N, 3]
        camera_xyz: torch.Tensor,        # [N, 3] or [1, 3]
    ) -> Dict[str, torch.Tensor]:
        N = anchor_xyz.shape[0]
        K = self.num_gaussians

        if camera_xyz.shape[0] == 1 and N > 1:
            camera_xyz = camera_xyz.expand(N, -1)
        vec = anchor_xyz - camera_xyz
        view_dist = vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        view_dir = vec / view_dist

        feat_low = self.encoder(voxel_features)
        geo_in = torch.cat([feat_low, view_dir, view_dist], dim=-1)
        g = self.geo_mlp(geo_in)

        delta_offsets = self.offset_head(g).view(N, K, 3)
        offsets = self.offset_template.unsqueeze(0) + delta_offsets

        anchor_scale_raw = self.scaling_head(g)
        anchor_scale = torch.nn.functional.softplus(anchor_scale_raw) + 1e-4
        offset_scale = anchor_scale[:, :3]

        cov_raw = self.cov_head(g).view(N, K, 7)
        scale_raw = cov_raw[..., :3]
        quat_raw = cov_raw[..., 3:7]

        raw_scales = torch.exp(scale_raw) * (self.voxel_size * self.scale_init_mult) + 1e-4
        if self.scale_clamp_distance_ref > 0:
            dist_full = vec.norm(dim=-1)
            adaptive_clamp = (
                self.voxel_size
                * self.scale_clamp_mult
                * (1.0 + dist_full / self.scale_clamp_distance_ref)
            ).view(-1, 1, 1)
            scales = torch.minimum(raw_scales, adaptive_clamp.expand_as(raw_scales))
        else:
            scales = raw_scales.clamp(max=self.voxel_size * self.scale_clamp_mult)

        quaternions = normalize_quaternion(quat_raw)
        centers = anchor_xyz[:, None, :] + torch.tanh(offsets * offset_scale[:, None, :]) * self.voxel_size

        return {
            "centers": centers,
            "offsets": offsets,
            "anchor_scale": anchor_scale,
            "scales": scales,
            "quaternions": quaternions,
            "offset_scale": offset_scale,
        }


class AppearanceHead(nn.Module):
    """v2 appearance path. View-independent (per spec §1 note). Consumes the
    full-dim per-anchor pixel-aligned DINO feature, outputs opacity + color.

    If voxel_colors is provided at forward time, color is treated as a residual
    on top of the per-voxel base color (matches v1 semantics so that the first
    v2 run is comparable to Run 6)."""

    def __init__(
        self,
        feat_dim_in: int = 3072,
        hidden_dim: int = 128,
        num_gaussians: int = 4,
        color_residual_scale: float = 0.25,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.color_residual_scale = color_residual_scale

        self.app_mlp = nn.Sequential(
            nn.Linear(feat_dim_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.opacity_head = nn.Linear(hidden_dim, num_gaussians)
        self.color_head = nn.Linear(hidden_dim, num_gaussians * 3)

    def forward(
        self,
        pixel_features: torch.Tensor,         # [N, feat_dim_in], may be fp16
        voxel_colors: Optional[torch.Tensor] = None,  # [N, 3] in [0,1]
    ) -> Dict[str, torch.Tensor]:
        N = pixel_features.shape[0]
        K = self.num_gaussians

        h = self.app_mlp(pixel_features.float())

        opacity = torch.sigmoid(self.opacity_head(h).unsqueeze(-1) - 2.0)  # [N, K, 1]
        color_raw = self.color_head(h).view(N, K, 3)

        if voxel_colors is not None:
            base_color = voxel_colors[:, None, :]
            delta_color = torch.tanh(color_raw) * self.color_residual_scale
            colors = (base_color + delta_color).clamp(0.0, 1.0)
        else:
            base_color = None
            delta_color = None
            colors = torch.sigmoid(color_raw)

        return {
            "opacity": opacity,
            "colors": colors,
            "base_color": base_color,
            "delta_color": delta_color,
        }


class VoxelGaussianDecoderV2(nn.Module):
    """Wrapper exposing the same forward signature as v1 plus an extra
    voxel_pixel_features tensor. Output dict matches v1's keys so the training
    loop stays unchanged downstream."""

    def __init__(
        self,
        dino_dim: int,
        pixel_dim: int = 3072,
        hidden_dim: int = 128,
        encoder_dim: int = 32,
        num_gaussians: int = 4,
        voxel_size: float = 0.4,
        scale_clamp_mult: float = 0.5,
        scale_clamp_distance_ref: float = 0.0,
        scale_init_mult: float = 0.1,
        use_voxel_color: bool = True,
        color_residual_scale: float = 0.25,
    ):
        super().__init__()
        self.use_voxel_color = use_voxel_color
        self.geometry = GeometryHead(
            feat_dim_in=dino_dim,
            hidden_dim=hidden_dim,
            encoder_dim=encoder_dim,
            num_gaussians=num_gaussians,
            voxel_size=voxel_size,
            scale_clamp_mult=scale_clamp_mult,
            scale_clamp_distance_ref=scale_clamp_distance_ref,
            scale_init_mult=scale_init_mult,
        )
        self.appearance = AppearanceHead(
            feat_dim_in=pixel_dim,
            hidden_dim=hidden_dim,
            num_gaussians=num_gaussians,
            color_residual_scale=color_residual_scale,
        )
        print(
            f"Initialized VoxelGaussianDecoderV2: dino_dim={dino_dim}, pixel_dim={pixel_dim}, "
            f"hidden_dim={hidden_dim}, encoder_dim={encoder_dim}, K={num_gaussians}, "
            f"voxel_size={voxel_size}, use_voxel_color={use_voxel_color}"
        )

    def forward(
        self,
        anchor_xyz: torch.Tensor,
        dino_feat: torch.Tensor,
        confidence: torch.Tensor,         # accepted for v1 compat, unused
        cov_diag: torch.Tensor,           # accepted for v1 compat, unused
        voxel_pixel_features: torch.Tensor,
        camera_xyz: Optional[torch.Tensor] = None,
        voxel_colors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if camera_xyz is None:
            raise ValueError("camera_xyz is required for v2 GeometryHead")

        geo = self.geometry(dino_feat, anchor_xyz, camera_xyz)
        app = self.appearance(
            voxel_pixel_features,
            voxel_colors=voxel_colors if self.use_voxel_color else None,
        )
        out = {**geo, **app}
        out["anchor_latent"] = None
        return out
