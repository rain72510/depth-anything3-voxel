"""v3 decoder — voxel-anchored geometry + per-Gaussian pixel-projected appearance.

Each Gaussian within a voxel projects its OWN center into every input view, picks
the most-confident visible view, and bilinear-samples the DINO feature + raw RGB
from that pixel. This gives every Gaussian a unique appearance signal even when
K > 1 share the same anchor.

Geometry path is unchanged from v2 (compact voxel feature + view dir/dist → MLP →
centers/scales/rotations). Output dict matches v1/v2 keys so train_step / loss
path stay unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


class GeometryHeadV3(nn.Module):
    """Same geometry head as v2 — voxel-aggregated feature + view conditioning
    drives center / scale / rotation. K Gaussians per anchor share this geometry
    path; per-Gaussian differentiation happens in AppearanceHeadV3."""

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

    def forward(self, voxel_features, anchor_xyz, camera_xyz):
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
        anchor_scale = F.softplus(anchor_scale_raw) + 1e-4
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


class AppearanceHeadV3(nn.Module):
    """Per-Gaussian pixel-projected appearance.

    For every Gaussian g with center μ_g coming out of the geometry head:
      1. project μ_g into each input view
      2. mask invisible views (depth ≤ 0 OR uv outside image)
      3. pick the view with highest confidence at the projected pixel
      4. bilinear-sample DINO feature and raw RGB at that pixel
      5. decode opacity + color residual from the per-Gaussian DINO feature
      6. final color = sampled_rgb + tanh(residual) × scale, clamped to [0, 1]

    Memory footprint:
      - per-view image-space DINO at fp16: V × Hf × Wf × C ≈ 70 MB at V=12, C=3072.
      - intermediate sampled features: NK × V × C float32 — chunked if needed.
    """

    def __init__(
        self,
        dino_dim: int = 3072,
        hidden_dim: int = 128,
        num_gaussians: int = 4,
        color_residual_scale: float = 0.25,
        invisible_opacity: float = 0.0,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.color_residual_scale = color_residual_scale
        self.invisible_opacity = invisible_opacity

        self.app_mlp = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.opacity_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Linear(hidden_dim, 3)

    @staticmethod
    def _project_centers(centers_flat, intrinsics, extrinsics):
        """centers_flat [NK,3] in world coords → returns:
           uv_pix [V, NK, 2] pixel coords, depth [V, NK]."""
        if extrinsics.shape[-2:] == (4, 4):
            E34 = extrinsics[:, :3, :]
        else:
            E34 = extrinsics
        ones = torch.ones(centers_flat.shape[0], 1, device=centers_flat.device, dtype=centers_flat.dtype)
        homo = torch.cat([centers_flat, ones], dim=-1)                    # [NK, 4]
        cam_pts = torch.einsum('vij,nj->vni', E34, homo)                  # [V, NK, 3]
        depth = cam_pts[..., 2]                                           # [V, NK]
        uv_h = torch.einsum('vij,vnj->vni', intrinsics, cam_pts)          # [V, NK, 3]
        uv_pix = uv_h[..., :2] / uv_h[..., 2:].clamp_min(1e-6)            # [V, NK, 2]
        return uv_pix, depth

    @staticmethod
    def _grid_sample_at_uv(input_chw, uv_pix, image_hw):
        """Bilinear-sample input [V, C, H_in, W_in] at pixel coords uv_pix [V, NK, 2]
        in the image-space dimensions image_hw=(H, W). Returns [V, C, NK]."""
        H, W = image_hw
        u_norm = uv_pix[..., 0] / (W - 1) * 2.0 - 1.0
        v_norm = uv_pix[..., 1] / (H - 1) * 2.0 - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(1)         # [V, 1, NK, 2]
        out = F.grid_sample(input_chw, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return out.squeeze(2)                                             # [V, C, NK]

    def forward(
        self,
        gaussian_centers: torch.Tensor,   # [N, K, 3]
        image_dino_feats: torch.Tensor,   # [V, Hf, Wf, C_full] (fp16 OK)
        raw_images: torch.Tensor,         # [V, 3, H, W] in [0, 1]
        intrinsics: torch.Tensor,         # [V, 3, 3]
        extrinsics: torch.Tensor,         # [V, 3, 4] or [V, 4, 4]
        confidence: torch.Tensor,         # [V, H, W]
        voxel_colors: Optional[torch.Tensor] = None,  # [N, 3] (unused in v3 — kept for sig parity)
    ) -> Dict[str, torch.Tensor]:
        N, K, _ = gaussian_centers.shape
        flat_centers = gaussian_centers.reshape(-1, 3)
        NK = flat_centers.shape[0]
        H, W = raw_images.shape[-2:]

        # 1. Project each Gaussian into every view
        uv_pix, depth = self._project_centers(flat_centers, intrinsics, extrinsics)  # [V, NK, 2], [V, NK]

        # 2. Visibility
        u_pix = uv_pix[..., 0]
        v_pix = uv_pix[..., 1]
        vis = (depth > 0.1) & (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H)  # [V, NK]
        any_visible = vis.any(dim=0)                                      # [NK]

        # 3. Sample confidence at each view, pick best visible view
        conf_at_uv = self._grid_sample_at_uv(confidence.unsqueeze(1).float(),
                                             uv_pix, (H, W)).squeeze(1)   # [V, NK]
        conf_masked = torch.where(vis, conf_at_uv, torch.full_like(conf_at_uv, -float('inf')))
        best_view = conf_masked.argmax(dim=0)                             # [NK]

        # 4a. Sample DINO at best view's projected pixel
        Hf, Wf = image_dino_feats.shape[1:3]
        # image_dino_feats is in patch grid coords; project to those resolution.
        # Easiest: bilinear in patch-grid uv space.
        dino_chw = image_dino_feats.permute(0, 3, 1, 2).float().contiguous()  # [V, C, Hf, Wf]
        # uv in image pixel coords; convert to patch-grid coords by scaling.
        scale_u = (Wf - 1) / max(W - 1, 1)
        scale_v = (Hf - 1) / max(H - 1, 1)
        uv_patch = torch.stack([uv_pix[..., 0] * scale_u, uv_pix[..., 1] * scale_v], dim=-1)
        dino_at_uv = self._grid_sample_at_uv(dino_chw, uv_patch, (Hf, Wf))  # [V, C, NK]
        # Gather best view per Gaussian
        # dino_at_uv: [V, C, NK] → permute [NK, V, C] then gather along dim 1
        dino_per_g = dino_at_uv.permute(2, 0, 1).contiguous()              # [NK, V, C]
        idx = best_view[:, None, None].expand(-1, 1, dino_per_g.shape[-1])
        sampled_dino = dino_per_g.gather(1, idx).squeeze(1)                # [NK, C]

        # 4b. Sample raw RGB at best view's projected pixel
        rgb_at_uv = self._grid_sample_at_uv(raw_images.float().contiguous(), uv_pix, (H, W))  # [V, 3, NK]
        rgb_per_g = rgb_at_uv.permute(2, 0, 1).contiguous()                # [NK, V, 3]
        idx_rgb = best_view[:, None, None].expand(-1, 1, 3)
        sampled_rgb = rgb_per_g.gather(1, idx_rgb).squeeze(1).clamp(0.0, 1.0)  # [NK, 3]

        # 5. Decode
        h = self.app_mlp(sampled_dino)
        opacity_raw = self.opacity_head(h)
        color_raw = self.color_head(h)

        opacity = torch.sigmoid(opacity_raw - 2.0)                         # [NK, 1]
        opacity = torch.where(any_visible.unsqueeze(-1), opacity,
                              torch.full_like(opacity, self.invisible_opacity))

        delta_color = torch.tanh(color_raw) * self.color_residual_scale
        colors = (sampled_rgb + delta_color).clamp(0.0, 1.0)

        return {
            "opacity": opacity.reshape(N, K, 1),
            "colors": colors.reshape(N, K, 3),
            "base_color": sampled_rgb.reshape(N, K, 3),
            "delta_color": delta_color.reshape(N, K, 3),
            # Diagnostics
            "any_visible": any_visible.reshape(N, K),
            "best_view": best_view.reshape(N, K),
        }


class VoxelGaussianDecoderV3(nn.Module):
    """Wrapper. Forward signature accepts everything v1/v2 do PLUS the per-view
    image-space DINO features, raw images, intrinsics, extrinsics, confidence
    needed for per-Gaussian projection."""

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
        color_residual_scale: float = 0.25,
    ):
        super().__init__()
        self.geometry = GeometryHeadV3(
            feat_dim_in=dino_dim,
            hidden_dim=hidden_dim,
            encoder_dim=encoder_dim,
            num_gaussians=num_gaussians,
            voxel_size=voxel_size,
            scale_clamp_mult=scale_clamp_mult,
            scale_clamp_distance_ref=scale_clamp_distance_ref,
            scale_init_mult=scale_init_mult,
        )
        self.appearance = AppearanceHeadV3(
            dino_dim=pixel_dim,
            hidden_dim=hidden_dim,
            num_gaussians=num_gaussians,
            color_residual_scale=color_residual_scale,
        )
        print(
            f"Initialized VoxelGaussianDecoderV3: dino_dim={dino_dim}, pixel_dim={pixel_dim}, "
            f"hidden_dim={hidden_dim}, encoder_dim={encoder_dim}, K={num_gaussians}, "
            f"voxel_size={voxel_size}, color_residual_scale={color_residual_scale}"
        )

    def forward(
        self,
        anchor_xyz: torch.Tensor,
        dino_feat: torch.Tensor,
        confidence: torch.Tensor,                 # per-anchor scalar confidence (v1 compat, unused)
        cov_diag: torch.Tensor,                   # v1 compat, unused
        image_dino_feats: torch.Tensor,           # [V, Hf, Wf, C_full]
        raw_images: torch.Tensor,                 # [V, 3, H, W]
        intrinsics_v: torch.Tensor,               # [V, 3, 3]
        extrinsics_v: torch.Tensor,               # [V, 3, 4] or [V, 4, 4]
        conf_map: torch.Tensor,                   # [V, H, W]
        camera_xyz: Optional[torch.Tensor] = None,
        voxel_colors: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if camera_xyz is None:
            raise ValueError("camera_xyz is required for v3 GeometryHead")

        geo = self.geometry(dino_feat, anchor_xyz, camera_xyz)
        app = self.appearance(
            gaussian_centers=geo["centers"],
            image_dino_feats=image_dino_feats,
            raw_images=raw_images,
            intrinsics=intrinsics_v,
            extrinsics=extrinsics_v,
            confidence=conf_map,
            voxel_colors=voxel_colors,
        )
        out = {**geo, **app}
        out["anchor_latent"] = None
        return out
