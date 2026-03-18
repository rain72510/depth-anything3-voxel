import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


def normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    q: [..., 4]
    """
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        activation: nn.Module = nn.ReLU,
        final_activation: Optional[nn.Module] = None,
    ):
        super().__init__()

        layers = []
        dim_list = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            if i < len(dim_list) - 2:
                layers.append(activation())
            elif final_activation is not None:
                layers.append(final_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VoxelGaussianDecoder(nn.Module):
    """
    Minimal voxel-anchor -> 4 Gaussians decoder.

    Inputs per voxel:
        - anchor_xyz:        [N, 3]
        - dino_feat:         [N, F]
        - confidence:        [N, 1] or [N]
        - cov_diag:          [N, 3]

    Optional view conditioning:
        - camera_xyz:        [N, 3] or [1, 3]
          used only for appearance heads (opacity, color)

    Outputs:
        - centers:           [N, K, 3]
        - offsets:           [N, K, 3]
        - anchor_scale:      [N, 3]
        - scales:            [N, K, 3]
        - quaternions:       [N, K, 4]
        - opacity:           [N, K, 1]
        - colors:            [N, K, 3]
    """

    def __init__(
        self,
        dino_dim: int,
        hidden_dim: int = 256,
        num_gaussians: int = 4,
        use_view_conditioning: bool = True,
        color_act: str = "sigmoid",
    ):
        super().__init__()

        self.dino_dim = dino_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        self.use_view_conditioning = use_view_conditioning
        self.color_act = color_act

        # --------------------------------------------------
        # Anchor input:
        #   dino_feat         -> F
        #   confidence        -> 1
        #   cov_diag          -> 3
        # total = F + 4
        # --------------------------------------------------
        self.anchor_in_dim = dino_dim + 1 + 3

        self.anchor_encoder = MLP(
            in_dim=self.anchor_in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=3,
            activation=nn.ReLU,
        )

        # --------------------------------------------------
        # Global learnable offset template, like scaffold prior
        # shape: [K, 3]
        # Initialized near zero
        # --------------------------------------------------
        self.offset_template = nn.Parameter(
            0.05 * torch.randn(num_gaussians, 3)
        )

        # Optional scale template
        self.scale_template = nn.Parameter(
            torch.zeros(num_gaussians, 3)
        )

        # --------------------------------------------------
        # Geometry heads (intrinsic, not view-dependent)
        # --------------------------------------------------
        self.offset_head = MLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=num_gaussians * 3,
            num_layers=2,
            activation=nn.ReLU,
        )

        self.anchor_scale_head = MLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=6,
            num_layers=2,
            activation=nn.ReLU,
        )

        self.cov_head = MLP(        # scale and quat
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=num_gaussians * 7,
            num_layers=2,
            activation=nn.ReLU,
        )

        # --------------------------------------------------
        # Appearance heads
        # If use_view_conditioning:
        #   input = [anchor_latent, view_dir(3), log_dist(1), inv_dist(1)]
        # else:
        #   input = anchor_latent
        # --------------------------------------------------
        self.view_feat_dim = 5  # (dir_x, dir_y, dir_z, log_dist, inv_dist)
        app_in_dim = hidden_dim + self.view_feat_dim if use_view_conditioning else hidden_dim

        self.opacity_head = MLP(
            in_dim=app_in_dim,
            hidden_dim=hidden_dim,
            out_dim=num_gaussians * 1,
            num_layers=2,
            activation=nn.ReLU,
        )

        self.color_head = MLP(
            in_dim=app_in_dim,
            hidden_dim=hidden_dim,
            out_dim=num_gaussians * 3,
            num_layers=2,
            activation=nn.ReLU,
        )

    def _build_anchor_input(
        self,
        dino_feat: torch.Tensor,
        confidence: torch.Tensor,
        cov_diag: torch.Tensor,
    ) -> torch.Tensor:
        """
        dino_feat:   [N, F]
        confidence:  [N] or [N, 1]
        cov_diag:    [N, 3]
        """
        if confidence.ndim == 1:
            confidence = confidence.unsqueeze(-1)

        return torch.cat([dino_feat, confidence, cov_diag], dim=-1)

    def _build_view_feature(
        self,
        anchor_xyz: torch.Tensor,
        camera_xyz: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        anchor_xyz: [N, 3]
        camera_xyz: [N, 3] or [1, 3]

        Returns:
            [N, 5] = [view_dir(3), log_dist(1), inv_dist(1)]
        """
        if camera_xyz.shape[0] == 1 and anchor_xyz.shape[0] > 1:
            camera_xyz = camera_xyz.expand(anchor_xyz.shape[0], -1)

        vec = anchor_xyz - camera_xyz
        dist = vec.norm(dim=-1, keepdim=True).clamp_min(eps)
        view_dir = vec / dist
        log_dist = dist.log()
        inv_dist = 1.0 / dist
        return torch.cat([view_dir, log_dist, inv_dist], dim=-1)

    def forward(
        self,
        anchor_xyz: torch.Tensor,
        dino_feat: torch.Tensor,
        confidence: torch.Tensor,
        cov_diag: torch.Tensor,
        camera_xyz: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        anchor_xyz: [N, 3]
        dino_feat:  [N, F]
        confidence: [N] or [N, 1]
        cov_diag:   [N, 3]
        camera_xyz: [N, 3] or [1, 3], optional if use_view_conditioning=False
        """

        def check_tensor(name, x):
            print(
                f"{name}: shape={tuple(x.shape)}, "
                f"nan={torch.isnan(x).any().item()}, "
                f"inf={torch.isinf(x).any().item()}, "
                f"min={x.nan_to_num().min().item():.6f}, "
                f"max={x.nan_to_num().max().item():.6f}, "
                f"mean={x.nan_to_num().mean().item():.6f}"
            )
        
        # check_tensor("anchor_xyz", anchor_xyz)
        # check_tensor("dino_feat", dino_feat)
        # check_tensor("confidence", confidence)
        # check_tensor("cov_diag", cov_diag)

        N = anchor_xyz.shape[0]
        K = self.num_gaussians

        # ---------------------------------------------
        # Anchor latent
        # ---------------------------------------------
        anchor_input = self._build_anchor_input(
            dino_feat=dino_feat,
            confidence=confidence,
            cov_diag=cov_diag,
        )  # [N, F+4]

        h = self.anchor_encoder(anchor_input)  # [N, H]

        # ---------------------------------------------
        # Geometry heads (intrinsic)
        # ---------------------------------------------
        delta_offsets = self.offset_head(h).view(N, K, 3)          # [N, K, 3]
        offsets = self.offset_template.unsqueeze(0) + delta_offsets

        anchor_scale_raw = self.anchor_scale_head(h)               # [N, 6]
        anchor_scale = F.softplus(anchor_scale_raw) + 1e-4   # [N, 6]
        offset_scale = anchor_scale[:, :3]
        gaussian_base_scale = anchor_scale[:, 3:]

        cov_raw = self.cov_head(h).view(N, K, 7)
        scale_raw = cov_raw[..., :3]
        quat_raw  = cov_raw[..., 3:7]

        # scale_raw = self.scale_head(h).view(N, K, 3)               # [N, K, 3]
        scales = gaussian_base_scale[:, None, :] * torch.sigmoid(scale_raw) + 1e-4

        # quat_raw = self.quat_head(h).view(N, K, 4)                 # [N, K, 4]
        quaternions = normalize_quaternion(quat_raw)

        centers = anchor_xyz[:, None, :] + offsets * offset_scale[:, None, :]

        # ---------------------------------------------
        # Appearance heads
        # ---------------------------------------------
        if self.use_view_conditioning:
            if camera_xyz is None:
                raise ValueError(
                    "camera_xyz must be provided when use_view_conditioning=True"
                )
            view_feat = self._build_view_feature(anchor_xyz, camera_xyz)  # [N, 5]
            app_input = torch.cat([h, view_feat], dim=-1)                # [N, H+5]
        else:
            app_input = h

        opacity_raw = self.opacity_head(app_input).view(N, K, 1)
        opacity = torch.sigmoid(opacity_raw)

        color_raw = self.color_head(app_input).view(N, K, 3)
        if self.color_act == "sigmoid":
            colors = torch.sigmoid(color_raw)
        elif self.color_act == "tanh":
            colors = torch.tanh(color_raw)
        else:
            colors = color_raw

        # print statistics for debugging
        # print(f"offsets: {offsets.mean().item():.4f} ± {offsets.std().item():.4f}")
        # print(f"scales: {scales.mean().item():.4f} ± {scales.std().item():.4f}")
        # print(f"quaternions: {quaternions.mean().item():.4f} ± {quaternions.std().item():.4f}")
        # print(f"opacity: {opacity.mean().item():.4f} ± {opacity.std().item():.4f}")
        # print(f"colors: {colors.mean().item():.4f} ± {colors.std().item():.4f}")

        return {
            "centers": centers,             # [N, K, 3]
            "offsets": offsets,             # [N, K, 3]
            "anchor_scale": anchor_scale,   # [N, 1]
            "scales": scales,               # [N, K, 3]
            "quaternions": quaternions,     # [N, K, 4]
            "opacity": opacity,             # [N, K, 1]
            "colors": colors,               # [N, K, 3]
            "anchor_latent": h,             # [N, H]
        }