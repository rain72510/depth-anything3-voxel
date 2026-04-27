# Auto-extracted from scripts/train_voxel_decoder.py
# Module: losses

from typing import Dict, Any
import torch
import torch.nn.functional as F
from depth_anything_3.utils.loss_utils import ssim

def quaternion_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (w, x, y, z) to rotation matrices.

    q: [..., 4]
    returns: [..., 3, 3]
    """
    # normalize for safety
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)
    return R


def per_gaussian_normal_alignment_loss(
    scales: torch.Tensor,            # [N, K, 3]
    rotations: torch.Tensor,         # [N, K, 4]
    voxel_var_points: torch.Tensor,  # [N, 3]
):
    """Encourage each Gaussian's smallest-scale axis (in world frame) to align
    with the local surface normal — proxied by the axis of smallest variance
    in the contributing-points distribution per anchor.

    Axis-aligned approximation: works well when surfaces are roughly aligned
    with world XYZ (driving: ground=Z-thin, walls=Y-thin or X-thin). For
    arbitrary orientations, full eigendecomposition would be needed.
    """
    import torch.nn.functional as _F
    # Surface normal target: axis of smallest per-anchor variance
    target_axis = voxel_var_points.argmin(dim=-1)                      # [N]
    target_normal = _F.one_hot(target_axis, num_classes=3).float().to(scales.device)  # [N, 3]

    # Each Gaussian's smallest local-axis index
    gauss_axis = scales.argmin(dim=-1)                                  # [N, K]
    local_hot = _F.one_hot(gauss_axis, num_classes=3).float()           # [N, K, 3]

    # Rotate that local axis into world frame
    R = quaternion_to_rotmat(rotations)                                 # [N, K, 3, 3]
    gauss_normal_world = (R @ local_hot.unsqueeze(-1)).squeeze(-1)      # [N, K, 3]

    # Compare to target — sign-ambiguous (n vs -n same plane), so use abs
    target_b = target_normal.unsqueeze(1)                               # [N, 1, 3]
    cos_sim = (gauss_normal_world * target_b).sum(dim=-1)               # [N, K]
    return (1.0 - cos_sim.abs()).mean()


def masked_l1_loss(pred, gt, valid_mask, eps=1e-8):
    # pred, gt: [B,3,H,W]
    # valid_mask: [B,H,W] or [B,1,H,W], True=keep
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    valid_mask = valid_mask.float()

    diff = (pred - gt).abs() * valid_mask
    denom = valid_mask.sum() * pred.shape[1]
    return diff.sum() / denom.clamp_min(eps)


def depth_to_normals(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Compute world-aligned per-pixel normals from depth + intrinsics.

    depth: [B, H, W] (or [H, W])
    intrinsics: [B, 3, 3] (or [3, 3])
    returns: [B, H-2, W-2, 3] unit-norm normals (cropped 1px on each side
             to keep finite differences valid)
    """
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
        intrinsics = intrinsics.unsqueeze(0)
    B, H, W = depth.shape
    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)

    u = torch.arange(W, device=depth.device, dtype=depth.dtype).view(1, 1, W).expand(B, H, W)
    v = torch.arange(H, device=depth.device, dtype=depth.dtype).view(1, H, 1).expand(B, H, W)
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    pts = torch.stack([X, Y, Z], dim=-1)  # [B, H, W, 3]

    dx = pts[:, 1:-1, 2:, :] - pts[:, 1:-1, :-2, :]   # right - left
    dy = pts[:, 2:, 1:-1, :] - pts[:, :-2, 1:-1, :]   # down - up
    n = torch.cross(dx, dy, dim=-1)                    # [B, H-2, W-2, 3]
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-6)
    return n


def masked_normal_consistency_loss(
    pred_depth: torch.Tensor,    # [B, H, W]
    gt_depth: torch.Tensor,      # [B, H, W]
    intrinsics: torch.Tensor,    # [B, 3, 3]
    valid_mask: torch.Tensor,    # [B, H, W] bool
    eps: float = 1e-8,
):
    """1 - cos(normal(pred_depth), normal(gt_depth)) averaged over masked pixels."""
    n_pred = depth_to_normals(pred_depth, intrinsics)  # [B, H-2, W-2, 3]
    n_gt = depth_to_normals(gt_depth, intrinsics)
    cos = (n_pred * n_gt).sum(dim=-1)  # [B, H-2, W-2]
    err = 1.0 - cos                    # 0 = perfect alignment, 2 = opposite
    # crop mask to match (drop 1px border on each side)
    m = valid_mask.float()[:, 1:-1, 1:-1]
    return (err * m).sum() / m.sum().clamp_min(eps)

def compute_photometric_loss(
    decoder_out: Dict[str, torch.Tensor],
    voxel_dict: Dict[str, Any],
    rendered_rgb: torch.Tensor,   # [v,3,H,W]
    gt_rgb: torch.Tensor,         # [v,3,H,W]
    valid_mask=None,   # [v,H,W], True=non-sky
    lambda_photo: float = 1.0,
    lambda_color: float = 0.05,
    lambda_offset: float = 1e-2,
    lambda_scale: float = 1e-1,
    lambda_opacity: float = 1e-2,
    lambda_anchor: float = 1e-4,
    lambda_disp: float = 1e-3,
    # lambda_dssim: float = 0.02,
    lambda_dssim: float = 0,
    lambda_scale_vol: float = 1e-2,
    lambda_lpips: float = 0.05,
    lambda_aniso: float = 0.05,
    lambda_sky: float = 1.0,
    lambda_depth: float = 0.0,
    lambda_normal: float = 0.0,
    lambda_shape: float = 0.0,
    lambda_normal_align: float = 0.0,
    lpips_fn=None,
    sky_rgb: torch.Tensor = None,         # [v,3,H,W] predicted sky, optional
    sky_mask_bool: torch.Tensor = None,   # [v,H,W] True=sky, optional
    rendered_depth: torch.Tensor = None,  # [v,H,W] gsplat output, optional
    gt_depth: torch.Tensor = None,        # [v,H,W] DA3 depth, optional
    intrinsics: torch.Tensor = None,      # [v,3,3] for normal computation, optional
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

    # print the first few pixel values of rendered_rgb and gt_rgb for debugging
    # print(f"rendered_rgb sample: {rendered_rgb.view(3, -1)[:, :5]}")
    # print(f"gt_rgb sample: {gt_rgb.view(3, -1)[:, :5]}")

    # print shape, dtype, device, min, max of rendered_rgb and gt_rgb for debugging
    # print(f"rendered_rgb: shape={rendered_rgb.shape}, dtype={rendered_rgb.dtype}, device={rendered_rgb.device}, "
    #       f"min={rendered_rgb.min().item():.4f}, max={rendered_rgb.max().item():.4f}")
    # print(f"gt_rgb: shape={gt_rgb.shape}, dtype={gt_rgb.dtype}, device={gt_rgb.device}, "
    #       f"min={gt_rgb.min().item():.4f}, max={gt_rgb.max().item():.4f}")

    if valid_mask is None:
        losses["photo"] = lambda_photo * F.l1_loss(rendered_rgb, gt_rgb)
    else:
        losses["photo"] = lambda_photo * masked_l1_loss(rendered_rgb, gt_rgb, valid_mask)
    # losses["photo"] = lambda_photo * F.l1_loss(rendered_rgb, gt_rgb)

    # for k, v in decoder_out.items():
    #     print(f"decoder_out[{k}]: shape={v.shape}, dtype={v.dtype}, device={v.device}, "
    #           f"min={v.min().item():.4f}, max={v.max().item():.4f}")
    
    # decoder_out["colors"] shape
    # print(f"decoder_out['colors']: shape={decoder_out['colors'].shape}, dtype={decoder_out['colors'].dtype}, device={decoder_out['colors'].device}, "
    #       f"min={decoder_out['colors'].min().item():.4f}, max={decoder_out['colors'].max().item():.4f}")

    # # 2. optional voxel color prior
    if voxel_dict.get("voxel_colors", None) is not None:
        target_color = voxel_dict["voxel_colors"].to(decoder_out["colors"].device).float()/255.0
        pred_color = decoder_out["colors"].mean(dim=1)
        # print dim, dtype, device, min, max of pred_color and target_color for debugging
        # print(f"pred_color: shape={pred_color.shape}, dtype={pred_color.dtype}, device={pred_color.device}, "
        #       f"min={pred_color.min().item():.4f}, max={pred_color.max().item():.4f}")
        # print(f"target_color: shape={target_color.shape}, dtype={target_color.dtype}, device={target_color.device}, "
        #       f"min={target_color.min().item():.4f}, max={target_color.max().item():.4f}")

        losses["color"] = lambda_color * F.l1_loss(pred_color, target_color)
    else:
        losses["color"] = lambda_color * torch.tensor(0.0, device=decoder_out["colors"].device)

    # # 3. geometry / regularization
    losses["offset_reg"] = lambda_offset * decoder_out["offsets"].pow(2).mean()
    # print(f"offset_reg: {losses['offset_reg'].item():.6f}")
    losses["scale_reg"] = lambda_scale * decoder_out["scales"].pow(2).mean()
    losses["scale_vol_reg"] = lambda_scale_vol * decoder_out["scales"].prod(dim=1).mean()
    losses["opacity_reg"] = lambda_opacity * decoder_out["opacity"].mean()

    if lambda_aniso > 0:
        _s = decoder_out["scales"]
        _aniso_ratio = _s.max(dim=-1).values / (_s.mean(dim=-1) + 1e-6)
        losses["aniso"] = lambda_aniso * (_aniso_ratio - 1.0).pow(2).mean()
    else:
        losses["aniso"] = torch.tensor(0.0, device=decoder_out["scales"].device)
    # losses["anchor_scale_reg"] = decoder_out["anchor_scale"].pow(2).mean()

    # Shape supervision: match per-Gaussian scales to the local point distribution's std.
    # voxel_var_points [N,3] is the variance of contributing 3D points per voxel; sqrt gives
    # the local std along XYZ. Each anchor produces K Gaussians sharing the anchor; we
    # constrain their scales toward the anchor's std so flat surfaces yield flat Gaussians,
    # poles yield needles, etc. Per-Gaussian residual still allowed via the K freedom.
    if (
        lambda_shape > 0
        and "voxel_var_points" in voxel_dict
        and voxel_dict["voxel_var_points"] is not None
    ):
        target_std = voxel_dict["voxel_var_points"].clamp_min(1e-8).sqrt().to(
            decoder_out["scales"].device
        )                                          # [N, 3]
        target_std = target_std.unsqueeze(1)       # [N, 1, 3] -> broadcasts over K
        scales = decoder_out["scales"]             # [N, K, 3]
        # log-ratio loss is scale-invariant; clamps prevent log(0)
        log_ratio = (scales.clamp_min(1e-8) / target_std.clamp_min(1e-8)).log()
        losses["shape"] = lambda_shape * log_ratio.pow(2).mean()
    else:
        losses["shape"] = torch.tensor(0.0, device=decoder_out["scales"].device)

    # Per-Gaussian normal alignment: rotate each Gaussian's smallest local axis
    # to match the per-anchor normal proxy (axis of min voxel variance).
    if (
        lambda_normal_align > 0
        and "voxel_var_points" in voxel_dict
        and voxel_dict["voxel_var_points"] is not None
        and "rotations" in decoder_out
    ):
        losses["normal_align"] = lambda_normal_align * per_gaussian_normal_alignment_loss(
            scales=decoder_out["scales"],
            rotations=decoder_out["rotations"],
            voxel_var_points=voxel_dict["voxel_var_points"].to(decoder_out["scales"].device),
        )
    else:
        losses["normal_align"] = torch.tensor(0.0, device=decoder_out["scales"].device)

    # # actual displacement regularization
    # disp = decoder_out["offsets"] * decoder_out["anchor_scale"].unsqueeze(-1)
    # losses["disp_reg"] = disp.pow(2).mean()
    

    ssim_val = ssim(rendered_rgb, gt_rgb)
    dssim = (1.0 - ssim_val) / 2.0
    losses["dssim"] = lambda_dssim * dssim

    # print all loss components for debugging
    # for k, v in losses.items():
    #     print(f"{k} loss: {v.item():.6f}")

    # Sky MLP loss — L1 on sky pixels only
    if sky_rgb is not None and sky_mask_bool is not None and lambda_sky > 0:
        losses["sky"] = lambda_sky * masked_l1_loss(sky_rgb, gt_rgb, sky_mask_bool)
    else:
        losses["sky"] = torch.tensor(0.0, device=rendered_rgb.device)

    # Depth supervision: L1 between rendered_depth and DA3 depth on non-sky pixels.
    if (
        lambda_depth > 0
        and rendered_depth is not None
        and gt_depth is not None
        and valid_mask is not None
    ):
        # rendered_depth: [v,H,W]; gt_depth: [v,H,W]; valid_mask: [v,H,W] True=non-sky
        m = valid_mask.float()
        diff = (rendered_depth - gt_depth.to(rendered_depth.device)).abs() * m
        denom = m.sum().clamp_min(1e-8)
        losses["depth"] = lambda_depth * (diff.sum() / denom)
    else:
        losses["depth"] = torch.tensor(0.0, device=rendered_rgb.device)

    # Normal-from-depth consistency: 1 - cos(normal(rendered), normal(gt)) over non-sky.
    if (
        lambda_normal > 0
        and rendered_depth is not None
        and gt_depth is not None
        and intrinsics is not None
        and valid_mask is not None
    ):
        losses["normal"] = lambda_normal * masked_normal_consistency_loss(
            rendered_depth,
            gt_depth.to(rendered_depth.device),
            intrinsics.to(rendered_depth.device),
            valid_mask,
        )
    else:
        losses["normal"] = torch.tensor(0.0, device=rendered_rgb.device)

    # LPIPS loss (same sky mask applied)
    if lpips_fn is not None and lambda_lpips > 0:
        if valid_mask is not None:
            mask4 = valid_mask.unsqueeze(1).float()  # [v,1,H,W]
            r_masked = rendered_rgb * mask4
            g_masked = gt_rgb * mask4
        else:
            r_masked = rendered_rgb
            g_masked = gt_rgb
        lpips_val = lpips_fn(r_masked.cpu() * 2 - 1, g_masked.cpu() * 2 - 1).mean().to(rendered_rgb.device)
        losses["lpips"] = lambda_lpips * lpips_val
    else:
        losses["lpips"] = torch.tensor(0.0, device=rendered_rgb.device)

    losses["total"] = (
        losses["photo"]
        + losses["color"]
        + losses["offset_reg"]
        + losses["scale_reg"]
        + losses["scale_vol_reg"]
        + losses["opacity_reg"]
        # + losses["anchor_scale_reg"]
        # + losses["disp_reg"]
        + losses["dssim"]
        + losses["lpips"]
        + losses["aniso"]
        + losses["sky"]
        + losses["depth"]
        + losses["normal"]
        + losses["shape"]
        + losses["normal_align"]

    )
    return losses
