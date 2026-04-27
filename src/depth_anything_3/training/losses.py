# Auto-extracted from scripts/train_voxel_decoder.py
# Module: losses

from typing import Dict, Any
import torch
import torch.nn.functional as F
from depth_anything_3.utils.loss_utils import ssim

def masked_l1_loss(pred, gt, valid_mask, eps=1e-8):
    # pred, gt: [B,3,H,W]
    # valid_mask: [B,H,W] or [B,1,H,W], True=keep
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    valid_mask = valid_mask.float()

    diff = (pred - gt).abs() * valid_mask
    denom = valid_mask.sum() * pred.shape[1]
    return diff.sum() / denom.clamp_min(eps)

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
    lpips_fn=None,
    sky_rgb: torch.Tensor = None,         # [v,3,H,W] predicted sky, optional
    sky_mask_bool: torch.Tensor = None,   # [v,H,W] True=sky, optional
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

    )
    return losses
