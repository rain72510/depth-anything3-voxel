# Auto-extracted from scripts/train_voxel_decoder.py
# Module: rendering

from typing import Dict
import torch
from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode, render_3dgs

# sibling imports (auto-generated)
from depth_anything_3.training.data_prep import flatten_gaussians, build_renderer_gaussians

def render_views_from_decoder_output(
    decoder_out: Dict[str, torch.Tensor],
    intrinsics: torch.Tensor,   # [V,3,3]
    extrinsics: torch.Tensor,   # [V,4,4] or [V,3,4]
    image_hw: tuple[int, int],
    view_indices: torch.Tensor,
    chunk_size: int = 2,
):
    
    def ensure_homogeneous_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
        """
        extrinsics:
            [V,4,4] or [V,3,4]
        returns:
            [V,4,4]
        """
        if extrinsics.shape[-2:] == (4, 4):
            return extrinsics

        if extrinsics.shape[-2:] == (3, 4):
            V = extrinsics.shape[0]
            bottom = torch.zeros(V, 1, 4, device=extrinsics.device, dtype=extrinsics.dtype)
            bottom[:, 0, 3] = 1.0
            extrinsics = torch.cat([extrinsics, bottom], dim=1)
            return extrinsics

        raise ValueError(f"Unsupported extrinsics shape: {extrinsics.shape}")

    def normalize_intrinsics(intrinsics: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        intrinsics: [V,3,3] in pixel coordinates
        returns:    [V,3,3] normalized by image width/height
        """
        K = intrinsics.clone()
        K[:, 0, 0] = K[:, 0, 0] / W   # fx
        K[:, 1, 1] = K[:, 1, 1] / H   # fy
        K[:, 0, 2] = K[:, 0, 2] / W   # cx
        K[:, 1, 2] = K[:, 1, 2] / H   # cy
        return K

    flat_scene = flatten_gaussians(decoder_out)
    gs_world = build_renderer_gaussians(flat_scene)
    # print("[DEBUG] built gaussian.means.shape      =", gs_world.means.shape)
    # print("[DEBUG] built gaussian.scales.shape     =", gs_world.scales.shape)
    # print("[DEBUG] built gaussian.rotations.shape  =", gs_world.rotations.shape)
    # print("[DEBUG] built gaussian.opacities.shape  =", gs_world.opacities.shape)
    # print("[DEBUG] built gaussian.harmonics.shape  =", gs_world.harmonics.shape)
    # print("[DEBUG] built gaussian.means.dtype      =", gs_world.means.dtype)
    # print("[DEBUG] built gaussian.means.device     =", gs_world.means.device)

    H, W = image_hw

    extrinsics = ensure_homogeneous_extrinsics(extrinsics)
    intrinsics = normalize_intrinsics(intrinsics, H=H, W=W)
    view_indices = torch.as_tensor(view_indices, device=extrinsics.device, dtype=torch.long).reshape(-1)
    if len(view_indices) == 1:
        extr_sel = extrinsics.index_select(0, view_indices)
        intr_sel = intrinsics.index_select(0, view_indices)
        color, depth = render_3dgs(
            gaussian=gs_world,
            extrinsics=extr_sel,
            intrinsics=intr_sel,
            image_shape=image_hw,
            chunk_size=chunk_size,
            trj_mode="original",
            # use_sh=True,
            use_sh=False,
            color_mode="RGB+ED",
            enable_tqdm=False,
        )
    else:
        extr_sel = extrinsics.index_select(0, view_indices).unsqueeze(0)
        intr_sel = intrinsics.index_select(0, view_indices).unsqueeze(0)   # [v,3,3]
        color, depth = run_renderer_in_chunk_w_trj_mode(
            gaussians=gs_world,
            extrinsics=extr_sel,
            intrinsics=intr_sel,
            image_shape=image_hw,
            chunk_size=chunk_size,
            trj_mode="original",
            use_sh=True,
            color_mode="RGB+ED",
            enable_tqdm=False,
        )

    # print(f"extr_sel: shape={extr_sel.shape}, dtype={extr_sel.dtype}, device={extr_sel.device}")
    # print(f"intr_sel: shape={intr_sel.shape}, dtype={intr_sel.dtype}, device={intr_sel.device}")


    # DA3 export code uses color[idx] as a video tensor, so color is batched at dim 0.
    # print color.shape, depth.shape
    # print(f"Rendered color: shape={color.shape}, dtype={color.dtype}, device={color.device}")
    # print(f"Rendered depth: shape={depth.shape}, dtype={depth.dtype}, device={depth.device}")
    rendered_rgb = color[0]   # [v,3,H,W]
    rendered_depth = depth[0] if depth.ndim >= 4 else depth
    # print(f"Extracted rendered_rgb: shape={rendered_rgb.shape}, dtype={rendered_rgb.dtype}, device={rendered_rgb.device}")
    # print(f"Extracted rendered_depth: shape={rendered_depth.shape}, dtype={rendered_depth.dtype}, device={rendered_depth.device}")

    return rendered_rgb, rendered_depth, flat_scene
