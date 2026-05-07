# Auto-extracted from scripts/train_voxel_decoder.py
# Module: checkpoint_io

import os
import glob
from pathlib import Path
from typing import Dict
import torch
import numpy as np
from depth_anything_3.utils.gsply_helpers import export_ply

# sibling imports (auto-generated)

def save_gaussian_scene_npz(scene: Dict[str, torch.Tensor], path: str):
    np.savez_compressed(
        path,
        means3D=scene["means3D"].detach().cpu().numpy(),
        scales=scene["scales"].detach().cpu().numpy(),
        rotations=scene["rotations"].detach().cpu().numpy(),
        opacity=scene["opacity"].detach().cpu().numpy(),
        colors=scene["colors"].detach().cpu().numpy(),
    )

SH_C0 = 0.28209479177387814  # 1 / (2 sqrt(pi)) — DC SH coefficient


def save_flat_scene_as_ply(
    flat_scene: dict,
    save_path: str,
    save_sh_dc_only: bool = True,
    shift_and_scale: bool = False,
):
    """
    flat_scene:
        means3D   [M,3]
        scales    [M,3]
        rotations [M,4]
        opacity   [M,1] or [M]   in [0,1] (post-sigmoid)
        colors    [M,3]          in [0,1] (post-sigmoid)

    The 3DGS PLY format expected by viewers (SuperSplat, gsplat-viewer, etc.)
    stores `f_dc_*` in SH-DC convention `(rgb - 0.5) / SH_C0` and `opacity`
    in logit space. Our decoder outputs already-sigmoided values, so we
    invert those transforms here before handing to `export_ply`. (Without
    this conversion, viewers display a washed-out gray scene because they
    apply the inverse transforms expecting unencoded values.)
    """
    means = flat_scene["means3D"].detach()
    scales = flat_scene["scales"].detach().clamp_min(1e-8)
    rotations = flat_scene["rotations"].detach()
    opacities = flat_scene["opacity"].detach().reshape(-1).clamp(1e-6, 1 - 1e-6)
    colors = flat_scene["colors"].detach().clamp(0.0, 1.0)

    # Convert to viewer-compatible 3DGS PLY conventions.
    f_dc = (colors - 0.5) / SH_C0                                          # [M, 3]
    opacity_logit = torch.log(opacities / (1.0 - opacities))               # [M]

    # SH degree 0 only: [M, 3, 1]
    harmonics = f_dc.unsqueeze(-1)

    export_ply(
        means=means,
        scales=scales,
        rotations=rotations,
        harmonics=harmonics,
        opacities=opacity_logit,
        path=Path(save_path),
        shift_and_scale=shift_and_scale,
        save_sh_dc_only=save_sh_dc_only,
        match_3dgs_mcmc_dev=False,
    )

def save_debug_ply_pair(
    flat_scene: dict,
    voxel_mean_points,  # [K, 3] tensor or None
    output_dir: str,
    global_step: int,
    scene_name: str,
    keep_last_k: int = 20,
):
    """Save Gaussian PLY + voxel XYZ PLY for a logged render_samples step."""
    import glob as _glob
    debug_dir = os.path.join(output_dir, "debug_ply")
    os.makedirs(debug_dir, exist_ok=True)

    prefix = f"step_{global_step:07d}_{scene_name}"

    # 3DGS PLY (with learned colors)
    gs_path = os.path.join(debug_dir, prefix + "_gaussians.ply")
    save_flat_scene_as_ply(flat_scene, gs_path)

    # Geometry-only PLY: same Gaussians, uniform mid-gray color so geometry
    # (positions, scales, rotations, opacity) is visible without color noise.
    import torch as _torch
    geom_scene = {
        **flat_scene,
        "colors": _torch.full_like(flat_scene["colors"], 0.5),
    }
    geom_path = os.path.join(debug_dir, prefix + "_geom.ply")
    save_flat_scene_as_ply(geom_scene, geom_path)

    # Voxel XYZ PLY (anchor scaffold)
    if voxel_mean_points is not None:
        from plyfile import PlyData, PlyElement
        import numpy as np
        pts = voxel_mean_points.float().numpy()
        el = PlyElement.describe(
            np.array([(pts[i, 0], pts[i, 1], pts[i, 2]) for i in range(len(pts))],
                     dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]),
            "vertex",
        )
        vox_path = os.path.join(debug_dir, prefix + "_voxels.ply")
        PlyData([el]).write(vox_path)

    # Keep only last K pairs (also clean up _geom.ply and _voxels.ply siblings)
    all_gs = sorted(_glob.glob(os.path.join(debug_dir, "step_*_gaussians.ply")), key=os.path.getmtime)
    if len(all_gs) > keep_last_k:
        for old_path in all_gs[:-keep_last_k]:
            os.remove(old_path)
            for sibling_suffix in ("_voxels.ply", "_geom.ply"):
                sibling = old_path.replace("_gaussians.ply", sibling_suffix)
                if os.path.exists(sibling):
                    os.remove(sibling)

def save_recent_training_ply(
    flat_scene: dict,
    output_dir: str,
    epoch: int,
    scene_name: str,
    keep_last_k: int = 10,
):
    ply_root = os.path.join(output_dir, "train_ply")
    os.makedirs(ply_root, exist_ok=True)

    save_path = os.path.join(
        ply_root,
        f"epoch_{epoch:04d}_{scene_name}.ply"
    )
    save_flat_scene_as_ply(flat_scene, save_path)

    all_ply = sorted(
        glob.glob(os.path.join(ply_root, "epoch_*.ply")),
        key=os.path.getmtime
    )
    if len(all_ply) > keep_last_k:
        for old_path in all_ply[:-keep_last_k]:
            os.remove(old_path)
            print(f"[INFO] Removed old ply: {old_path}")

def save_checkpoint(
    decoder,
    optimizer,
    epoch,
    output_dir,
    keep_last_k=5,
):
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:04d}.pth")

    torch.save({
        "epoch": epoch,
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, ckpt_path)

    print(f"[INFO] Saved checkpoint: {ckpt_path}")

    # 刪舊的
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_epoch_*.pth")))
    if len(ckpts) > keep_last_k:
        for old_ckpt in ckpts[:-keep_last_k]:
            os.remove(old_ckpt)
