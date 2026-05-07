# Auto-extracted from scripts/train_voxel_decoder.py
# Module: validation

import os
import torch
import traceback

# sibling imports (auto-generated)
from depth_anything_3.training.rendering import render_views_from_decoder_output
from depth_anything_3.training.data_prep import build_decoder_inputs
from depth_anything_3.training.checkpoint_io import save_flat_scene_as_ply
from depth_anything_3.training.sky_mask import load_precomputed_sky_mask
from depth_anything_3.training.losses import masked_l1_loss
from depth_anything_3.training.utils import select_valid_view_indices, save_tensor_image, save_diff_image

@torch.no_grad()
def verify_scene_no_cache(
    decoder,
    model,
    voxelizer,
    image_paths,
    scene_name,
    output_dir,
    cache_root,
    epoch,
    device,
    view_indices=(0,10,20),
):
    decoder.eval()

    print(f"[VERIFY] preparing scene {scene_name}")

    prediction = model.inference(image_paths)

    # sky_mask_np = build_sky_mask_from_gt(
    #     prediction.processed_images,
    #     prediction.depth,
    #     prediction.conf,
    # )

    sky_mask_np = load_precomputed_sky_mask(
        cache_dir=os.path.join(cache_root, scene_name),
        images_u8=prediction.processed_images,
    )

    gt_images = (
        torch.from_numpy(prediction.processed_images)
        .permute(0,3,1,2)
        .contiguous()
        .float() / 255.0
    )

    intrinsics = torch.from_numpy(prediction.intrinsics).float().to(device)
    extrinsics = torch.from_numpy(prediction.extrinsics).float().to(device)

    V, _, H, W = gt_images.shape
    valid_view_indices = select_valid_view_indices(V, list(view_indices))
    view_tensor = torch.tensor(valid_view_indices, device=device, dtype=torch.long)

    voxel_dict = voxelizer.voxelize_prediction(prediction)
    decoder_inputs = build_decoder_inputs(voxel_dict, device=device)
    camera_xyz = extrinsics[:, :3, 3].to(device)

    rendered_rgbs = []
    flat_scene = None

    for view in valid_view_indices:
        extra_kwargs = {}
        if decoder_inputs.get("voxel_pixel_features", None) is not None:
            extra_kwargs["voxel_pixel_features"] = decoder_inputs["voxel_pixel_features"]
        decoder_out = decoder(
            anchor_xyz=decoder_inputs["anchor_xyz"],
            camera_xyz=camera_xyz[view:view+1],
            dino_feat=decoder_inputs["dino_feat"],
            confidence=decoder_inputs["confidence"],
            cov_diag=decoder_inputs["cov_diag"],
            voxel_colors=decoder_inputs["voxel_colors"],
            **extra_kwargs,
        )

        pred_rgb, pred_depth, flat_scene = render_views_from_decoder_output(
            decoder_out=decoder_out,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_hw=(H, W),
            view_indices=[view],
        )

        rendered_rgbs.append(pred_rgb)   # [3,H,W]

    rendered_rgbs = torch.stack(rendered_rgbs, dim=0)   # [v,3,H,W]

    gt_images = gt_images.to(device)
    gt_rgb = gt_images[view_tensor]                     # [v,3,H,W]

    sky_mask = torch.from_numpy(sky_mask_np).to(device) # [V,H,W]
    sky_mask = sky_mask[view_tensor]                    # [v,H,W]
    valid_mask = ~sky_mask

    l1 = masked_l1_loss(rendered_rgbs, gt_rgb, valid_mask).item()

    scene_dir = os.path.join(output_dir, "verification", f"epoch_{epoch:04d}", scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    # print gt_rgb, rendered_rgbs
    # print(f"[VERIFY] gt_rgb: shape={gt_rgb.shape}, dtype={gt_rgb.dtype}, device={gt_rgb.device}, "
    #       f"min={gt_rgb.min().item():.4f}, max={gt_rgb.max().item():.4f}")
    # print(f"[VERIFY] rendered_rgbs: shape={rendered_rgbs.shape}, dtype={rendered_rgbs.dtype}, device={rendered_rgbs.device}, "
    #       f"min={rendered_rgbs.min().item():.4f}, max={rendered_rgbs.max().item():.4f}")
    # print(f"[VERIFY] sky_mask: shape={sky_mask.shape}, dtype={sky_mask.dtype}, device={sky_mask.device}, "
    #       f"num_sky_pixels={sky_mask.sum().item()}, num_valid_pixels={valid_mask.sum().item()}")

    for i, view_idx in enumerate(valid_view_indices):
        save_tensor_image(gt_rgb[i], f"{scene_dir}/view_{view_idx}_gt.png")
        save_tensor_image(rendered_rgbs[i], f"{scene_dir}/view_{view_idx}_pred.png")
        save_diff_image(rendered_rgbs[i], gt_rgb[i], f"{scene_dir}/view_{view_idx}_diff.png")

    save_flat_scene_as_ply(
        flat_scene,
        os.path.join(scene_dir, "gaussian_scene.ply"),
    )

    print(f"[VERIFY] {scene_name} L1 = {l1:.6f}")

    return {
        "scene_name": scene_name,
        "l1": l1,
        "pred_rgb": rendered_rgbs.detach().cpu(),
        "gt_rgb": gt_rgb.detach().cpu(),
        "sky_mask": sky_mask.detach().cpu(),
        "view_indices": valid_view_indices,
        "num_voxels": int(voxel_dict["num_voxels"]),
        "num_gaussians": int(flat_scene["means3D"].shape[0]),
    }

def safe_verify_scene_no_cache(*args, **kwargs):
    try:
        out = verify_scene_no_cache(*args, **kwargs)
        return {
            "ok": True,
            "result": out,
            "error": None,
        }
    except Exception as e:
        print(f"[WARN] verify_scene_no_cache failed: {e}")
        traceback.print_exc()

        # 嘗試同步，讓錯誤更完整地暴露
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        return {
            "ok": False,
            "result": None,
            "error": repr(e),
        }
