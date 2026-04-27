# Auto-extracted from scripts/train_voxel_decoder.py
# Module: train_step

import os
import torch
from depth_anything_3.model.voxel_gaussian_decoder import VoxelGaussianDecoder
from depth_anything_3.model.sky_mlp import compute_ray_dirs_world

# sibling imports (auto-generated)
from depth_anything_3.training.losses import compute_photometric_loss
from depth_anything_3.training.rendering import render_views_from_decoder_output
from depth_anything_3.training.scene_cache import prepare_scene_cache
from depth_anything_3.training.scene_discovery import sample_even_odd_window

def train_one_step_on_scene(
    decoder: VoxelGaussianDecoder,
    optimizer: torch.optim.Optimizer,
    scene_cache,
    device: torch.device,
    views_per_step: int = 2,
    render_chunk_size: int = 2,
    lpips_fn=None,
    sky_mlp=None,          # Optional[SkyMLP]
    lambda_sky: float = 1.0,
):
    decoder.train()
    optimizer.zero_grad()

    dec_in = scene_cache["decoder_inputs"]
    voxel_dict = scene_cache["voxel_dict"]
    gt_images = scene_cache["images"]         # [V,3,H,W]
    intrinsics = scene_cache["intrinsics"]    # [V,3,3]
    extrinsics = scene_cache["extrinsics"]    # [V,4,4] or [V,3,4]
    camera_xyz = scene_cache["camera_xyz"]      # [V,3]
    sky_mask_all = scene_cache["sky_mask"]   # [V,H,W]


    gt_images = (
        torch.from_numpy(gt_images)
        .permute(0,3,1,2)
        .contiguous()
        .float() / 255.0
    )
    if gt_images.max() > 1.0:
        gt_images = gt_images / 255.0

    V, C, H, W = gt_images.shape
    if V == 0:
        raise RuntimeError("No supervision views available in scene_cache.")

    supervision_global = scene_cache.get("supervision_indices", None)
    input_global = scene_cache.get("input_indices", None)

    if supervision_global is not None and input_global is not None:
        input_global_set = set(input_global)

        seen_local = [i for i, g in enumerate(supervision_global) if g in input_global_set]
        novel_local = [i for i, g in enumerate(supervision_global) if g not in input_global_set]

        target_seen = views_per_step // 2
        target_novel = views_per_step - target_seen

        sampled = []

        if len(seen_local) > 0:
            num_seen = min(target_seen, len(seen_local))
            seen_perm = torch.randperm(len(seen_local), device=device)[:num_seen]
            sampled.extend([seen_local[i] for i in seen_perm.cpu().tolist()])

        if len(novel_local) > 0:
            num_novel = min(target_novel, len(novel_local))
            novel_perm = torch.randperm(len(novel_local), device=device)[:num_novel]
            sampled.extend([novel_local[i] for i in novel_perm.cpu().tolist()])

        # 如果某一邊不夠，就從另一邊補足
        if len(sampled) < min(views_per_step, V):
            remaining_pool = [i for i in range(V) if i not in sampled]
            need = min(views_per_step, V) - len(sampled)
            if len(remaining_pool) > 0:
                extra_perm = torch.randperm(len(remaining_pool), device=device)[:need]
                sampled.extend([remaining_pool[i] for i in extra_perm.cpu().tolist()])

        view_indices = torch.tensor(sampled, device=device, dtype=torch.long)
        num_views = len(sampled)
    else:
        num_views = min(views_per_step, V)
        view_indices = torch.randperm(V, device=device)[:num_views]

    num_seen_sampled = 0
    num_novel_sampled = 0
    if supervision_global is not None and input_global is not None:
        input_global_set = set(input_global)
        sampled_global = [supervision_global[i] for i in view_indices.cpu().tolist()]
        num_seen_sampled = sum([g in input_global_set for g in sampled_global])
        num_novel_sampled = len(sampled_global) - num_seen_sampled

    # Now we need to select the camera_xyz to input into the decoder.

    decoder_outs = []
    rendered_rgbs = []

    total_loss = 0.0
    loss_photo_sum = 0.0
    loss_lpips_sum = 0.0
    loss_offset_reg_sum = 0.0
    loss_scale_reg_sum = 0.0
    loss_opacity_reg_sum = 0.0
    loss_dssim_sum = 0.0
    loss_color_sum = 0.0
    loss_scale_vol_reg_sum = 0.0
    loss_aniso_sum = 0.0
    loss_sky_sum = 0.0
    delta_color_abs_mean_sum = 0.0
    rendered_rgbs_to_log = []
    gt_rgbs_to_log = []
    sky_masks_to_log = []

    flat_scene_stats = None
    
    for i, view in enumerate(view_indices):
        # start = time.time()
        decoder_out = decoder(
            anchor_xyz=dec_in["anchor_xyz"],
            camera_xyz=camera_xyz[view:view+1],  # select one view's camera_xyz at a time, shape [1,3]
            dino_feat=dec_in["dino_feat"],
            confidence=dec_in["confidence"],
            cov_diag=dec_in["cov_diag"],
            voxel_colors=dec_in["voxel_colors"],
        )
        # print(f"Decoder forward pass done. Time: {time.time() - start:.2f} seconds.")

        # start = time.time()
        rendered_rgb, rendered_depth, flat_scene = render_views_from_decoder_output(
            decoder_out=decoder_out,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            image_hw=(H, W),
            view_indices=[view],
            # view_indices=view_indices,
            chunk_size=render_chunk_size,
        )
        # print(f"Render views from decoder output done. Time: {time.time() - start:.2f} seconds.")
        # decoder_outs.append(decoder_out)
        # rendered_rgbs.append(rendered_rgb)
        gt_rgb = gt_images[view:view+1]
        gt_rgb = gt_rgb.to(rendered_rgb.device)
        sky_mask = sky_mask_all[view:view+1].to(rendered_rgb.device)   # [1,H,W]
        valid_mask = ~sky_mask


        if rendered_rgb.ndim == 3:
            rendered_rgb = rendered_rgb.unsqueeze(0)

        # Optional sky MLP: predict sky color from ray direction + global feat
        sky_rgb = None
        if sky_mlp is not None:
            K = intrinsics[view].to(rendered_rgb.device)
            E = extrinsics[view].to(rendered_rgb.device)
            if E.shape == (3, 4):
                bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=E.dtype, device=E.device)
                E = torch.cat([E, bottom], dim=0)
            c2w = torch.linalg.inv(E)
            ray_dirs = compute_ray_dirs_world(H, W, K, c2w)  # [H, W, 3]
            global_feat = dec_in["dino_feat"].mean(0)         # [D]
            sky_rgb_hwc = sky_mlp(ray_dirs, global_feat)      # [H, W, 3]
            sky_rgb = sky_rgb_hwc.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        # start = time.time()
        losses = compute_photometric_loss(
            decoder_out=decoder_out,
            voxel_dict=voxel_dict,
            rendered_rgb=rendered_rgb,
            gt_rgb=gt_rgb,
            valid_mask=valid_mask,
            lpips_fn=lpips_fn,
            sky_rgb=sky_rgb,
            sky_mask_bool=sky_mask,
            lambda_sky=lambda_sky,
        )
        # print(f"Compute photometric loss done. Time: {time.time() - start:.2f} seconds.")

        total_loss = total_loss + losses["total"]
        loss_photo_sum = loss_photo_sum + losses["photo"].item()
        loss_offset_reg_sum = loss_offset_reg_sum + losses["offset_reg"].item()
        loss_scale_reg_sum = loss_scale_reg_sum + losses["scale_reg"].item()
        loss_dssim_sum = loss_dssim_sum + losses["dssim"].item()
        loss_opacity_reg_sum = loss_opacity_reg_sum + losses["opacity_reg"].item()
        loss_color_sum = loss_color_sum + losses["color"].item()
        loss_scale_vol_reg_sum = loss_scale_vol_reg_sum + losses["scale_vol_reg"].item()
        loss_lpips_sum = loss_lpips_sum + losses["lpips"].item()
        loss_aniso_sum = loss_aniso_sum + losses["aniso"].item()
        loss_sky_sum = loss_sky_sum + losses["sky"].item()
        delta_color = decoder_out.get("delta_color", None)
        if delta_color is not None:
            dc = delta_color.detach()
            delta_color_abs_mean_sum += dc.abs().mean().item()

        if i == 0:
            rendered_rgbs_to_log.append(rendered_rgb.detach().cpu())
            gt_rgbs_to_log.append(gt_rgb.detach().cpu())
            sky_masks_to_log = sky_mask.detach().cpu()
            flat_scene_stats = {
                "num_gaussians": int(flat_scene["means3D"].shape[0]),
                "mean_opacity": float(flat_scene["opacity"].mean().item()),
                "mean_scale": float(flat_scene["scales"].mean().item()),
                "mean_abs_center": float(flat_scene["means3D"].abs().mean().item()),
            }
            flat_scene_to_save = {
                "means3D": flat_scene["means3D"].detach().cpu(),
                "scales": flat_scene["scales"].detach().cpu(),
                "rotations": flat_scene["rotations"].detach().cpu(),
                "opacity": flat_scene["opacity"].detach().cpu(),
                "colors": flat_scene["colors"].detach().cpu(),
            }

        del decoder_out, rendered_rgb, rendered_depth, flat_scene, losses

    total_loss = total_loss / num_views
    # start = time.time()
    total_loss.backward()
    optimizer.step()
    # print(f"Backward and optimizer step done. Time: {time.time() - start:.2f} seconds.")

    delta_color_stats = None
    if delta_color_abs_mean_sum > 0:
        delta_color_stats = {
            "abs_mean": delta_color_abs_mean_sum / num_views,
        }


    return {
        "losses": {
            "total": total_loss.detach(),
            "photo": loss_photo_sum / num_views,
            "offset_reg": loss_offset_reg_sum / num_views,
            "scale_reg": loss_scale_reg_sum / num_views,
            "dssim": loss_dssim_sum / num_views,
            "opacity_reg": loss_opacity_reg_sum / num_views,
            "color": loss_color_sum / num_views,
            "scale_vol_reg": loss_scale_vol_reg_sum / num_views,
            "lpips": loss_lpips_sum / num_views,
            "aniso": loss_aniso_sum / num_views,
            "sky": loss_sky_sum / num_views,
        },
        "rendered_rgb": rendered_rgbs_to_log[0],
        "gt_rgb": gt_rgbs_to_log[0],
        "view_indices": view_indices.detach().cpu(),
        "scene_stats": flat_scene_stats,
        "flat_scene_to_save": flat_scene_to_save,
        "sky_mask": sky_masks_to_log[0],
        "delta_color_stats": delta_color_stats,
        "sampling_stats": {
            "num_seen": num_seen_sampled,
            "num_novel": num_novel_sampled,
        },
    }

def train_one_group(
    decoder,
    optimizer,
    group_scenes,
    model,
    voxelizer,
    cache_root,
    views_per_step,
    steps_per_group,
    device,
    sequence_length,
    supervision_mode,
    lpips_fn=None,
    sky_mlp=None,
    lambda_sky: float = 1.0,
):
    scene_names = [s["scene_name"] for s in group_scenes]
    scene_map = {s["scene_name"]: s for s in group_scenes}
    step_logs = []

    for step in range(steps_per_group):
        scene_name = scene_names[step % len(scene_names)]
        scene = scene_map[scene_name]
        all_image_paths = scene["image_paths"]
        num_scene_views = len(all_image_paths)

        if num_scene_views < 3:
            print(f"[WARN] Skip scene {scene_name}: only {num_scene_views} view(s)")
            continue

        if supervision_mode == "even_input_mixed_supervision":
            effective_sequence_length = min(sequence_length, len(scene["image_paths"]))

            split = sample_even_odd_window(
                scene["image_paths"],
                sequence_length=effective_sequence_length,
            )
            try:
                scene_cache = prepare_scene_cache(
                    model=model,
                    voxelizer=voxelizer,
                    input_image_paths=split["input_paths"],
                    supervision_image_paths=[scene["image_paths"][i] for i in split["window_indices"]],
                    output_dir=os.path.join(cache_root, scene_name, f"{split['start']:04d}_{split['end']:04d}"),
                    scene_cache_dir=os.path.join(cache_root, scene_name),
                    full_scene_num_views=len(scene["image_paths"]),
                    input_indices=split["input_indices"],
                    supervision_indices=split["window_indices"],
                    device=device,
                )
            except Exception as e:
                print(f"[WARN] Skip scene {scene_name} at step {step}: {e}")
                continue
        else:
            scene_cache = prepare_scene_cache(
                model=model,
                voxelizer=voxelizer,
                input_image_paths=scene["image_paths"],
                output_dir=os.path.join(cache_root, scene_name),
                scene_cache_dir=os.path.join(cache_root, scene_name),
                full_scene_num_views=len(scene["image_paths"]),
                input_indices=list(range(len(scene["image_paths"]))),
                supervision_indices=list(range(len(scene["image_paths"]))),
                device=device,
            )

        info = train_one_step_on_scene(
            decoder=decoder,
            optimizer=optimizer,
            scene_cache=scene_cache,
            views_per_step=views_per_step,
            device=device,
            lpips_fn=lpips_fn,
            sky_mlp=sky_mlp,
            lambda_sky=lambda_sky,
        )

        step_logs.append({
            "scene_name": scene_name,
            "total": info["losses"]["total"].item(),
            "losses": info["losses"],
            "rendered_rgb": info["rendered_rgb"],
            "gt_rgb": info["gt_rgb"],
            "view_indices": info["view_indices"],
            "scene_stats": info.get("scene_stats", {}),
            "flat_scene_to_save": info.get("flat_scene_to_save", None),
            "voxel_mean_points": scene_cache["voxel_dict"]["voxel_mean_points"].detach().cpu() if scene_cache.get("voxel_dict") else None,
            "sky_mask": info.get("sky_mask", None),
            "delta_color_stats": info.get("delta_color_stats", None),
            "sampling_stats": info.get("sampling_stats", None),
        })
        # break  # for debugging, remove this in actual training

    return step_logs
