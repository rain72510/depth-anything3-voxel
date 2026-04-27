# Auto-extracted from scripts/train_voxel_decoder.py
# Module: sky_mask

import os
import numpy as np

def build_sky_mask_from_gt(
    images_u8: np.ndarray,   # [V,H,W,3], uint8
    depth: np.ndarray,       # [V,H,W]
    conf: np.ndarray,        # [V,H,W]
):
    V, H, W, _ = images_u8.shape

    # 上方區域 prior
    ys = np.arange(H)[None, :, None]
    top_mask = ys < int(0.45 * H)
    top_mask = np.broadcast_to(top_mask, (V, H, W))

    # 顏色條件：偏亮、低飽和
    img = images_u8.astype(np.float32) / 255.0
    rgb_max = img.max(axis=-1)
    rgb_min = img.min(axis=-1)
    sat = rgb_max - rgb_min
    bright_low_sat = (rgb_max > 0.6) & (sat < 0.18)

    # 幾何條件：遠 depth 或低 conf
    valid_depth = np.isfinite(depth) & (depth > 0)
    if valid_depth.any():
        depth_thr = np.percentile(depth[valid_depth], 90)
    else:
        depth_thr = np.inf
    far_mask = depth >= depth_thr

    valid_conf = np.isfinite(conf)
    if valid_conf.any():
        conf_thr = np.percentile(conf[valid_conf], 25)
    else:
        conf_thr = -np.inf
    low_conf = conf <= conf_thr

    sky_mask = top_mask & bright_low_sat & (far_mask | low_conf)
    return sky_mask.astype(np.bool_)

def load_precomputed_sky_mask(cache_dir: str, images_u8: np.ndarray) -> np.ndarray:
    sky_mask_path = os.path.join(cache_dir, "sky_mask.npz")

    if not os.path.exists(sky_mask_path):
        raise FileNotFoundError(
            f"Precomputed sky mask not found: {sky_mask_path}\n"
            f"Please run precompute_sky_mask.py first."
        )

    data = np.load(sky_mask_path)
    sky_mask = data["sky_mask"].astype(np.bool_)

    expected_shape = images_u8.shape[:3]   # [V,H,W]
    if tuple(sky_mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"Sky mask shape mismatch: got {sky_mask.shape}, expected {expected_shape}"
        )

    return sky_mask

def load_precomputed_sky_mask_subset(
    scene_cache_dir: str,
    full_num_views: int,
    selected_indices: list[int],
):
    sky_mask_path = os.path.join(scene_cache_dir, "sky_mask.npz")

    if not os.path.exists(sky_mask_path):
        raise FileNotFoundError(
            f"Precomputed sky mask not found: {sky_mask_path}\n"
            f"Please run precompute_sky_mask.py first."
        )

    data = np.load(sky_mask_path)
    sky_mask = data["sky_mask"].astype(np.bool_)   # [V,H,W]

    cached_num_views = sky_mask.shape[0]

    if cached_num_views != full_num_views:
        print(
            f"[WARN] sky mask num_views mismatch in {scene_cache_dir}: "
            f"cache={cached_num_views}, current_scene={full_num_views}"
        )

    if len(selected_indices) == 0:
        raise ValueError("selected_indices is empty.")

    max_idx = max(selected_indices)
    min_idx = min(selected_indices)

    if min_idx < 0 or max_idx >= cached_num_views:
        raise ValueError(
            f"Sky mask index out of range: min={min_idx}, max={max_idx}, "
            f"cached_num_views={cached_num_views}"
        )

    return sky_mask[selected_indices]
