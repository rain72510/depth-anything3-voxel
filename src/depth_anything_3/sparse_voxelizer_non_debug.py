@torch.no_grad()
def voxelize_prediction(self, prediction: Any) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    depth = torch.from_numpy(prediction.depth).to(device)
    intrinsics = torch.from_numpy(prediction.intrinsics).to(device)
    extrinsics = torch.from_numpy(prediction.extrinsics).to(device)
    conf = torch.from_numpy(prediction.conf).to(device) if prediction.conf is not None else None
    images = (
        torch.from_numpy(prediction.processed_images).to(device)
        if prediction.processed_images is not None
        else None
    )

    # 1) validity mask
    mask = torch.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
    if conf is not None:
        conf_thresh = torch.nanquantile(conf, self.conf_percentile / 100.0)
        mask &= torch.isfinite(conf)
        mask &= (conf >= conf_thresh)

    # 2) sparse unprojection
    world_points, view_ids = self._unproject_vectorized(
        depth, intrinsics, extrinsics, mask
    )

    if world_points.shape[0] == 0:
        return {
            "voxel_indices": torch.empty((0, 3), dtype=torch.long, device=device),
            "voxel_mean_points": torch.empty((0, 3), dtype=torch.float32, device=device),
            "voxel_var_points": torch.empty((0, 3), dtype=torch.float32, device=device),
            "voxel_point_counts": torch.empty((0,), dtype=torch.long, device=device),
            "voxel_view_counts": torch.empty((0,), dtype=torch.long, device=device),
            "voxel_colors": None,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_extent": None,
            "num_points": 0,
            "num_voxels": 0,
            "voxel_size": self.voxel_size,
            "summary": {},
        }

    # 3) quantize to voxels
    voxel_coords = torch.floor(world_points / self.voxel_size).long()
    unique_voxels, inverse_indices = torch.unique(
        voxel_coords, dim=0, return_inverse=True
    )

    num_voxels = unique_voxels.shape[0]
    num_points = world_points.shape[0]

    # 4) point count per voxel
    voxel_point_counts = torch.zeros(num_voxels, dtype=torch.long, device=device)
    voxel_point_counts.index_add_(
        0,
        inverse_indices,
        torch.ones(num_points, dtype=torch.long, device=device),
    )

    # 5) mean point per voxel
    voxel_point_sum = torch.zeros((num_voxels, 3), dtype=world_points.dtype, device=device)
    voxel_point_sum.index_add_(0, inverse_indices, world_points)
    voxel_mean_points = voxel_point_sum / voxel_point_counts.unsqueeze(-1).clamp_min(1)

    # 6) point variance per voxel
    voxel_sq_sum = torch.zeros((num_voxels, 3), dtype=world_points.dtype, device=device)
    voxel_sq_sum.index_add_(0, inverse_indices, world_points * world_points)
    voxel_mean_sq = voxel_sq_sum / voxel_point_counts.unsqueeze(-1).clamp_min(1)
    voxel_var_points = torch.clamp(voxel_mean_sq - voxel_mean_points * voxel_mean_points, min=0.0)

    # 7) unique view count per voxel
    voxel_view_pairs = torch.stack([inverse_indices, view_ids], dim=1)
    unique_voxel_view_pairs = torch.unique(voxel_view_pairs, dim=0)

    voxel_view_counts = torch.zeros(num_voxels, dtype=torch.long, device=device)
    voxel_view_counts.index_add_(
        0,
        unique_voxel_view_pairs[:, 0],
        torch.ones(unique_voxel_view_pairs.shape[0], dtype=torch.long, device=device),
    )

    # 8) mean color per voxel
    voxel_colors = None
    if images is not None:
        pixel_colors = images.reshape(-1, 3)[mask.reshape(-1)].float()

        color_sum = torch.zeros((num_voxels, 3), dtype=torch.float32, device=device)
        color_count = torch.zeros((num_voxels, 1), dtype=torch.float32, device=device)

        color_sum.index_add_(0, inverse_indices, pixel_colors)
        color_count.index_add_(
            0,
            inverse_indices,
            torch.ones((pixel_colors.shape[0], 1), dtype=torch.float32, device=device),
        )

        voxel_colors = color_sum / color_count.clamp_min(1e-6)

    # 9) bounding box in world coordinates
    bbox_min = unique_voxels.min(dim=0).values * self.voxel_size
    bbox_max = (unique_voxels.max(dim=0).values + 1) * self.voxel_size
    bbox_extent = bbox_max - bbox_min

    # 10) concise summary
    summary = self._summarize_voxels(
        voxel_point_counts=voxel_point_counts,
        voxel_view_counts=voxel_view_counts,
        num_points=num_points,
        num_voxels=num_voxels,
    )

    return {
        "voxel_indices": unique_voxels,             # (K, 3)
        "voxel_mean_points": voxel_mean_points,     # (K, 3)
        "voxel_var_points": voxel_var_points,       # (K, 3)
        "voxel_point_counts": voxel_point_counts,   # (K,)
        "voxel_view_counts": voxel_view_counts,     # (K,)
        "voxel_colors": voxel_colors,               # (K, 3) or None
        "bbox_min": bbox_min,                       # (3,)
        "bbox_max": bbox_max,                       # (3,)
        "bbox_extent": bbox_extent,                 # (3,)
        "num_points": int(num_points),
        "num_voxels": int(num_voxels),
        "voxel_size": self.voxel_size,
        "summary": summary,
    }