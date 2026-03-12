import torch
import time
from typing import Optional, Tuple, Dict, Any

class SparseVoxelizer:
    def __init__(
        self,
        max_depth: float = 50.0,
        voxel_size: float = 0.4, # 建議自駕場景從 0.4m 開始
        conf_percentile: float = 30.0,
        truncation_band: float = 0.5  
    ):
        self.max_depth = max_depth
        self.voxel_size = voxel_size
        self.conf_percentile = conf_percentile
        self.truncation_band = truncation_band 

    @torch.no_grad()
    def voxelize_prediction(self, prediction: Any) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 轉換 Tensor 並移至 GPU
        depth = torch.from_numpy(prediction.depth).to(device)
        intrinsics = torch.from_numpy(prediction.intrinsics).to(device)
        extrinsics = torch.from_numpy(prediction.extrinsics).to(device)
        images = torch.from_numpy(prediction.processed_images).to(device) if prediction.processed_images is not None else None
        conf = torch.from_numpy(prediction.conf).to(device) if prediction.conf is not None else None

        print(f"Extrinsics[0]: {extrinsics[0]}")
        print(f"Intrinsics[0]: {intrinsics[0]}")

        # 2. 計算 Mask
        mask = torch.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        if conf is not None:
            conf_thresh = torch.nanquantile(conf, self.conf_percentile / 100.0)
            mask &= torch.isfinite(conf)
            mask &= (conf >= conf_thresh)

        # 3. 反投影
        world_points, view_ids = self._unproject_vectorized(depth, intrinsics, extrinsics, mask)

        if world_points.shape[0] == 0:
            return {"num_voxels": 0}

        # 4. Voxelization
        voxel_coords = torch.floor(world_points / self.voxel_size).long()
        unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)

        num_unique = unique_voxels.shape[0]
        num_points = world_points.shape[0]

        # ---------------------------------------------------
        # A. 每個 voxel 的 point count
        # ---------------------------------------------------
        voxel_point_counts = torch.zeros(num_unique, device=device, dtype=torch.long)
        voxel_point_counts.index_add_(
            0,
            inverse_indices,
            torch.ones(num_points, device=device, dtype=torch.long)
        )

        # ---------------------------------------------------
        # B. 每個 voxel 的 mean point
        # ---------------------------------------------------
        voxel_point_sum = torch.zeros((num_unique, 3), device=device, dtype=world_points.dtype)
        voxel_point_sum.index_add_(0, inverse_indices, world_points)
        voxel_mean_points = voxel_point_sum / voxel_point_counts.unsqueeze(-1).clamp_min(1)

        # ---------------------------------------------------
        # C. 每個 voxel 的 point variance
        # ---------------------------------------------------
        sq_points = world_points * world_points
        voxel_sq_sum = torch.zeros((num_unique, 3), device=device, dtype=world_points.dtype)
        voxel_sq_sum.index_add_(0, inverse_indices, sq_points)
        voxel_mean_sq = voxel_sq_sum / voxel_point_counts.unsqueeze(-1).clamp_min(1)
        voxel_var_points = voxel_mean_sq - voxel_mean_points * voxel_mean_points
        voxel_var_points = torch.clamp(voxel_var_points, min=0.0)

        # ---------------------------------------------------
        # D. 顏色聚合
        # ---------------------------------------------------
        voxel_colors = None
        if images is not None:
            pixel_colors = images.reshape(-1, 3)[mask.reshape(-1)].float()

            color_sum = torch.zeros((num_unique, 3), device=device, dtype=torch.float32)
            color_count = torch.zeros((num_unique, 1), device=device, dtype=torch.float32)

            color_sum.index_add_(0, inverse_indices, pixel_colors)
            color_count.index_add_(
                0,
                inverse_indices,
                torch.ones((pixel_colors.shape[0], 1), device=device, dtype=torch.float32)
            )

            voxel_colors = color_sum / color_count.clamp_min(1e-6)

        # ---------------------------------------------------
        # E. 每個 voxel 的 unique view count
        # ---------------------------------------------------
        # 做法：先把 (voxel_id, view_id) 配對 unique，再對 voxel_id 計數
        voxel_view_pairs = torch.stack([inverse_indices, view_ids], dim=1)  # (M, 2)
        unique_voxel_view_pairs = torch.unique(voxel_view_pairs, dim=0)
        voxel_view_counts = torch.zeros(num_unique, device=device, dtype=torch.long)
        voxel_view_counts.index_add_(
            0,
            unique_voxel_view_pairs[:, 0],
            torch.ones(unique_voxel_view_pairs.shape[0], device=device, dtype=torch.long)
        )

        # ---------------------------------------------------
        # F. bbox
        # ---------------------------------------------------
        bbox_min = unique_voxels.min(dim=0)[0] * self.voxel_size
        bbox_max = (unique_voxels.max(dim=0)[0] + 1) * self.voxel_size
        bbox_extent = bbox_max - bbox_min

        # ---------------------------------------------------
        # G. summary stats
        # ---------------------------------------------------
        stats = {
            "num_points": int(num_points),
            "num_voxels": int(num_unique),

            "avg_points_per_voxel": float(voxel_point_counts.float().mean().item()),
            "median_points_per_voxel": float(voxel_point_counts.float().median().item()),
            "max_points_per_voxel": int(voxel_point_counts.max().item()),
            "min_points_per_voxel": int(voxel_point_counts.min().item()),

            "avg_views_per_voxel": float(voxel_view_counts.float().mean().item()),
            "median_views_per_voxel": float(voxel_view_counts.float().median().item()),
            "max_views_per_voxel": int(voxel_view_counts.max().item()),
            "min_views_per_voxel": int(voxel_view_counts.min().item()),

            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "bbox_extent": bbox_extent.tolist(),
        }

        print("Voxelization complete.")
        for k, v in stats.items():
            print(f"{k}: {v}")

        return {
            "voxel_indices": unique_voxels,             # (K, 3)
            "voxel_colors": voxel_colors,               # (K, 3) or None
            "voxel_point_counts": voxel_point_counts,   # (K,)
            "voxel_mean_points": voxel_mean_points,     # (K, 3)
            "voxel_var_points": voxel_var_points,       # (K, 3)
            "voxel_view_counts": voxel_view_counts,     # (K,)
            "num_voxels": int(num_unique),
            "num_points": int(num_points),
            "voxel_size": self.voxel_size,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_extent": bbox_extent,
            "stats": stats,
        }

    def _unproject_vectorized(self, depth, K, E, mask):
        N, H, W = depth.shape
        device = depth.device

        v, u = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        valid_coords = torch.nonzero(mask, as_tuple=False)   # (M, 3) = [view, v, u]
        if valid_coords.numel() == 0:
            return torch.empty((0, 3), device=device, dtype=depth.dtype)

        view_ids = valid_coords[:, 0]
        v_m = valid_coords[:, 1].float()
        u_m = valid_coords[:, 2].float()
        z = depth[view_ids, valid_coords[:, 1], valid_coords[:, 2]]  # (M,)

        # homogeneous image coords
        homo_coords = torch.stack([u_m, v_m, torch.ones_like(u_m)], dim=-1)  # (M, 3)

        # per-view inverse intrinsics
        inv_K = torch.inverse(K)                    # (N, 3, 3)
        inv_K_sel = inv_K[view_ids]                 # (M, 3, 3)

        # camera-space points
        points_cam = torch.bmm(inv_K_sel, homo_coords.unsqueeze(-1)).squeeze(-1)  # (M, 3)
        points_cam = points_cam * z.unsqueeze(-1)

        # convert E -> c2w
        E_h = self._as_homogeneous_batch(E)         # (N, 4, 4)
        c2w = torch.inverse(E_h)                    # (N, 4, 4)
        c2w_sel = c2w[view_ids]                     # (M, 4, 4)

        R = c2w_sel[:, :3, :3]                      # (M, 3, 3)
        t = c2w_sel[:, :3, 3]                       # (M, 3)

        points_world = torch.bmm(R, points_cam.unsqueeze(-1)).squeeze(-1) + t
        return points_world, view_ids
    
    def _as_homogeneous_batch(self, E):
        # E: (N, 3, 4)
        N = E.shape[0]
        bottom = torch.tensor([0, 0, 0, 1], device=E.device, dtype=E.dtype).view(1, 1, 4).expand(N, 1, 4)
        return torch.cat([E, bottom], dim=1)   # (N, 4, 4)