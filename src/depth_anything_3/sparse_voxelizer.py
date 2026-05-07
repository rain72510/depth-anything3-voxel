import torch
import time
from typing import Optional, Tuple, Dict, Any

class SparseVoxelizer:
    def __init__(
        self,
        max_depth: float = 50.0,
        voxel_size: float = 0.4, # 建議自駕場景從 0.4m 開始
        conf_percentile: float = 30.0,
        truncation_band: float = 0.5,
        feat_mode: str = "last2_avg",      # "last", "last2_avg", "all4_avg"
        patch_size: int = 14,
        feat_dim_out: Optional[int] = None, # e.g. 256; None means keep original dim
        neighbor_patch_radius: int = 0,     # 0=center only, 1=3x3, 2=5x5 patch neighborhood
        perview_conf: bool = False,         # if True, compute conf threshold per view instead of globally
        voxel_size_dist_ref: float = 0.0,   # >0 enables distance-adaptive sizing: vsize = voxel_size * max(1, d/ref)^exp
        voxel_size_exp: float = 1.0,        # 1.0 = linear-with-distance, 2.0 ≈ inverse-depth-uniform sampling
        keep_pixel_features: bool = False,  # v2: also aggregate full-dim DINO per voxel for appearance head
        keep_image_features: bool = False,  # v3: keep per-view image-space DINO + camera params for per-Gaussian projection
    ):
        self.max_depth = max_depth
        self.voxel_size = voxel_size
        # self.voxel_size = 0.1
        self.conf_percentile = conf_percentile
        self.truncation_band = truncation_band
        self.feat_mode = feat_mode
        self.patch_size = patch_size
        self.feat_dim_out = feat_dim_out
        self.neighbor_patch_radius = neighbor_patch_radius
        self.perview_conf = perview_conf
        self.voxel_size_dist_ref = voxel_size_dist_ref
        self.voxel_size_exp = voxel_size_exp
        self.keep_pixel_features = keep_pixel_features
        self.keep_image_features = keep_image_features
        self.pixel_feature_dtype = torch.float16

    @torch.no_grad()
    def voxelize_prediction(self, prediction: Any) -> Dict[str, Any]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 轉換 Tensor 並移至 GPU
        depth = torch.from_numpy(prediction.depth).to(device)
        intrinsics = torch.from_numpy(prediction.intrinsics).to(device)
        extrinsics = torch.from_numpy(prediction.extrinsics).to(device)
        images = torch.from_numpy(prediction.processed_images).to(device) if prediction.processed_images is not None else None
        conf = torch.from_numpy(prediction.conf).to(device) if prediction.conf is not None else None

        # print(f"Extrinsics[0]: {extrinsics[0]}")
        # print(f"Intrinsics[0]: {intrinsics[0]}")

        raw_feats = getattr(prediction, "raw_feats", None)

        # 2. 計算 Mask
        mask = torch.isfinite(depth) & (depth > 0) & (depth < self.max_depth)
        if conf is not None:
            mask &= torch.isfinite(conf)
            if self.perview_conf:
                # per-view threshold: each camera gets its own percentile cutoff
                for v in range(conf.shape[0]):
                    view_conf = conf[v]
                    valid = torch.isfinite(view_conf)
                    if valid.any():
                        thresh = torch.nanquantile(view_conf[valid], self.conf_percentile / 100.0)
                        mask[v] &= (view_conf >= thresh)
            else:
                # global threshold across all views
                conf_thresh = torch.nanquantile(conf, self.conf_percentile / 100.0)
                mask &= (conf >= conf_thresh)

        # 3. 反投影
        world_points, view_ids, ys, xs = self._unproject_vectorized(depth, intrinsics, extrinsics, mask)

        if world_points.shape[0] == 0:
            return {"num_voxels": 0}

        # 4. Voxelization
        # Distance-adaptive voxel size: when voxel_size_dist_ref > 0, points farther
        # from the source camera get coarser voxels (saves anchors in regions where
        # image-space pixel area covers more world-space anyway). vsize stays at
        # self.voxel_size for d <= ref and grows as (d/ref)^exp beyond.
        if self.voxel_size_dist_ref > 0.0:
            E_h = self._as_homogeneous_batch(extrinsics)
            c2w = torch.inverse(E_h)
            cam_xyz = c2w[:, :3, 3]                                    # [V, 3]
            point_cam = cam_xyz[view_ids]                              # [N, 3]
            point_dist = (world_points - point_cam).norm(dim=-1)       # [N]
            ratio = (point_dist / self.voxel_size_dist_ref).clamp_min(1.0)
            # Quantize ratio into log2 depth bins so all points within a bin share
            # the same vsize, and the bin id prefix prevents cross-bin coord
            # collisions (a real bug — without this, with exp=2 a 5 m point and a
            # 20 m point can map to the same voxel coord and torch.unique merges
            # them, producing a degenerate scaffold and all-black renders).
            bin_id = ratio.log2().floor().clamp_min(0).long()                  # [N]
            bin_ratio = (2.0 ** bin_id.float())                                # [N]
            vsize_per_point = self.voxel_size * (bin_ratio ** self.voxel_size_exp)  # [N]
            voxel_coords_3d = torch.floor(world_points / vsize_per_point.unsqueeze(-1)).long()
            # 4-D coords: [bin_id, vx, vy, vz]; downstream code that touches
            # `unique_voxels` either ignores extra dims or operates in adaptive
            # mode (bbox computed from world_points instead).
            voxel_coords = torch.cat([bin_id.unsqueeze(-1), voxel_coords_3d], dim=-1)
        else:
            voxel_coords = torch.floor(world_points / self.voxel_size).long()
        unique_voxels, inverse_indices = torch.unique(voxel_coords, dim=0, return_inverse=True)

        num_unique = unique_voxels.shape[0]
        num_points = world_points.shape[0]

        voxel_features = None
        voxel_feature_dim = None

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

        point_conf = conf[view_ids, ys, xs].float()   # [num_points]

        voxel_conf_sum = torch.zeros(num_unique, device=device, dtype=torch.float32)
        voxel_conf_sum.index_add_(0, inverse_indices, point_conf)

        voxel_confidence = voxel_conf_sum / voxel_point_counts.clamp_min(1).float()

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

        voxel_pixel_features = None
        image_dino_feats = None  # v3: per-view image-space DINO patch features
        if raw_feats is not None:
            voxel_features = self._aggregate_voxel_features_from_tokens_chunked(
                raw_feats=raw_feats,
                view_ids=view_ids,
                ys=ys,
                xs=xs,
                inverse_indices=inverse_indices,
                voxel_point_counts=voxel_point_counts,
                image_hw=depth.shape[-2:],
                device=device,
                chunk_size=200000,   # 可調
            )
            voxel_feature_dim = voxel_features.shape[1]

            if self.keep_pixel_features:
                voxel_pixel_features = self._aggregate_full_dim_pixel_features(
                    raw_feats=raw_feats,
                    view_ids=view_ids,
                    ys=ys,
                    xs=xs,
                    inverse_indices=inverse_indices,
                    voxel_point_counts=voxel_point_counts,
                    image_hw=depth.shape[-2:],
                    device=device,
                    chunk_size=200000,
                )

            if self.keep_image_features:
                # Same layer-combination as the geometry path; full-dim, fp16,
                # stored as [V, Hf, Wf, C_full] for v3 per-Gaussian projection.
                image_dino_feats = self._extract_image_dino_feats(
                    raw_feats=raw_feats,
                    image_hw=depth.shape[-2:],
                    device=device,
                )

        # ---------------------------------------------------
        # F. bbox
        # ---------------------------------------------------
        if self.voxel_size_dist_ref > 0.0:
            # Adaptive sizing: voxel coords are in heterogeneous grids; derive bbox
            # from the actual world points instead of from voxel coords * size.
            bbox_min = world_points.min(dim=0)[0]
            bbox_max = world_points.max(dim=0)[0]
        else:
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
            "voxel_feature_dim": voxel_feature_dim,
        }

        print("Voxelization complete.")
        # for k, v in stats.items():
        #     print(f"{k}: {v}")

        # # print voxel_features min, max, mean
        # if voxel_features is not None:
        #     print(f"voxel_features: min={voxel_features.min().item()}, max={voxel_features.max().item()}, mean={voxel_features.mean().item()}")


        return {
            "voxel_indices": unique_voxels,             # (K, 3)
            "voxel_colors": voxel_colors,               # (K, 3) or None
            "voxel_point_counts": voxel_point_counts,   # (K,)
            "voxel_mean_points": voxel_mean_points,     # (K, 3)
            "voxel_var_points": voxel_var_points,       # (K, 3)
            "voxel_view_counts": voxel_view_counts,     # (K,)
            "voxel_features": voxel_features,           # (K, C) or None
            "voxel_pixel_features": voxel_pixel_features,  # (K, C_full) fp16 or None — v2 appearance head input
            "image_dino_feats": image_dino_feats,        # [V, Hf, Wf, C_full] fp16 or None — v3 per-Gaussian projection input
            "intrinsics_v": intrinsics if self.keep_image_features else None,    # [V, 3, 3]
            "extrinsics_v": extrinsics if self.keep_image_features else None,    # [V, 3, 4] or [V, 4, 4]
            "image_hw": tuple(depth.shape[-2:]) if self.keep_image_features else None,
            "raw_images": images if self.keep_image_features else None,          # [V, H, W, 3] uint8
            "raw_conf": conf if self.keep_image_features else None,              # [V, H, W]
            "voxel_confidence": voxel_confidence,
            "num_voxels": int(num_unique),
            "num_points": int(num_points),
            "voxel_size": self.voxel_size,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_extent": bbox_extent,
            "stats": stats,
        }
    
    def _aggregate_voxel_features_from_tokens_chunked(
        self,
        raw_feats,
        view_ids: torch.Tensor,
        ys: torch.Tensor,
        xs: torch.Tensor,
        inverse_indices: torch.Tensor,
        voxel_point_counts: torch.Tensor,
        image_hw: Tuple[int, int],
        device: torch.device,
        chunk_size: int = 200000,
    ) -> torch.Tensor:
        """
        直接 chunk gather point features，並累加到 voxel_feature_sum。
        不建立完整 [num_points, C]，避免 OOM。
        """

        # 1. choose spatial token features
        if self.feat_mode == "last":
            feat_tokens = raw_feats[3][0]
        elif self.feat_mode == "last2_avg":
            feat_tokens = 0.5 * (raw_feats[2][0] + raw_feats[3][0])
        elif self.feat_mode == "all4_avg":
            feat_tokens = sum(raw_feats[i][0] for i in range(4)) / 4.0
        else:
            raise ValueError(f"Unknown feat_mode: {self.feat_mode}")

        # [1, V, Ntok, C] -> [V, Ntok, C]
        feat_tokens = feat_tokens[0].to(device)

        V, Ntok, C = feat_tokens.shape
        H, W = image_hw
        patch = self.patch_size
        Hf, Wf = H // patch, W // patch

        assert Ntok == Hf * Wf, f"Ntok={Ntok}, expected {Hf * Wf} from image_hw={image_hw}, patch={patch}"

        # print(f"Before cut from 3072 to feat_dim_out={self.feat_dim_out}, feat_tokens shape: {feat_tokens.shape}")

        # 2. optional dim truncation (先切，再 gather)
        if self.feat_dim_out is not None and self.feat_dim_out < C:
            feat_tokens = feat_tokens[..., :self.feat_dim_out]

        # 3. half precision 節省記憶體
        feat_tokens = feat_tokens.to(torch.float16)

        # print feat_tokens min, max, mean before aggregation
        # print(f"[before aggregation] feat_tokens: min={feat_tokens.min().item()}, max={feat_tokens.max().item()}, mean={feat_tokens.mean().item()}")

        C_small = feat_tokens.shape[-1]
        num_voxels = voxel_point_counts.shape[0]

        # voxel_feature_sum = torch.zeros((num_voxels, C_small), device=device, dtype=feat_tokens.dtype)
        voxel_feature_sum = torch.zeros((num_voxels, C_small), device=device, dtype=torch.float32)  # 用 float32 累加，減少精度損失

        num_points = view_ids.shape[0]

        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)

            view_ids_chunk = view_ids[start:end]
            ys_chunk = ys[start:end]
            xs_chunk = xs[start:end]
            inv_chunk = inverse_indices[start:end]

            patch_y = ys_chunk // patch
            patch_x = xs_chunk // patch

            # [chunk, C_small] — gather from center patch or NxN neighborhood
            if self.neighbor_patch_radius == 0:
                token_idx = patch_y * Wf + patch_x
                point_feat_chunk = feat_tokens[view_ids_chunk, token_idx].to(torch.float32)
            else:
                r = self.neighbor_patch_radius
                n_neighbors = (2 * r + 1) ** 2
                point_feat_chunk = torch.zeros(len(view_ids_chunk), C_small, device=device, dtype=torch.float32)
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny = (patch_y + dy).clamp(0, Hf - 1)
                        nx = (patch_x + dx).clamp(0, Wf - 1)
                        tidx = ny * Wf + nx
                        point_feat_chunk += feat_tokens[view_ids_chunk, tidx].to(torch.float32)
                point_feat_chunk /= n_neighbors

            voxel_feature_sum.index_add_(0, inv_chunk, point_feat_chunk)

            # 可選，幫助釋放暫時 tensor
            del view_ids_chunk, ys_chunk, xs_chunk, inv_chunk, point_feat_chunk

        # print voxel_feature_sum min, max, mean
        # in func _aggregate_voxel_features_from_tokens_chunked, after the for loop
        # print(f"[in func _aggregate_voxel_features_from_tokens_chunked] voxel_feature_sum: min={voxel_feature_sum.min().item()}, max={voxel_feature_sum.max().item()}, mean={voxel_feature_sum.mean().item()}")

        # print voxel_point_counts min, max, mean
        # print(f"voxel_point_counts: min={voxel_point_counts.min().item()}, max={voxel_point_counts.max().item()}, mean={voxel_point_counts.float().mean().item()}")

        voxel_counts = voxel_point_counts.unsqueeze(-1).clamp_min(1).to(torch.float32)
        voxel_features = voxel_feature_sum / voxel_counts
        return voxel_features

    def _extract_image_dino_feats(self, raw_feats, image_hw, device):
        """v3 path: keep per-view image-space DINO patch features at full dim, fp16.
        Returns [V, Hf, Wf, C_full]. Same layer combination as the geometry path.
        Memory: V × Hf × Wf × C_full × 2 bytes (~70 MB at V=12, Hf×Wf=925, C=3072)."""
        if self.feat_mode == "last":
            feat_tokens = raw_feats[3][0]
        elif self.feat_mode == "last2_avg":
            feat_tokens = 0.5 * (raw_feats[2][0] + raw_feats[3][0])
        elif self.feat_mode == "all4_avg":
            feat_tokens = sum(raw_feats[i][0] for i in range(4)) / 4.0
        else:
            raise ValueError(f"Unknown feat_mode: {self.feat_mode}")

        feat_tokens = feat_tokens[0].to(device)  # [V, Ntok, C_full]
        V, Ntok, C_full = feat_tokens.shape
        H, W = image_hw
        patch = self.patch_size
        Hf, Wf = H // patch, W // patch
        assert Ntok == Hf * Wf, f"Ntok={Ntok}, expected {Hf * Wf}"
        return feat_tokens.reshape(V, Hf, Wf, C_full).to(self.pixel_feature_dtype)

    def _aggregate_full_dim_pixel_features(
        self,
        raw_feats,
        view_ids: torch.Tensor,
        ys: torch.Tensor,
        xs: torch.Tensor,
        inverse_indices: torch.Tensor,
        voxel_point_counts: torch.Tensor,
        image_hw: Tuple[int, int],
        device: torch.device,
        chunk_size: int = 200000,
    ) -> torch.Tensor:
        """v2 appearance path: aggregate FULL-dim DINO patch features per voxel,
        store as fp16 to halve memory. Mirrors the truncated aggregator but skips
        feat_dim_out and uses center patch only (neighbor_patch_radius is geometry-only)."""
        if self.feat_mode == "last":
            feat_tokens = raw_feats[3][0]
        elif self.feat_mode == "last2_avg":
            feat_tokens = 0.5 * (raw_feats[2][0] + raw_feats[3][0])
        elif self.feat_mode == "all4_avg":
            feat_tokens = sum(raw_feats[i][0] for i in range(4)) / 4.0
        else:
            raise ValueError(f"Unknown feat_mode: {self.feat_mode}")

        feat_tokens = feat_tokens[0].to(device)  # [V, Ntok, C_full]
        V, Ntok, C_full = feat_tokens.shape
        H, W = image_hw
        patch = self.patch_size
        Hf, Wf = H // patch, W // patch
        assert Ntok == Hf * Wf, f"Ntok={Ntok}, expected {Hf * Wf}"

        feat_tokens = feat_tokens.to(torch.float16)

        num_voxels = voxel_point_counts.shape[0]
        # accumulate in fp32 to limit precision loss, downcast at the end
        voxel_feature_sum = torch.zeros((num_voxels, C_full), device=device, dtype=torch.float32)

        num_points = view_ids.shape[0]
        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            view_ids_chunk = view_ids[start:end]
            patch_y = ys[start:end] // patch
            patch_x = xs[start:end] // patch
            inv_chunk = inverse_indices[start:end]

            token_idx = patch_y * Wf + patch_x
            point_feat_chunk = feat_tokens[view_ids_chunk, token_idx].to(torch.float32)
            voxel_feature_sum.index_add_(0, inv_chunk, point_feat_chunk)
            del view_ids_chunk, patch_y, patch_x, inv_chunk, point_feat_chunk

        voxel_counts = voxel_point_counts.unsqueeze(-1).clamp_min(1).to(torch.float32)
        voxel_pixel_features = (voxel_feature_sum / voxel_counts).to(self.pixel_feature_dtype)
        return voxel_pixel_features

    def _unproject_vectorized(self, depth, K, E, mask):
        N, H, W = depth.shape
        device = depth.device

        valid_coords = torch.nonzero(mask, as_tuple=False)   # (M, 3) = [view, y, x]
        if valid_coords.numel() == 0:
            empty_xyz = torch.empty((0, 3), device=device, dtype=depth.dtype)
            empty_idx = torch.empty((0,), device=device, dtype=torch.long)
            return empty_xyz, empty_idx, empty_idx, empty_idx

        view_ids = valid_coords[:, 0]
        ys = valid_coords[:, 1]
        xs = valid_coords[:, 2]

        y_f = ys.float()
        x_f = xs.float()
        z = depth[view_ids, ys, xs]

        homo_coords = torch.stack([x_f, y_f, torch.ones_like(x_f)], dim=-1)

        inv_K = torch.inverse(K)
        inv_K_sel = inv_K[view_ids]

        points_cam = torch.bmm(inv_K_sel, homo_coords.unsqueeze(-1)).squeeze(-1)
        points_cam = points_cam * z.unsqueeze(-1)

        E_h = self._as_homogeneous_batch(E)
        c2w = torch.inverse(E_h)
        c2w_sel = c2w[view_ids]

        R = c2w_sel[:, :3, :3]
        t = c2w_sel[:, :3, 3]

        points_world = torch.bmm(R, points_cam.unsqueeze(-1)).squeeze(-1) + t
        return points_world, view_ids, ys, xs
    
    def _as_homogeneous_batch(self, E):
        # E: (N, 3, 4)
        N = E.shape[0]
        bottom = torch.tensor([0, 0, 0, 1], device=E.device, dtype=E.dtype).view(1, 1, 4).expand(N, 1, 4)
        return torch.cat([E, bottom], dim=1)   # (N, 4, 4)
    