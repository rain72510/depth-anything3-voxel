import torch
from torch_scatter import scatter_add, scatter_max


def voxelizaton_with_fusion(img_feat, pts3d, voxel_size, conf=None):
    """
    Aggregate per-pixel 3D points/features into voxels using confidence-weighted average.

    Args:
        img_feat: tensor [B*V, C, H, W] - image features per view
        pts3d: tensor [B*V, 3, H, W] - corresponding 3D points per view
        voxel_size: float - voxel size used to quantize points
        conf: tensor [B*V, H, W] - confidence per pixel (optional)

    Returns:
        voxel_pts: [num_unique_voxels, 3]
        voxel_feats: [num_unique_voxels, feat_dim]
    """
    V, C, H, W = img_feat.shape
    pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)

    voxel_indices = (pts3d_flatten / voxel_size).round().int()
    unique_voxels, inverse_indices, counts = torch.unique(
        voxel_indices, dim=0, return_inverse=True, return_counts=True
    )

    conf_flat = conf.flatten() if conf is not None else torch.ones((pts3d_flatten.shape[0],), device=pts3d.device)
    anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)

    conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
    conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
    voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0)
    weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1)

    weighted_pts = pts3d_flatten * weights
    weighted_feats = anchor_feats_flat.squeeze(1) * weights

    voxel_pts = scatter_add(weighted_pts, inverse_indices, dim=0)
    voxel_feats = scatter_add(weighted_feats, inverse_indices, dim=0)

    return voxel_pts, voxel_feats
