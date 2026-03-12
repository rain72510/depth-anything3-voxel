# src/depth_anything_3/utils/export/voxel_glb.py  
import os  
import numpy as np  
import trimesh  
from typing import Optional, Dict, Any  
  
from ...specs import Prediction  
from ...voxelizer import BoundedVoxelizer  
from ...sparse_voxelizer import SparseVoxelizer
  
  
def export_voxel_centers_glb(  
    prediction: Prediction,  
    export_dir: str,  
    voxelizer_cfg: Optional[Dict[str, Any]] = None,  
    filename: str = "voxels.glb",  
    colors: Optional[np.ndarray] = None,  
) -> str:  
    """  
    Export voxel centers as GLB point cloud.  
      
    Args:  
        prediction: DA3 Prediction object  
        export_dir: Output directory  
        voxelizer_cfg: Config for BoundedVoxelizer  
        filename: Output filename  
        colors: Optional per-point colors (K, 3) uint8 RGB  
          
    Returns:  
        Path to exported GLB file  
    """  
    # Initialize voxelizer with provided config  
    voxelizer_cfg = voxelizer_cfg or {}  
    # voxelizer = BoundedVoxelizer(**voxelizer_cfg)  
    voxelizer = SparseVoxelizer(**voxelizer_cfg)  
      
    # Voxelize prediction  
    voxel_result = voxelizer.voxelize_prediction(prediction)  
      
    # Extract voxel data  
    voxel_indices = voxel_result['voxel_indices']  
    bbox_min = voxel_result['bbox_min']  
    voxel_size = voxel_result['voxel_size']  
    voxel_colors = voxel_result.get('voxel_colors')  # Get computed colors  
      
    if voxel_indices is None or len(voxel_indices) == 0:  
        # Create empty point cloud if no voxels  
        voxel_centers = np.empty((0, 3), dtype=np.float32)  
        final_colors = np.empty((0, 3), dtype=np.uint8)  
    else:  
        # Compute voxel centers: bbox_min + (indices + 0.5) * voxel_size  
        # voxel_centers = bbox_min + (voxel_indices.astype(np.float32) + 0.5) * voxel_size  
        # 'Tensor' object has no attribute 'astype'else voxel_centers.astype(np.float32)
        voxel_centers = bbox_min + (voxel_indices.float() + 0.5) * voxel_size
        voxel_centers[:, 1] = -voxel_centers[:, 1]
        voxel_centers[:, 0] = -voxel_centers[:, 0]
        # Use computed colors or provided colors  
        if voxel_colors is not None:  
            # Convert float colors [0,255] to uint8  
            # final_colors = voxel_colors.astype(np.uint8)  
            final_colors = voxel_colors.detach().cpu().numpy().astype(np.uint8)
        elif colors is not None:  
            if len(colors) != len(voxel_centers):  
                raise ValueError(f"Number of colors ({len(colors)}) doesn't match number of voxels ({len(voxel_centers)})")  
            final_colors = colors  
        else:  
            # Default red color  
            final_colors = np.array([[255, 0, 0]], dtype=np.uint8)  
            final_colors = np.repeat(final_colors, len(voxel_centers), axis=0)  
      
    # Create point cloud mesh  
    # point_cloud = trimesh.PointCloud(vertices=voxel_centers)  
    point_cloud = trimesh.PointCloud(
        vertices=voxel_centers.detach().cpu().numpy()
    )
      
    # Set colors with alpha channel  
    colors_with_alpha = np.concatenate([final_colors, np.full((len(final_colors), 1), 255, dtype=np.uint8)], axis=1)  
    point_cloud.visual.vertex_colors = colors_with_alpha  
      
    # Export as GLB  
    output_path = os.path.join(export_dir, filename)  
    point_cloud.export(output_path)  
      
    return output_path