# src/depth_anything_3/voxelizer.py  
import torch  
import numpy as np  
from typing import Optional, Tuple, Dict, Any  
from .utils.geometry import unproject_depth  
from .specs import Prediction  
import time
  
class BoundedVoxelizer:  
    """  
    Voxelizer that creates bounded 3D voxel grids from DA3 predictions.  
    Separates bounded/unbounded geometry using depth and confidence.  
    """  
      
    def __init__(  
        self,  
        max_depth: float = 50.0,  
        voxel_size: float = 0.1,  
        conf_percentile: float = 40.0,  
        truncation_band: float = 0.5  
    ):  
        self.max_depth = max_depth  
        self.voxel_size = voxel_size  
        self.conf_percentile = conf_percentile  
        self.truncation_band = truncation_band  
      
    def voxelize_prediction(self, prediction: Prediction) -> Dict[str, Any]:  
        """  
        Convert DA3 prediction to bounded voxel grid.  
          
        Args:  
            prediction: DA3 Prediction object with depth, conf, extrinsics, intrinsics  
              
        Returns:  
            Dictionary with voxelization results  
        """  
        # Convert to tensors  
        depth = torch.from_numpy(prediction.depth)  # (N, H, W)  
        conf = torch.from_numpy(prediction.conf) if prediction.conf is not None else None  
        extrinsics = torch.from_numpy(prediction.extrinsics)  # (N, 3, 4)  
        intrinsics = torch.from_numpy(prediction.intrinsics)  # (N, 3, 3)  

        if hasattr(prediction, 'processed_images') and prediction.processed_images is not None:  
            images = torch.from_numpy(prediction.processed_images)  # (N, H, W, 3)  
        else:  
            images = None 
          
        N, H, W = depth.shape  
          
        # Compute bounded mask  
        t=time.time()
        bounded_mask = self._compute_bounded_mask(depth, conf)  
        print("_compute_bounded_mask", time.time()-t)
          
        # Unproject to world space  
        t=time.time()
        world_points = self._unproject_to_world(depth, intrinsics, extrinsics) 
        print("_unproject_to_world", time.time()-t) 
          
        # Compute tight bounding box from bounded points  
        t=time.time()
        bbox_min, bbox_max = self._compute_bounded_bbox(world_points, bounded_mask) 
        print("_compute_bounded_bbox", time.time()-t)  
          
        # Create voxel grid  
        # voxel_grid, voxel_indices = self._create_voxel_grid(  
        #     world_points, bounded_mask, bbox_min, bbox_max  
        # )  
        t=time.time()
        voxel_grid, voxel_indices, voxel_colors = self._create_voxel_grid_with_colors(  
            world_points, bounded_mask, bbox_min, bbox_max, images  
        )  
        print("_create_voxel_grid_with_colors", time.time()-t)  
          
        # Apply depth-aware truncation  
        # t=time.time()
        # voxel_grid = self._apply_truncation(  
        #     voxel_grid, world_points, depth, conf, voxel_indices, bounded_mask  
        # )  
        # print("mask", time.time()-t)  
        
        # print the return values
        print("Voxelization Results:")
        print(f"Voxel Grid Shape: {voxel_grid.shape}")
        print(f"BBox Min: {bbox_min}")
        print(f"BBox Max: {bbox_max}")
        print(f"Voxel Size: {self.voxel_size}")
        print(f"Bounded Mask Shape: {bounded_mask.shape}")
        if voxel_indices is not None:
            print(f"Voxel Indices Shape: {voxel_indices.shape}")
        else:
            print("Voxel Indices: None")

        return {  
            'voxel_grid': voxel_grid.cpu().numpy(),  
            'bbox_min': bbox_min.cpu().numpy(),  
            'bbox_max': bbox_max.cpu().numpy(),  
            'voxel_size': self.voxel_size,  
            'bounded_mask': bounded_mask.cpu().numpy(),  
            'num_voxels': voxel_grid.shape,  
            'voxel_indices': voxel_indices.cpu().numpy() if voxel_indices is not None else None,  
            'voxel_colors': voxel_colors.cpu().numpy() if voxel_colors is not None else None  
        }  
      
    def _compute_bounded_mask(self, depth: torch.Tensor, conf: Optional[torch.Tensor]) -> torch.Tensor:  
        """Compute mask for bounded geometry using depth and confidence."""  
        # Depth-based mask  
        depth_mask = depth < self.max_depth  
          
        # Confidence-based mask  
        if conf is not None:  
            conf_thresh = torch.quantile(conf, self.conf_percentile / 100.0)  
            conf_mask = conf >= conf_thresh  
        else:  
            conf_mask = torch.ones_like(depth, dtype=torch.bool)  
          
        # Combine masks  
        bounded_mask = depth_mask & conf_mask  
          
        return bounded_mask  
      
    def _unproject_to_world(  
        self,   
        depth: torch.Tensor,   
        intrinsics: torch.Tensor,   
        extrinsics: torch.Tensor  
    ) -> torch.Tensor:  
        """Unproject depth to world coordinates."""  
        from .utils.geometry import as_homogeneous, affine_inverse  
        
        # Add channel dimension for unproject_depth  
        depth_expanded = depth[:, None, ..., None]  # (N, 1, H, W, 1)  
        
        # Add view dimension to intrinsics to match expected shape (N, 1, 3, 3)  
        intrinsics = intrinsics[:, None, :, :]  # (N, 3, 3) -> (N, 1, 3, 3)  
        
        # Convert extrinsics to homogeneous 4x4 and then to camera-to-world  
        extrinsics_homo = as_homogeneous(extrinsics)  # (N, 4, 4)  
        c2w = affine_inverse(extrinsics_homo)  # (N, 4, 4)  
        
        # Add view dimension to c2w to match expected shape (N, 1, 4, 4)  
        c2w = c2w[:, None, :, :]  # (N, 4, 4) -> (N, 1, 4, 4)  
        
        # Unproject to world space  
        world_points = unproject_depth(  
            depth=depth_expanded,  
            intrinsics=intrinsics,  
            c2w=c2w  
        )  # (N, 1, H, W, 3)  
        
        return world_points.squeeze(1)  # (N, H, W, 3)

    def _compute_bounded_bbox(  
        self,   
        world_points: torch.Tensor,   
        bounded_mask: torch.Tensor  
    ) -> Tuple[torch.Tensor, torch.Tensor]:  
        """Compute tight bounding box from bounded points."""  
        bounded_points = world_points[bounded_mask]  
          
        if bounded_points.numel() == 0:  
            # Fallback to full scene bounds  
            bbox_min = world_points.view(-1, 3).min(dim=0).values  
            bbox_max = world_points.view(-1, 3).max(dim=0).values  
        else:  
            bbox_min = bounded_points.min(dim=0).values  
            bbox_max = bounded_points.max(dim=0).values  
          
        return bbox_min, bbox_max  
      
    def _create_voxel_grid(  
        self,  
        world_points: torch.Tensor,  
        bounded_mask: torch.Tensor,  
        bbox_min: torch.Tensor,  
        bbox_max: torch.Tensor  
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  
        """Create voxel grid and map points to voxel indices."""  
        # Compute grid resolution  
        extent = bbox_max - bbox_min  
        grid_shape = ((extent / self.voxel_size).ceil() + 1).int()  
          
        # Initialize voxel grid  
        voxel_grid = torch.zeros(grid_shape.tolist(), dtype=torch.float32, device=world_points.device)  
          
        # Map bounded points to voxel indices  
        if bounded_mask.sum() > 0:  
            bounded_points = world_points[bounded_mask]  
            voxel_indices = ((bounded_points - bbox_min) / self.voxel_size).floor().long()  
              
            # Clip to grid bounds  
            voxel_indices = torch.clamp(voxel_indices, torch.zeros_like(voxel_indices), grid_shape - 1)  
              
            # Mark occupied voxels  
            unique_indices, counts = torch.unique(voxel_indices, dim=0, return_counts=True)  
            voxel_grid[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] = counts.float()  
        else:  
            voxel_indices = None  
          
        return voxel_grid, voxel_indices  
      
    def _apply_truncation(  
        self,  
        voxel_grid: torch.Tensor,  
        world_points: torch.Tensor,  
        depth: torch.Tensor,  
        conf: Optional[torch.Tensor],  
        voxel_indices: Optional[torch.Tensor],  
        bounded_mask: torch.Tensor  
    ) -> torch.Tensor:  
        """Apply simple confidence-based truncation."""  
        # If no confidence or voxel indices, return as-is  
        if conf is None or voxel_indices is None:  
            return voxel_grid  
        
        # Get confidence values for bounded points  
        bounded_conf = conf[bounded_mask]  
        
        # Simple confidence threshold (e.g., keep top 80%)  
        conf_thresh = torch.quantile(bounded_conf, 0.2)  
        reliable_mask = bounded_conf >= conf_thresh  
        
        # Filter voxel_indices using the reliable mask  
        if reliable_mask.sum() > 0:  
            reliable_voxel_indices = voxel_indices[reliable_mask]  
            
            # Create new grid with only reliable voxels  
            truncated_grid = torch.zeros_like(voxel_grid)  
            unique_indices, counts = torch.unique(reliable_voxel_indices, dim=0, return_counts=True)  
            truncated_grid[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] = counts.float()  
            return truncated_grid  
        
        return voxel_grid
    
    def _create_voxel_grid_with_colors(  
        self,  
        world_points: torch.Tensor,  
        bounded_mask: torch.Tensor,  
        bbox_min: torch.Tensor,  
        bbox_max: torch.Tensor,  
        images: Optional[torch.Tensor]  
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:  
        """Create voxel grid and compute average colors per voxel."""  
        # Compute grid resolution  
        extent = bbox_max - bbox_min  
        grid_shape = ((extent / self.voxel_size).ceil() + 1).int()  
        
        # Initialize voxel grid  
        voxel_grid = torch.zeros(grid_shape.tolist(), dtype=torch.float32, device=world_points.device)  
        
        # Initialize color accumulators  
        voxel_colors = None  
        if images is not None:  
            voxel_colors = torch.zeros(grid_shape.tolist() + [3], dtype=torch.float32, device=world_points.device)  
            voxel_counts = torch.zeros(grid_shape.tolist(), dtype=torch.float32, device=world_points.device)  
        
        # Map bounded points to voxel indices  
        if bounded_mask.sum() > 0:  
            bounded_points = world_points[bounded_mask]  
            voxel_indices = ((bounded_points - bbox_min) / self.voxel_size).floor().long()  
            
            # Clip to grid bounds  
            voxel_indices = torch.clamp(voxel_indices, torch.zeros_like(voxel_indices), grid_shape - 1)  
            
            # Accumulate colors per voxel  
            if images is not None:  
                # Get colors for bounded pixels  
                N, H, W = bounded_mask.shape  
                bounded_coords = torch.nonzero(bounded_mask, as_tuple=False)  
                
                # Map pixel coordinates to voxel indices  
                pixel_colors = images[bounded_coords[:, 0], bounded_coords[:, 1], bounded_coords[:, 2]]  # (K, 3)  
                
                # Accumulate colors  
                for i in range(len(voxel_indices)):  
                    idx = voxel_indices[i]  
                    voxel_colors[idx[0], idx[1], idx[2]] += pixel_colors[i]  
                    voxel_counts[idx[0], idx[1], idx[2]] += 1  
                
                # Compute average colors  
                valid_mask = voxel_counts > 0  
                # Get valid voxel coordinates from the 3D grid  
                valid_coords = torch.nonzero(valid_mask, as_tuple=False)  # (K_valid, 3)  
                
                # Create a mapping from voxel coordinates to their indices in voxel_indices  
                coord_to_idx = {tuple(coord.tolist()): idx for idx, coord in enumerate(voxel_indices)}  
                
                # Filter to keep only voxels that have accumulated colors  
                valid_indices = []  
                valid_colors = []  
                for coord in valid_coords:  
                    coord_tuple = tuple(coord.tolist())  
                    if coord_tuple in coord_to_idx:  
                        idx = coord_to_idx[coord_tuple]  
                        valid_indices.append(voxel_indices[idx])  
                        # Compute average color for this voxel  
                        color = voxel_colors[coord[0], coord[1], coord[2]] / voxel_counts[coord[0], coord[1], coord[2]]  
                        valid_colors.append(color)  
                
                if valid_indices:  
                    voxel_indices = torch.stack(valid_indices)  # (K_valid, 3)  
                    voxel_colors = torch.stack(valid_colors)    # (K_valid, 3)  
                else:  
                    voxel_indices = None  
                    voxel_colors = None
            
            # Mark occupied voxels  
            unique_indices, counts = torch.unique(voxel_indices, dim=0, return_counts=True)  
            voxel_grid[unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]] = counts.float()  
        else:  
            voxel_indices = None  
        
        return voxel_grid, voxel_indices, voxel_colors