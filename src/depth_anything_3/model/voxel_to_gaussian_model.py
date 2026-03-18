import torch.nn as nn

class VoxelToGaussianModel(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def build_decoder_inputs(self, voxel_dict, device):
        return {
            "anchor_xyz": voxel_dict["voxel_mean_points"].to(device),
            "dino_feat": voxel_dict["voxel_features"].to(device),
            "confidence": voxel_dict["voxel_confidence"].to(device),
            "cov_diag": voxel_dict["voxel_var_points"].to(device),
        }

    def flatten_gaussians(self, gaussian_out):
        return {
            "means3D": gaussian_out["centers"].reshape(-1, 3),
            "scales": gaussian_out["scales"].reshape(-1, 3),
            "rotations": gaussian_out["quaternions"].reshape(-1, 4),
            "opacity": gaussian_out["opacity"].reshape(-1, 1),
            "colors": gaussian_out["colors"].reshape(-1, 3),
        }

    def forward(self, voxel_dict, camera_xyz):
        device = voxel_dict["voxel_mean_points"].device
        decoder_inputs = self.build_decoder_inputs(voxel_dict, device)

        gaussian_out = self.decoder(
            anchor_xyz=decoder_inputs["anchor_xyz"],
            dino_feat=decoder_inputs["dino_feat"],
            confidence=decoder_inputs["confidence"],
            cov_diag=decoder_inputs["cov_diag"],
            camera_xyz=camera_xyz,
        )

        gaussians = self.flatten_gaussians(gaussian_out)
        return gaussian_out, gaussians