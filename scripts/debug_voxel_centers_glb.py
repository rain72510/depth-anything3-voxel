import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.sparse_voxelizer import SparseVoxelizer


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def discover_scene_images(dataset_root: str, scene_name: str, camera_name: str = "FRONT"):
    cam_dir = Path(dataset_root) / scene_name / camera_name
    if not cam_dir.exists():
        raise FileNotFoundError(f"Camera dir not found: {cam_dir}")

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(sorted(cam_dir.glob(ext)))
    image_paths = sorted(str(p) for p in image_paths)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {cam_dir}")

    return image_paths


def sample_even_odd_window(image_paths, sequence_length, stride=1):
    num_imgs = len(image_paths)
    if num_imgs < sequence_length:
        start = 0
        end = num_imgs
    else:
        max_start = num_imgs - sequence_length
        start = np.random.randint(0, max_start + 1)
        end = start + sequence_length

    window_indices = list(range(start, end, stride))
    input_indices = window_indices[::2]
    target_indices = window_indices[1::2]

    if len(input_indices) == 0:
        input_indices = [window_indices[0]]
    if len(target_indices) == 0:
        target_indices = [window_indices[-1]]

    input_paths = [image_paths[i] for i in input_indices]

    return {
        "start": start,
        "end": end,
        "window_indices": window_indices,
        "input_indices": input_indices,
        "target_indices": target_indices,
        "input_paths": input_paths,
    }


def transform_for_viewer(xyz: np.ndarray, mode: str = "flip_y") -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).copy()

    if mode == "none":
        return xyz
    elif mode == "flip_y":
        xyz[:, 1] *= -1.0
    elif mode == "flip_z":
        xyz[:, 2] *= -1.0
    elif mode == "swap_yz":
        xyz = xyz[:, [0, 2, 1]]
    elif mode == "swap_yz_flip_z":
        xyz = xyz[:, [0, 2, 1]]
        xyz[:, 2] *= -1.0
    else:
        raise ValueError(f"Unknown transform mode: {mode}")

    return xyz


def write_ascii_ply_xyzrgb(path: str, xyz: np.ndarray, rgb: np.ndarray):
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be [N,3], got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be [N,3], got {rgb.shape}")
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must have same length")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def export_glb_pointcloud(path: str, xyz: np.ndarray, rgb: np.ndarray):
    import trimesh

    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)

    if rgb.shape[0] != xyz.shape[0]:
        raise ValueError("xyz/rgb size mismatch")

    if rgb.shape[1] == 3:
        alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgb, alpha], axis=1)
    elif rgb.shape[1] == 4:
        rgba = rgb
    else:
        raise ValueError("rgb must be [N,3] or [N,4]")

    pc = trimesh.points.PointCloud(vertices=xyz, colors=rgba)
    glb_bytes = trimesh.exchange.gltf.export_glb(pc)
    with open(path, "wb") as f:
        f.write(glb_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--scene-name", type=str, required=True)
    parser.add_argument("--camera-name", type=str, default="FRONT")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--model-id", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--voxel-size", type=float, default=0.4)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--conf-percentile", type=float, default=40.0)
    parser.add_argument("--truncation-band", type=float, default=0.5)
    parser.add_argument("--feat-mode", type=str, default="last2_avg")
    parser.add_argument("--feat-dim-out", type=int, default=-1)

    parser.add_argument(
        "--supervision-mode",
        type=str,
        default="even_input_mixed_supervision",
        choices=["random_views", "even_input_mixed_supervision"],
    )
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--sequence-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--viewer-transform",
        type=str,
        default="flip_y",
        choices=["none", "flip_y", "flip_z", "swap_yz", "swap_yz_flip_z"],
        help="Coordinate transform only for viewer/export. Does not change voxelization itself.",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ensure_dir(args.output_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_image_paths = discover_scene_images(
        dataset_root=args.dataset_root,
        scene_name=args.scene_name,
        camera_name=args.camera_name,
    )

    if args.supervision_mode == "even_input_mixed_supervision":
        split = sample_even_odd_window(
            all_image_paths,
            sequence_length=min(args.sequence_length, len(all_image_paths)),
            stride=args.sequence_stride,
        )
        input_image_paths = split["input_paths"]
        split_info = split
    else:
        input_image_paths = all_image_paths
        split_info = {
            "start": 0,
            "end": len(all_image_paths),
            "window_indices": list(range(len(all_image_paths))),
            "input_indices": list(range(len(all_image_paths))),
            "target_indices": list(range(len(all_image_paths))),
        }

    print(f"[INFO] scene_name={args.scene_name}")
    print(f"[INFO] total scene views={len(all_image_paths)}")
    print(f"[INFO] input views used for voxelization={len(input_image_paths)}")
    print(f"[INFO] viewer_transform={args.viewer_transform}")

    model = DepthAnything3.from_pretrained(args.model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    feat_dim_out = None if args.feat_dim_out < 0 else args.feat_dim_out

    voxelizer = SparseVoxelizer(
        max_depth=args.max_depth,
        voxel_size=args.voxel_size,
        conf_percentile=args.conf_percentile,
        truncation_band=args.truncation_band,
        feat_mode=args.feat_mode,
        feat_dim_out=feat_dim_out,
    )

    with torch.no_grad():
        prediction = model.inference(
            input_image_paths,
            export_dir=args.output_dir,
            export_format="none",
        )

    voxel_dict = voxelizer.voxelize_prediction(prediction)
    if voxel_dict["num_voxels"] == 0:
        raise RuntimeError("Voxelization returned zero voxels.")

    centers = voxel_dict["voxel_mean_points"].detach().cpu().numpy()
    centers_view = transform_for_viewer(centers, mode=args.viewer_transform)

    voxel_colors = voxel_dict.get("voxel_colors", None)
    if voxel_colors is None:
        rgb = np.full((centers.shape[0], 3), 200, dtype=np.uint8)
    else:
        rgb = voxel_colors.detach().cpu().numpy()
        if rgb.max() <= 1.0:
            rgb = (rgb * 255.0).round().clip(0, 255).astype(np.uint8)
        else:
            rgb = rgb.round().clip(0, 255).astype(np.uint8)

    ply_path = os.path.join(args.output_dir, "voxel_centers_color.ply")
    glb_path = os.path.join(args.output_dir, "voxel_centers_color.glb")
    summary_path = os.path.join(args.output_dir, "summary.json")

    write_ascii_ply_xyzrgb(ply_path, centers_view, rgb)
    export_glb_pointcloud(glb_path, centers_view, rgb)

    stats = voxel_dict.get("stats", {})
    summary = {
        "scene_name": args.scene_name,
        "camera_name": args.camera_name,
        "num_total_scene_views": len(all_image_paths),
        "num_input_views_used_for_voxelization": len(input_image_paths),
        "split_info": split_info,
        "voxel_size": args.voxel_size,
        "max_depth": args.max_depth,
        "conf_percentile": args.conf_percentile,
        "truncation_band": args.truncation_band,
        "feat_mode": args.feat_mode,
        "feat_dim_out": feat_dim_out,
        "viewer_transform": args.viewer_transform,
        "num_voxels": int(voxel_dict["num_voxels"]),
        "num_points": int(voxel_dict["num_points"]),
        "stats": stats,
        "outputs": {
            "ply": ply_path,
            "glb": glb_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[DONE] saved: {ply_path}")
    print(f"[DONE] saved: {glb_path}")
    print(f"[DONE] saved: {summary_path}")


if __name__ == "__main__":
    main()