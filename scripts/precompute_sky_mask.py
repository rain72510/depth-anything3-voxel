
"""
python scripts/precompute_sky_mask.py
  --dataset-root datasets/waymo
  --camera-name FRONT
  --cache-root output_train_voxel_decoder/cache
  --device cuda
  --save-preview

"""

import os
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from depth_anything_3.api import DepthAnything3


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def discover_waymo_scenes(
    dataset_root: str,
    camera_name: str = "FRONT",
    exts=(".jpg", ".jpeg", ".png"),
):
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {dataset_root}")
    if not root.is_dir():
        raise NotADirectoryError(f"dataset_root is not a directory: {dataset_root}")

    scenes = []
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue

        cam_dir = scene_dir / camera_name
        if not cam_dir.exists() or not cam_dir.is_dir():
            continue

        image_paths = []
        for ext in exts:
            image_paths.extend(sorted(cam_dir.glob(f"*{ext}")))
        image_paths = sorted(str(p) for p in image_paths)

        if len(image_paths) == 0:
            continue

        scenes.append(
            {
                "scene_name": scene_dir.name,
                "camera_name": camera_name,
                "image_dir": str(cam_dir),
                "image_paths": image_paths,
            }
        )

    if len(scenes) == 0:
        raise FileNotFoundError(
            f"No scenes found under {dataset_root} with camera folder '{camera_name}'"
        )
    return scenes


def load_segmentation_model(device: torch.device):
    model_name = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
    model.eval()

    id2label = model.config.id2label
    sky_class_ids = [int(k) for k, v in id2label.items() if str(v).lower() == "sky"]
    if len(sky_class_ids) == 0:
        raise RuntimeError("Cannot find 'sky' class in Mask2Former id2label.")

    print(f"[INFO] sky_class_ids = {sky_class_ids}")
    return processor, model, sky_class_ids


@torch.no_grad()
def segment_sky_from_processed_images(
    images_u8: np.ndarray,   # [V,H,W,3], uint8
    processor,
    seg_model,
    sky_class_ids,
    device: torch.device,
    batch_size: int = 2,
) -> np.ndarray:
    assert images_u8.ndim == 4 and images_u8.shape[-1] == 3, \
        f"Unexpected processed_images shape: {images_u8.shape}"

    V, H, W, _ = images_u8.shape
    masks = []

    for start in range(0, V, batch_size):
        end = min(start + batch_size, V)
        pil_images = [Image.fromarray(images_u8[i]) for i in range(start, end)]

        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = seg_model(**inputs)

        segs = processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(H, W)] * len(pil_images),
        )

        for seg in segs:
            sky_mask = torch.zeros_like(seg, dtype=torch.bool)
            for cid in sky_class_ids:
                sky_mask |= (seg == cid)
            masks.append(sky_mask.cpu().numpy())

    sky_mask = np.stack(masks, axis=0).astype(np.bool_)  # [V,H,W]
    return sky_mask


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    x = mask.astype(np.uint8) * 255
    return x


def save_preview_images(
    processed_images: np.ndarray,   # [V,H,W,3]
    sky_mask: np.ndarray,           # [V,H,W]
    save_dir: str,
    max_save: int = 3,
):
    ensure_dir(save_dir)
    V = processed_images.shape[0]
    num_save = min(V, max_save)

    for i in range(num_save):
        img = processed_images[i]
        mask = sky_mask[i]

        Image.fromarray(img).save(os.path.join(save_dir, f"view_{i:03d}_processed.png"))
        Image.fromarray(mask_to_uint8(mask)).save(os.path.join(save_dir, f"view_{i:03d}_sky_mask.png"))

        overlay = img.copy()
        overlay = overlay.astype(np.float32)
        overlay[mask] = 0.6 * overlay[mask] + 0.4 * np.array([255, 0, 0], dtype=np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        Image.fromarray(overlay).save(os.path.join(save_dir, f"view_{i:03d}_overlay.png"))


def process_scene(
    da3_model: DepthAnything3,
    seg_processor,
    seg_model,
    sky_class_ids,
    image_paths,
    scene_cache_dir: str,
    device: torch.device,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    seg_batch_size: int = 2,
    overwrite: bool = False,
    save_preview: bool = False,
):
    ensure_dir(scene_cache_dir)

    sky_mask_path = os.path.join(scene_cache_dir, "sky_mask.npz")
    meta_path = os.path.join(scene_cache_dir, "sky_mask_meta.npz")

    if os.path.exists(sky_mask_path) and not overwrite:
        print(f"[INFO] skip existing: {sky_mask_path}")
        return

    print(f"[INFO] running DA3 inference for {scene_cache_dir}")
    with torch.no_grad():
        prediction = da3_model.inference(
            image_paths,
            export_dir=scene_cache_dir,
            export_format="none",
            process_res=process_res,
            process_res_method=process_res_method,
        )

    processed_images = prediction.processed_images  # [V,H,W,3], uint8
    print(f"[INFO] processed_images shape = {processed_images.shape}")

    sky_mask = segment_sky_from_processed_images(
        images_u8=processed_images,
        processor=seg_processor,
        seg_model=seg_model,
        sky_class_ids=sky_class_ids,
        device=device,
        batch_size=seg_batch_size,
    )

    valid_mask = ~sky_mask
    valid_pixel_ratio = valid_mask.mean(dtype=np.float64)

    np.savez_compressed(
        sky_mask_path,
        sky_mask=sky_mask.astype(np.uint8),   # compact storage
    )
    np.savez_compressed(
        meta_path,
        process_res=np.array([process_res], dtype=np.int32),
        process_res_method=np.array([process_res_method]),
        num_views=np.array([processed_images.shape[0]], dtype=np.int32),
        height=np.array([processed_images.shape[1]], dtype=np.int32),
        width=np.array([processed_images.shape[2]], dtype=np.int32),
        valid_pixel_ratio=np.array([valid_pixel_ratio], dtype=np.float32),
    )

    print(
        f"[INFO] saved sky mask: {sky_mask_path} | "
        f"shape={sky_mask.shape}, valid_pixel_ratio={valid_pixel_ratio:.4f}"
    )

    if save_preview:
        save_preview_images(
            processed_images=processed_images,
            sky_mask=sky_mask,
            save_dir=os.path.join(scene_cache_dir, "sky_preview"),
            max_save=3,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--camera-name", type=str, default="FRONT")
    parser.add_argument("--cache-root", type=str, required=True)
    parser.add_argument("--model-id", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", type=str, default="upper_bound_resize")
    parser.add_argument("--seg-batch-size", type=int, default=2)
    parser.add_argument("--scene-names", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    scenes = discover_waymo_scenes(
        dataset_root=args.dataset_root,
        camera_name=args.camera_name,
    )

    if args.scene_names is not None and len(args.scene_names) > 0:
        scene_names_set = set(args.scene_names)
        scenes = [s for s in scenes if s["scene_name"] in scene_names_set]

    print(f"[INFO] num_scenes = {len(scenes)}")

    da3_model = DepthAnything3.from_pretrained(args.model_id).to(device)
    da3_model.eval()
    for p in da3_model.parameters():
        p.requires_grad = False

    seg_processor, seg_model, sky_class_ids = load_segmentation_model(device)

    for scene in scenes:
        scene_name = scene["scene_name"]
        scene_cache_dir = os.path.join(args.cache_root, scene_name)

        try:
            process_scene(
                da3_model=da3_model,
                seg_processor=seg_processor,
                seg_model=seg_model,
                sky_class_ids=sky_class_ids,
                image_paths=scene["image_paths"],
                scene_cache_dir=scene_cache_dir,
                device=device,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                seg_batch_size=args.seg_batch_size,
                overwrite=args.overwrite,
                save_preview=args.save_preview,
            )
        except Exception as e:
            print(f"[WARN] failed scene {scene_name}: {e}")

    print("[INFO] done.")


if __name__ == "__main__":
    main()