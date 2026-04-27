# Auto-extracted from scripts/train_voxel_decoder.py
# Module: scene_discovery

import os
from pathlib import Path
import numpy as np

def discover_waymo_scenes(
    dataset_root: str,
    camera_name: str = "FRONT",
    exts=(".jpg", ".jpeg", ".png"),
):
    """
    Scan dataset structure like:

    dataset_root/
      scene_001/FRONT/*.jpg
      scene_002/FRONT/*.jpg
      ...

    Returns:
        scenes: list[dict]
            [
                {
                    "scene_name": "...",
                    "camera_name": "FRONT",
                    "image_dir": ".../scene_xxx/FRONT",
                    "image_paths": [...]
                },
                ...
            ]
    """
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
            f"No scenes found under {dataset_root} with camera folder '{camera_name}' "
            f"and extensions {exts}"
        )

    return scenes

def load_scene_name_list(txt_path: str) -> list[str]:
    if txt_path is None:
        return []

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"scene list file not found: {txt_path}")

    scene_names = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if len(name) == 0:
                continue
            scene_names.append(name)

    return scene_names

def filter_scenes_by_name(
    scenes: list[dict],
    selected_scene_names: list[str],
    mode: str = "restrict",
) -> list[dict]:
    selected_set = set(selected_scene_names)

    if mode == "restrict":
        filtered = [s for s in scenes if s["scene_name"] in selected_set]
    elif mode == "exclude":
        filtered = [s for s in scenes if s["scene_name"] not in selected_set]
    else:
        raise ValueError(f"Unsupported scene filter mode: {mode}")

    return filtered

def report_missing_scene_names(discovered_scenes: list[dict], requested_scene_names: list[str]):
    discovered = {s["scene_name"] for s in discovered_scenes}
    missing = sorted(set(requested_scene_names) - discovered)
    if len(missing) > 0:
        print(f"[WARN] {len(missing)} scene(s) from list file were not found in dataset:")
        for name in missing:
            print(f"  - {name}")

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
    input_indices = window_indices[::2]   # 偶數位置
    target_indices = window_indices[1::2] # 奇數位置

    if len(input_indices) == 0:
        input_indices = [window_indices[0]]
    if len(target_indices) == 0:
        target_indices = [window_indices[-1]]

    input_paths = [image_paths[i] for i in input_indices]
    target_global_indices = target_indices  # 對原 scene 的 index

    return {
        "start": start,
        "end": end,
        "window_indices": window_indices,
        "input_indices": input_indices,
        "target_indices": target_global_indices,
        "input_paths": input_paths,
    }
