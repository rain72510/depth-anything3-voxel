import torch
import numpy as np
from PIL import Image

def tensor_to_uint8_image(x: torch.Tensor) -> np.ndarray:
    """
    x: [3,H,W], float in [0,1]
    return: uint8 [H,W,3]
    """
    x = x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return x
def save_tensor_image(x: torch.Tensor, path: str):
    Image.fromarray(tensor_to_uint8_image(x)).save(path)