import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 inference with Gaussian Splatting support."
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to image or directory containing images (supports png, jpg, jpeg)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="da3nested-giant-large",
        choices=["da3-small", "da3-base", "da3-large", "da3-giant", "da3nested-giant-large"],
        help="Model name to use for inference"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="npz-glb-gs_ply-gs_video",
        help="Export format (comma-separated list)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process from directory"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip visualization of results"
    )
    parser.add_argument(
        "--use-anysplat",
        action="store_true",
        help="Use AnySplat-style voxelization for GS export",
    )
    return parser.parse_args()


def get_image_paths(image_path):
    """Get image paths from file or directory."""
    if os.path.isfile(image_path):
        # Single image file
        return [image_path]
    elif os.path.isdir(image_path):
        # Directory of images
        supported_formats = ('.png', '.jpg', '.jpeg')
        image_files = [
            os.path.join(image_path, f) 
            for f in os.listdir(image_path) 
            if f.lower().endswith(supported_formats)
        ]
        if not image_files:
            raise ValueError(f"No image files found in {image_path}")
        return sorted(image_files)
    else:
        raise FileNotFoundError(f"Image path not found: {image_path}")


def main():
    """Main inference function."""
    args = parse_arguments()
    
    # Get image paths
    image_paths = get_image_paths(args.image_path)
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    n_images = len(image_paths)
    print(f"Processing {n_images} image(s)...")
    print(f"Image paths: {image_paths}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model '{args.model_name}' on {device}...")
    # model = DepthAnything3(model_name=args.model_name).to(device)
    model_dir = "depth-anything/DA3NESTED-GIANT-LARGE"
    model = DepthAnything3.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    # Run inference
    print(f"Running inference with export format: {args.export_format}")
    prediction = model.inference(
        image=image_paths,
        extrinsics=None,
        intrinsics=None,
        export_dir=args.output_dir,
        export_format=args.export_format,
        align_to_input_ext_scale=True,
        infer_gs=True,  # Required for gs_ply and gs_video exports
        # export_kwargs={"gs_ply": {"use_anysplat": args.use_anysplat}},
    )
    
    print(f"Inference completed. Results saved to {args.output_dir}")
    
    # Visualization
    if not args.no_visualization:
        print("Generating visualization...")
        fig, axes = plt.subplots(2, n_images, figsize=(12, 6))
        
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_images):
            # Show original image
            if prediction.processed_images is not None:
                axes[0, i].imshow(prediction.processed_images[i])
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            
            # Show depth map
            depth_vis = visualize_depth(prediction.depth[i], cmap="Spectral")
            axes[1, i].imshow(depth_vis)
            axes[1, i].set_title(f"Depth {i+1}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()