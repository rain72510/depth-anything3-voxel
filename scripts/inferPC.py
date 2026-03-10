import glob, os, torch, numpy as np  
from pathlib import Path  
from depth_anything_3.api import DepthAnything3  
  
# === 設定 ===  
device = torch.device("cuda")  
model_id = "depth-anything/DA3NESTED-GIANT-LARGE"  
example_path = "assets/examples/SOH"          # 改成你的影像資料夾  
export_dir = "./output_scene"                # 輸出資料夾  
  
# === 載模型 ===  
model = DepthAnything3.from_pretrained(model_id).to(device)  
  
# === 讀影像清單 ===  
images = sorted(glob.glob(os.path.join(example_path, "*.png")))  
print(f"Found {len(images)} images")  
  
# === 推理並匯出 GLB（含點雲與相機）與可選的 PLY ===  
# GLB 可直接在 SuperSplat 開啟；PLY 也可開啟（若需純點雲）  
prediction = model.inference(  
    images,  
    export_dir=export_dir,  
    export_format="glb-ply",          # 同時匯出 GLB 與 PLY  
    conf_thresh_percentile=40.0,      # GLB 點雲信心過濾  
    num_max_points=1_000_000,         # GLB 最大點數  
    show_cameras=True,                # GLB 內顯示相機線框  
)  
  
# === 儲存 processed_images 與 depth 到 npz（方便後續使用） ===  
npz_path = Path(export_dir) / "predictions.npz"  
np.savez_compressed(  
    npz_path,  
    processed_images=prediction.processed_images,  # [N, H, W, 3] uint8  
    depth=prediction.depth,                        # [N, H, W] float32  
    conf=prediction.conf,                          # [N, H, W] float32  
    extrinsics=prediction.extrinsics,              # [N, 3, 4] float32  
    intrinsics=prediction.intrinsics,              # [N, 3, 3] float32  
)  
print(f"Saved predictions.npz to {npz_path}")  
  
# === 可選：單獨存出深度圖為彩色影像（用於快速檢視） ===  
# 若要深度可視化，可在 inference 時加入 export_format="glb-depth_vis"  
# 這會在 export_dir/depth_vis/ 下產生與影像同名的深度彩色圖  
  
print("Done. Open the following in SuperSplat or a web viewer:")  
print(f"- GLB: {Path(export_dir) / 'scene.glb'}")  
print(f"- PLY: {Path(export_dir) / 'scene.ply'}")