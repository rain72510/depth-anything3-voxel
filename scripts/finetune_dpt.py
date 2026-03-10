from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from depth_anything_3.utils.export.anysplat_voxel import voxelize_prediction
from depth_anything_3.utils.export.gs import export_to_gs_ply
from depth_anything_3.utils.gsply_helpers import export_ply


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default="da3nested-giant-large", help="model preset name from configs")
    p.add_argument("--out-dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--mode", type=str, choices=["dpt", "gs"], default="gs",
                   help="Which part to finetune: 'dpt' or 'gs' (gaussian decoder)")
    p.add_argument("--image-dir", type=str, default=None, help="Directory with input images (for gs mode)")
    p.add_argument("--poses-npz", type=str, default=None, help=".npz file with 'extrinsics' and 'intrinsics' arrays (for gs mode)")
    p.add_argument("--sample-scene", type=str, default=None, help="Path to a single scene to run a dry-run on (optional)")
    p.add_argument("--dry-run-only", action="store_true", help="Run single-batch dry-run and exit")

    # AMP controls
    p.add_argument("--amp", action="store_true", help="Enable AMP (autocast + GradScaler)")
    p.add_argument("--amp-dtype", type=str, default="fp16", choices=["fp16", "bf16"],
                   help="AMP dtype (fp16 or bf16). bf16 requires newer GPUs.")
    p.add_argument("--debug-mem", action="store_true")
    p.add_argument("--topk", type=int, default=0, help="render only top-k gaussians per view by opacity (0=disable)")


    return p.parse_args()


class VoxelGaussDataset(Dataset):
    def __init__(self, device: torch.device, mode: str = "gs",
                 image_dir: Optional[str] = None, poses_npz: Optional[str] = None):
        self.device = device
        self.mode = mode
        self.image_dir = image_dir
        self.poses = None
        if mode == "gs":
            if poses_npz is not None:
                data = np.load(poses_npz)
                self.poses = {"extrinsics": data["extrinsics"], "intrinsics": data["intrinsics"]}
            else:
                self.poses = None

        if mode == "gs":
            print(f"Collecting images from {image_dir} for 'gs' mode dataset")
            if image_dir is None:
                self.img_files = []
            else:
                root = Path(image_dir)
                jpgs = sorted(root.glob("*/FRONT/*.jpg"))
                pngs = sorted(root.glob("*/FRONT/*.png"))
                print(f"Found {len(jpgs)} JPGs and {len(pngs)} PNGs")
                self.img_files = [p for p in jpgs] + [p for p in pngs]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        return img_np, None, None, None


def collate_fn(batch):
    imgs = [b[0] for b in batch]
    depth_items = [b[1] for b in batch]
    if any(d is None for d in depth_items):
        depths = None
    else:
        depths = torch.stack(depth_items, dim=0)
    extrs = [b[2] for b in batch]
    ixts = [b[3] for b in batch]
    return imgs, depths, extrs, ixts

def slice_gaussians_topk_by_opacity(gs: Any, k: int) -> Any:
    """
    Slice gaussian fields to top-k by opacity.
    Supports opacities shape: (B,N) or (B,N,1) or (N,) or (N,1) with B=1 case.
    Assumes we call this on per-view gs_i (B=1) OR on a gs with leading dim B already sliced.
    """
    if k <= 0:
        return gs

    # find opacity field
    op = None
    for nm in ("opacities", "opacity", "alphas", "alpha"):
        if hasattr(gs, nm):
            op = getattr(gs, nm)
            break
    if op is None or (not torch.is_tensor(op)):
        return gs

    # Normalize opacity to shape (N,)
    if op.dim() == 2:
        # (B,N) or (N,1) unlikely
        if op.shape[0] == 1:
            op_1 = op[0]          # (N,)
        else:
            # If still batched (B,N), caller should have sliced to B=1 first.
            # Fallback: use first batch item.
            op_1 = op[0]
    elif op.dim() == 3:
        # (B,N,1) common
        if op.shape[0] == 1:
            op_1 = op[0, :, 0]
        else:
            op_1 = op[0, :, 0]
    elif op.dim() == 1:
        op_1 = op
    else:
        # unexpected layout; don't slice
        return gs

    N = op_1.numel()
    k = min(k, N)
    idx = torch.topk(op_1, k=k, largest=True, sorted=False).indices

    import copy
    gs2 = copy.copy(gs)

    # Slice all tensor fields that look like (1,N,...) or (N,...)
    for attr in ("harmonics", "rotations", "means", "scales", "opacities", "opacity", "alphas", "alpha"):
        if not hasattr(gs, attr):
            continue
        v = getattr(gs, attr)
        if not torch.is_tensor(v):
            continue

        if v.dim() >= 2 and v.shape[0] == 1 and v.shape[1] == N:
            # (1,N,...) -> keep batch dim
            setattr(gs2, attr, v[:, idx, ...])
        elif v.dim() >= 1 and v.shape[0] == N:
            # (N,...) no batch dim
            setattr(gs2, attr, v[idx, ...])
        elif v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == N:
            setattr(gs2, attr, v[:, idx])
        elif v.dim() == 1 and v.shape[0] == N:
            setattr(gs2, attr, v[idx])

    return gs2


def main():
    args = parse_args()
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    from depth_anything_3.api import DepthAnything3
    # api = DepthAnything3(model_name=args.model_name)
    api = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
    net = api.model
    net.to(device)
    net.train()

    # AMP setup
    use_amp = bool(args.amp) and (device.type == "cuda")
    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # dump model structure
    with open("model_structure.txt", "w") as f:
        for name, module in net.named_modules():
            num_params = sum(p.numel() for p in module.parameters())
            f.write(f"{name}: {module.__class__.__name__}, params: {num_params / 1e6:.3f}M\n")
    print("Model structure written to model_structure.txt")

    # ---- NOTE: 你這裡目前是解凍 gs_adapter / gs_head。
    # 若你真正要「只 finetune GS-DPT head」，請把 target_prefixes 換成正確的 head module name。
    gs_params = []
    gs_module_names = []
    target_prefixes = ("da3.gs_adapter", "da3.gs_head")  # TODO: 改成你真正的 GS-DPT head prefix
    for name, m in net.named_modules():
        if any(name == tp or name.startswith(tp + ".") for tp in target_prefixes):
            gs_module_names.append(name)
            for p in m.parameters():
                gs_params.append(p)

    # freeze all
    for _, p in net.named_parameters():
        p.requires_grad = False

    # unfreeze target params
    for p in gs_params:
        p.requires_grad = True

    params = [p for p in gs_params if p is not None]
    if len(params) == 0:
        raise RuntimeError("No target parameters found to optimize. Check target_prefixes.")
    total_params = sum(p.numel() for p in net.parameters())
    train_param_count = sum(p.numel() for p in params)
    print(f"Total params: {total_params:,}, train params: {train_param_count:,}")
    print(f"train modules: {gs_module_names}")

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    loss_fn = torch.nn.L1Loss()
    os.makedirs(args.out_dir, exist_ok=True)

    use_photometric = args.mode == "gs"

    # ----------------------------
    # Differentiable render helpers
    # ----------------------------
    from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
    import copy

    def render_one_view(gaussians: Any,
                        extr_1: torch.Tensor,
                        ixt_1: torch.Tensor,
                        hw: tuple[int, int],
                        chunk_size: int = 1) -> torch.Tensor:
        """
        Render a SINGLE view (B=1 for extr/ixt).
        Returns: (1,3,H,W)
        """
        # slice gaussian batch dim to 1 if needed
        B = extr_1.shape[0]
        assert B == 1, "render_one_view expects B=1"

        gs_i = copy.copy(gaussians)
        if args.topk > 0:
            gs_i = slice_gaussians_topk_by_opacity(gs_i, args.topk)
        # if gaussians fields are batched, slice to [bi:bi+1]
        for k in dir(gaussians):
            if k.startswith("_"):
                continue
            try:
                v = getattr(gaussians, k)
            except Exception:
                continue
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] >= 1:
                # if original gaussians is batched over B, we'll pass already-sliced from caller;
                # if it's not batched, this will just keep it unchanged (v.shape[0] may be N not B).
                # We only slice when it looks like leading dim is batch (caller will provide slice).
                pass

        n_views = extr_1.shape[1] if extr_1 is not None else 1
        trj_mode = "wander" if n_views == 1 else "original"

        color, _ = run_renderer_in_chunk_w_trj_mode(gs_i, extr_1, ixt_1, hw, trj_mode=trj_mode, chunk_size=chunk_size)
        rendered = color[:, 0, ...] if color.dim() == 5 else color  # (1,3,H,W)
        return rendered

    def photometric_backward_per_view(gaussians: Any,
                                     pred_extr: torch.Tensor,
                                     pred_ixt: torch.Tensor,
                                     gt_imgs: torch.Tensor,
                                     hw: tuple[int, int],
                                     chunk_size: int = 1,
                                     save_render_for_debug: bool = False):
        """
        Method-1: render per view and backward per view to release renderer buffers ASAP.
        - gaussians: output from net for the whole batch
        - pred_extr/pred_ixt: (B,1,3,4)/(B,1,3,3)
        - gt_imgs: (B,3,H,W)
        Returns: total_loss (python float), maybe_rendered (Tensor or None)
        """
        def check_pose(name, x):
            print(name, x.shape, x.dtype, x.device,
                "nan", torch.isnan(x).any().item(),
                "inf", torch.isinf(x).any().item(),
                "min", x.min().item(), "max", x.max().item())
            # print x
            for i in range(min(2, x.shape[0])):
                print(f" {name}[{i}]:", x[i])
            
        import time
        torch.cuda.synchronize()
        t0 = time.time()
        print("[dbg] enter photometric_backward_per_view")
        B = pred_extr.shape[0]
        check_pose("pred_extr", pred_extr)
        total_loss_val = 0.0
        rendered_first = None

        # important: we keep the forward graph from net() while doing multiple backward calls
        for bi in range(B):
            # slice inputs to (1,1,*,*)
            torch.cuda.synchronize()
            print(f"[dbg] view {bi} start, dt={time.time()-t0:.3f}s")
            extr_i = pred_extr[bi:bi+1]
            ixt_i = pred_ixt[bi:bi+1]
            gt_i = gt_imgs[bi:bi+1]

            # slice gaussians if it is batched (leading dim == B)
            gs_i = copy.copy(gaussians)
            for k in dir(gaussians):
                if k.startswith("_"):
                    continue
                try:
                    v = getattr(gaussians, k)
                except Exception:
                    continue
                if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == B:
                    setattr(gs_i, k, v[bi:bi+1])

            rendered_i = render_one_view(gs_i, extr_i, ixt_i, hw, chunk_size=chunk_size)

            if save_render_for_debug and rendered_first is None:
                rendered_first = rendered_i

            loss_i = torch.nn.functional.l1_loss(rendered_i, gt_i) / B

            # backward immediately; keep graph except for last view
            if use_amp:
                scaler.scale(loss_i).backward(retain_graph=(bi != B - 1))
            else:
                loss_i.backward(retain_graph=(bi != B - 1))

            total_loss_val += float(loss_i.detach().item()) * B  # store un-normalized view loss

            # cleanup refs
            del gs_i, extr_i, ixt_i, gt_i, rendered_i, loss_i
            torch.cuda.empty_cache()

        return total_loss_val / B, rendered_first  # average loss

    # ----------------------------
    # dry-run loader
    # ----------------------------
    def make_scene_loader(scene_path: str, bs: int = 1):
        scene_front = Path(scene_path) / "FRONT"
        imgs = sorted(scene_front.glob("*.jpg")) + sorted(scene_front.glob("*.png"))
        print(f"Making scene loader for {scene_path}, found {len(imgs)} images")
        # for i, p in enumerate(imgs):
        #     print(f" Image {i}: {p}")

        class _DS(Dataset):
            def __len__(self):
                return len(imgs)

            def __getitem__(self, idx):
                from PIL import Image
                p = imgs[idx]
                img = Image.open(p).convert("RGB")
                arr = np.array(img)
                return arr, None, None, None

        return DataLoader(_DS(), batch_size=bs, shuffle=False, collate_fn=collate_fn)

    sample_scene = args.sample_scene
    default_scene = "datasets/waymo/10275144660749673822_5755_561_5775_561"
    if sample_scene is None and Path(default_scene).exists():
        sample_scene = default_scene

    # ----------------------------
    # DRY RUN
    # ----------------------------
    if sample_scene is not None:
        if args.debug_mem:
            from depth_anything_3.utils.mem_debug import summarize_gaussians, CUDAPeakMem
        else:
            summarize_gaussians = None
            CUDAPeakMem = None
        print(f"Found sample scene: {sample_scene} -> running single-batch dry-run")
        scene_loader = make_scene_loader(sample_scene, bs=min(args.batch_size, 8))

        try:
            imgs, depths, extrs, ixts = next(iter(scene_loader))
        except StopIteration:
            print("No images found in sample scene FRONT folder; skipping dry-run")
            imgs = None

        if imgs is not None:
            imgs_list = imgs
            print(f"Running dry-run with batch size {len(imgs_list)}")

            imgs_cpu, ex_t_cpu, ixt_t_cpu = api._preprocess_inputs(imgs_list, None, None)
            imgs_tensor, ex_t_raw, ixt_t_raw = api._prepare_model_inputs(imgs_cpu, ex_t_cpu, ixt_t_cpu)
            imgs_model = imgs_tensor.permute(1, 0, 2, 3, 4).to(device)  # (B,1,3,H,W)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                # out = net(imgs_model, None, None, infer_gs=True)
                ex_t_norm = api._normalize_extrinsics(ex_t_raw.clone() if ex_t_raw is not None else None)
                out = api._run_model_forward(
                    imgs_model, ex_t_norm, ixt_t_raw, [], infer_gs=True, use_ray_pose = False, ref_view_strategy = "saddle_balanced"
                )
                
            pred_extr = out.get("extrinsics", None)
            pred_ixt = out.get("intrinsics", None)

            prediction = api._convert_to_prediction(out)
            prediction = api._align_to_input_extrinsics_intrinsics(
                pred_extr.detach().cpu(), pred_ixt.detach().cpu(), prediction, True
            )
            prediction = api._add_processed_images(prediction, imgs_cpu)
            print(f"3. prediction.gaussians.means shape: {prediction.gaussians.means.shape}")
                # out = voxelize_prediction(out)


            gaussians = prediction.gaussians
            if gaussians is None:
                raise RuntimeError("Model did not return 'gaussians' in output during dry-run")
            if args.debug_mem:
                summarize_gaussians(gaussians, only_cuda=True)
            
            print(f"gaussians.means shape: {gaussians.means.shape}, dtype: {gaussians.means.dtype}, device: {gaussians.means.device}")
            gauss = gaussians  # assume dataclass-like with tensors
            B, N, _ = gauss.means.shape

            means_flat = gauss.means.reshape(1, B * N, 3)           # (1, B*N, 3)
            scales_flat = gauss.scales.reshape(1, B * N, 3)         # (1, B*N, 3)
            rots_flat = gauss.rotations.reshape(1, B * N, 4)        # (1, B*N, 4)

            # harmonics may have tail dims, preserve them:
            harm_tail = gauss.harmonics.shape[2:]                   # e.g., (C,)
            harm_flat = gauss.harmonics.reshape(1, B * N, *harm_tail)

            # opacities shape handling (B,N,1) or (B,N)
            op = gauss.opacities
            if op.ndim == 3 and op.shape[-1] == 1:
                op_flat = op.reshape(1, B * N, 1)
            else:
                op_flat = op.reshape(1, B * N)

            from depth_anything_3.specs import Gaussians
            merged = Gaussians(
                means=means_flat, scales=scales_flat, rotations=rots_flat,
                harmonics=harm_flat, opacities=op_flat
            )

            # assign back to prediction if needed
            prediction.gaussians = merged
            
            export_dir = Path("./dryrun_output")
            export_dir.mkdir(parents=True, exist_ok=True)

            # Normalize prediction.depth to shape (v, H, W) expected by exporter.
            # Possible incoming shapes: (B, v, H, W) or (v, H, W) or (1, v, H, W)
            if hasattr(prediction, "depth") and isinstance(prediction.depth, np.ndarray):
                d = prediction.depth
                if d.ndim == 4:
                    # (B, v, H, W) -> (B*v, H, W)
                    B, v, H, W = d.shape
                    prediction.depth = d.reshape(B * v, H, W)
                elif d.ndim == 5 and d.shape[0] == 1:
                    # (1, v, H, W, ?) unlikely but handle
                    prediction.depth = d.squeeze(0)

            from depth_anything_3.utils.export import export_to_gs_ply
            # export_to_gs_ply expects an export directory
            export_to_gs_ply(prediction, str(export_dir), use_anysplat=False)


            if pred_extr is None or pred_ixt is None:
                raise RuntimeError("Model did not return predicted extrinsics/intrinsics required for rendering during dry-run")

            print(f"pred_extr shape: {pred_extr.shape}, pred_ixt shape: {pred_ixt.shape}")
            pred_extr = pred_extr.to(device)
            pred_ixt = pred_ixt.to(device)

            H, W = imgs_model.shape[-2], imgs_model.shape[-1]
            gt = imgs_model[:, 0, ...].to(device)  # (B,3,H,W)

            # AMP for renderer+loss too
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                avg_loss, rendered_first = photometric_backward_per_view(
                    gaussians, pred_extr, pred_ixt, gt, (H, W),
                    chunk_size=1, save_render_for_debug=True
                )

            # save debug images (use first rendered view to avoid huge concat)
            import torchvision.utils as vutils
            if rendered_first is not None:
                vutils.save_image(rendered_first.clamp(0, 1).detach().cpu(), "dryrun_rendered_first.png")
            vutils.save_image(gt[0:1].clamp(0, 1).detach().cpu(), "dryrun_gt_first.png")
            print("Saved dry-run images to dryrun_rendered_first.png and dryrun_gt_first.png")

            # step
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            grad_count = sum(1 for p in params if p.grad is not None)
            print(f"Dry-run avg loss: {avg_loss:.6f}; train params with grads: {grad_count}/{len(params)}")

        if args.dry_run_only:
            print("Dry-run complete, exiting as --dry-run-only specified")
            return

    # ----------------------------
    # TRAIN LOOP (non-dry-run)
    # ----------------------------
    dataset = VoxelGaussDataset(device=device, mode=args.mode, image_dir=args.image_dir, poses_npz=args.poses_npz)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_steps = 0

        for i, (imgs, depths, extrs, ixts) in enumerate(loader):
            imgs_list = imgs
            imgs_cpu, ex_t_cpu, ixt_t_cpu = api._preprocess_inputs(imgs_list, None, None)
            imgs_tensor, ex_t_raw, ixt_t_raw = api._prepare_model_inputs(imgs_cpu, ex_t_cpu, ixt_t_cpu)
            imgs_model = imgs_tensor.permute(1, 0, 2, 3, 4).to(device)  # (B,1,3,H,W)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = net(imgs_model, None, None, infer_gs=True)

            if use_photometric:
                gaussians = out.get("gaussians", None)
                if gaussians is None:
                    raise RuntimeError("Model did not return 'gaussians' in output when infer_gs=True")

                pred_extr = out.get("extrinsics", None)
                pred_ixt = out.get("intrinsics", None)
                if pred_extr is None or pred_ixt is None:
                    raise RuntimeError("Model did not return predicted extrinsics/intrinsics required for rendering")
                pred_extr = pred_extr.to(device)
                pred_ixt = pred_ixt.to(device)

                H, W = imgs_model.shape[-2], imgs_model.shape[-1]
                gt = imgs_model[:, 0, ...].to(device)  # (B,3,H,W)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    avg_loss, _ = photometric_backward_per_view(
                        gaussians, pred_extr, pred_ixt, gt, (H, W),
                        chunk_size=1, save_render_for_debug=False
                    )

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                loss_val = avg_loss

            else:
                pred = out["depth"].squeeze(1)
                target = depths.to(device)
                if pred.shape != target.shape:
                    pred = torch.nn.functional.interpolate(
                        pred.unsqueeze(1), size=target.shape[-2:], mode="bilinear", align_corners=False
                    ).squeeze(1)

                loss = loss_fn(pred, target)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    loss_val = float(loss.detach().item())
                else:
                    loss.backward()
                    optimizer.step()
                    loss_val = float(loss.detach().item())

            total_loss += loss_val
            n_steps += 1

            if i % 10 == 0:
                print(f"Epoch {epoch} iter {i} loss {loss_val:.4f}")

        avg = total_loss / max(1, n_steps)
        print(f"Epoch {epoch} avg loss {avg:.4f}")
        ckpt = os.path.join(args.out_dir, f"finetune_epoch{epoch}.pth")
        torch.save({"epoch": epoch, "model_state": net.state_dict(), "optimizer": optimizer.state_dict()}, ckpt)


if __name__ == "__main__":
    main()
