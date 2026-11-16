#!/usr/bin/env python3
"""
Study Experiment and Architecture Training Harness
==================================================
Paper: "Learnable Gegenbauer Activations and Direct Coordinate Regression for Lightweight Lesion Detection"
Authors: M. Deepthi, G. N. V. R. Vikram (2026)

This script provides the multi-model architecture and loss training harness supporting:
1. Proposed Detector (Cubic Gegenbauer + 4-channel Direct Continuous Head + Log-Cosh Loss)
2. Baseline YOLOv8n (SiLU + 64-channel DFL + CIoU)
3. Polynomial Basis Variants: Monomial, Chebyshev, Legendre, Gegenbauer
4. Localization Losses: Log-Cosh, MSE, MAE, Smooth L1, CIoU, EIoU
5. Degradation Conditions: Clean, Attenuated (Canonical RF Bias + Noise), N4ITK, CLAHE
6. Strict Patient-Level Splitting (876 train / 188 val / 187 test volumes)
"""

import os
import sys
import math
import random
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from models.yolov8_gegenbauer import GegenbauerYOLOv8, BaselineYOLOv8, verify_parameter_accounting
    from models.gegenbauer_activation import GegenbauerActivation
    from losses.logcosh_loss import LogCoshLoss, TotalLesionDetectionLoss
    from data.bias_field_simulation import generate_canonical_bias_field, apply_bias_field_and_rician_noise
    from data.mri_dataset import BraTSPatientLevelDataset
    from experiments.train_gegenbauer_detector import TaskAlignedAssigner, apply_mri_augmentations, nms_boxes, evaluate_cohort_detections
except (ImportError, ValueError) as err:
    print(f"[!] Warning: module import note: {err}")


def set_seed(seed: int = 101):
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True


def run_study_experiment(model_type: str = "proposed",
                         loss_type: str = "logcosh",
                         head_type: str = "4ch",
                         degradation: str = "attenuated",
                         bmin: float = 0.354,
                         epochs: int = 100,
                         seed: int = 101,
                         batch_size: int = 16,
                         lr0: float = 1e-3,
                         data_root: str = "",
                         save_dir: str = "",
                         max_train_samples: int = None,
                         max_val_samples: int = None):
    print("=" * 85)
    print("GEGENBAUER-YOLOV8 EXPERIMENTAL PIPELINE HARNESS")
    print("=" * 85)
    print(f"[*] Model Architecture:  {model_type.upper()} ({head_type} Head)")
    print(f"[*] Localization Loss:   {loss_type.upper()}")
    print(f"[*] Degradation Mode:    {degradation.upper()} (B_min={bmin:.3f})")
    print(f"[*] Training Epochs:     {epochs}")
    print(f"[*] Random Seed:         {seed}")
    print(f"[*] Learning Rate:       Initial={lr0:.1e}, 3-epoch linear warm-up, Cosine Annealing to 1e-5")
    print(f"[*] PyTorch Status:      {HAS_TORCH}")
    print("-" * 85)

    set_seed(seed)

    if not save_dir:
        save_dir = f"runs/study/{model_type}_{loss_type}_{head_type}_s{seed}"
    os.makedirs(save_dir, exist_ok=True)

    if not HAS_TORCH:
        print("[!] PyTorch is not installed in the current environment.")
        print("[*] To execute live neural network training, backpropagation, and weight updates:")
        print("      pip install torch torchvision")
        print("\n[*] Note on Manuscript Tables 1 - 6:")
        print("    Full retraining of all 155 master study runs (5 seeds x 31 configurations)")
        print("    on the full 100 GB BraTS cohort requires ~1,800 GPU hours on an NVIDIA RTX 3090 cluster.")
        print("    To evaluate the study's compiled per-seed benchmark registry and reproduce all")
        print("    tables, confidence intervals, paired t-tests, and ANCOVA models:")
        print("      python3 experiments/evaluate_all_tables.py")
        print("=" * 85)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device:     {device}")

    # 1. Instantiate the exact requested architecture variant
    if model_type.lower() == "baseline":
        model = BaselineYOLOv8(num_classes=1, head_type=head_type).to(device)
    else:
        # Proposed, Monomial, Chebyshev, or Legendre polynomial activations
        basis = "gegenbauer" if model_type.lower() in ["proposed", "gegenbauer"] else model_type.lower()
        model = GegenbauerYOLOv8(num_classes=1, basis_type=basis, head_type=head_type).to(device)

    # 2. Instantiate the exact requested loss function
    loss_fn = TotalLesionDetectionLoss(lambda_cls=0.5, lambda_loc=7.5, loss_type=loss_type)
    assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)

    warmup_epochs = 3
    lr_min = 1e-5

    def get_lr_factor(epoch):
        if epoch < warmup_epochs:
            # 3-epoch linear warm-up from lr_min to lr0 (Section 4.1)
            alpha = (epoch + 1) / float(warmup_epochs)
            return (lr_min / lr0) + (1.0 - lr_min / lr0) * alpha
        else:
            # Cosine annealing from lr0 to lr_min over remaining epochs
            progress = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return (lr_min / lr0) + 0.5 * (1.0 - lr_min / lr0) * (1.0 + math.cos(math.pi * progress))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_factor)

    print(f"[*] Model Parameters:    {sum(p.numel() for p in model.parameters()):,}")

    # 3. Load dataset
    if data_root and os.path.isdir(data_root):
        print(f"[*] Loading patient volumes from: {data_root}")
        train_ds = BraTSPatientLevelDataset(data_root=data_root, split="train", run_seed=seed)
        val_ds = BraTSPatientLevelDataset(data_root=data_root, split="val", run_seed=seed + 1000)
    else:
        print("[!] SYNTHETIC STAND-IN DATA IN USE - metrics from this run are NOT BraTS results (pipeline verification only).")
        train_ds = BraTSPatientLevelDataset(synthetic_mode=True, split="train", run_seed=seed)
        val_ds = BraTSPatientLevelDataset(synthetic_mode=True, split="val", run_seed=seed + 1000)

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(save_dir, "best.pt")
    last_ckpt_path = os.path.join(save_dir, "last.pt")

    # Default to full dataset splits; support optional capping flags with loud warning
    if max_train_samples is not None and max_train_samples > 0:
        n_train_samples = min(len(train_ds), int(max_train_samples))
        print(f"\n[!] WARNING: Active --max-train-samples={n_train_samples} cap! "
              f"Training on a limited subset of {n_train_samples}/{len(train_ds)} slices (SMOKE-TEST MODE).")
    else:
        n_train_samples = len(train_ds)
        print(f"[*] Full Training Split: {n_train_samples:,} slices across patient volumes.")

    if max_val_samples is not None and max_val_samples > 0:
        n_val_samples = min(len(val_ds), int(max_val_samples))
        print(f"[!] WARNING: Active --max-val-samples={n_val_samples} cap! "
              f"Validation on a limited subset of {n_val_samples}/{len(val_ds)} slices (SMOKE-TEST MODE).")
    else:
        n_val_samples = len(val_ds)
        print(f"[*] Full Validation Split: {n_val_samples:,} slices across patient volumes.")

    print(f"\n---> Commencing Training ({epochs} epochs, batch size {batch_size}):")
    for ep in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        train_epoch_losses = []

        # Fresh random permutation each epoch across the active training set
        epoch_perm = np.random.permutation(n_train_samples)

        for start_idx in range(0, n_train_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_train_samples)
            batch_indices = epoch_perm[start_idx:end_idx]
            cur_bs = len(batch_indices)
            if cur_bs <= 0:
                continue

            batch_imgs = []
            batch_boxes = []
            for idx in batch_indices:
                img_i, box_i, _ = train_ds[int(idx)]
                if isinstance(img_i, torch.Tensor):
                    img_i = img_i.cpu().numpy()
                if isinstance(box_i, torch.Tensor):
                    box_i = box_i.cpu().numpy()
                batch_imgs.append(img_i)
                batch_boxes.append(box_i)

            batch_imgs_np = np.stack(batch_imgs, axis=0)
            batch_boxes_np = np.stack(batch_boxes, axis=0)

            # Spatial augmentation
            aug_imgs, aug_boxes = apply_mri_augmentations(batch_imgs_np, batch_boxes_np)

            # Optional RF bias attenuation simulation
            if degradation == "attenuated":
                bias_f = generate_canonical_bias_field(aug_imgs.shape[2], aug_imgs.shape[3])
                for b_i in range(cur_bs):
                    for c_i in range(aug_imgs.shape[1]):
                        aug_imgs[b_i, c_i] = apply_bias_field_and_rician_noise(
                            aug_imgs[b_i, c_i], bias_field=bias_f, add_rician=True, seed=seed + ep * 100 + b_i
                        )

            img_tensor = torch.from_numpy(aug_imgs).float().to(device)
            optimizer.zero_grad()
            preds = model(img_tensor)

            pred_boxes = preds["boxes"]
            pred_scores = preds["scores"]

            batch_loss = torch.tensor(0.0, device=device)
            for b_i in range(cur_bs):
                pb_np = pred_boxes[b_i].detach().cpu().numpy()
                ps_np = pred_scores[b_i, :, 0].detach().cpu().numpy()
                gt_np = aug_boxes[b_i]

                is_pos, assigned_gt, weights = assigner.assign(pb_np, ps_np, gt_np)
                pos_mask_t = torch.from_numpy(is_pos).to(device)
                weights_t = torch.from_numpy(weights).float().to(device)
                target_scores_t = torch.zeros_like(pred_scores[b_i])
                target_scores_t[pos_mask_t, 0] = weights_t[pos_mask_t]

                if pos_mask_t.sum() > 0:
                    matched_gt_np = gt_np[assigned_gt[is_pos]]
                    target_boxes_t = torch.from_numpy(matched_gt_np).float().to(device)
                    loc_loss_b = loss_fn.loc_loss_fn(pred_boxes[b_i][pos_mask_t], target_boxes_t, weights=weights_t[pos_mask_t])
                else:
                    loc_loss_b = torch.tensor(0.0, device=device)

                cls_loss_b = nn.functional.binary_cross_entropy(pred_scores[b_i].clamp(1e-6, 1.0 - 1e-6), target_scores_t)
                batch_loss = batch_loss + (0.5 * cls_loss_b + 7.5 * loc_loss_b)

            batch_loss = batch_loss / float(cur_bs)
            batch_loss.backward()
            optimizer.step()
            train_epoch_losses.append(batch_loss.item())

        mean_train_loss = float(np.mean(train_epoch_losses)) if train_epoch_losses else 0.0

        # Authentic Validation Loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for v_idx in range(n_val_samples):
                v_img, v_box, _ = val_ds[v_idx]
                v_tensor = torch.from_numpy(v_img if isinstance(v_img, np.ndarray) else v_img.numpy()).float().unsqueeze(0).to(device)
                if v_tensor.shape[1] == 1:
                    v_tensor = v_tensor.repeat(1, 3, 1, 1)

                v_out = model(v_tensor)
                v_pb = v_out["boxes"][0].cpu().numpy()
                v_ps = v_out["scores"][0, :, 0].cpu().numpy()
                gt_v = v_box.numpy() if isinstance(v_box, torch.Tensor) else v_box

                is_pos, assigned_gt, weights = assigner.assign(v_pb, v_ps, gt_v)
                pos_mask_v = torch.from_numpy(is_pos).to(device)
                weights_v = torch.from_numpy(weights).float().to(device)
                target_scores_v = torch.zeros_like(v_out["scores"][0])
                target_scores_v[pos_mask_v, 0] = weights_v[pos_mask_v]

                if pos_mask_v.sum() > 0:
                    matched_gt_v = torch.from_numpy(gt_v[assigned_gt[is_pos]]).float().to(device)
                    v_loc = loss_fn.loc_loss_fn(v_out["boxes"][0][pos_mask_v], matched_gt_v, weights=weights_v[pos_mask_v])
                else:
                    v_loc = torch.tensor(0.0, device=device)

                v_cls = nn.functional.binary_cross_entropy(v_out["scores"][0].clamp(1e-6, 1.0 - 1e-6), target_scores_v)
                val_losses.append((0.5 * v_cls + 7.5 * v_loc).item())

        mean_val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        # Checkpoint Selection
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": mean_val_loss,
                "model_type": model_type,
                "loss_type": loss_type,
                "head_type": head_type,
                "seed": seed
            }, best_ckpt_path)
            best_tag = f"[Saved Best: {best_ckpt_path}]"
        else:
            best_tag = ""

        print(f"  Epoch [{ep:2d}/{epochs:2d}] | lr: {current_lr:.2e} | Train Loss: {mean_train_loss:.4f} | "
              f"Val Loss: {mean_val_loss:.4f} {best_tag}")
        scheduler.step()

    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": mean_val_loss,
        "model_type": model_type,
        "loss_type": loss_type,
        "head_type": head_type,
        "seed": seed
    }, last_ckpt_path)

    print(f"[+] Architecture '{model_type}' ({head_type} head, {loss_type} loss) trained successfully.")
    print(f"    Best checkpoint: {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")
    print("=" * 85)


def run_ablation_study(ablation_type: str = "activations", epochs: int = 100, seed: int = 101, data_root: str = "",
                       max_train_samples: int = None, max_val_samples: int = None):
    print("=" * 85)
    print(f"RUNNING ARCHITECTURAL ABLATION SWEEP: {ablation_type.upper()}")
    print("=" * 85)

    if not HAS_TORCH:
        print("[!] PyTorch is required to execute multi-epoch backpropagation on individual ablation variants.")
        print("[*] To evaluate the published ablation benchmark results across all 5 seeds:")
        print("      python3 experiments/evaluate_all_tables.py")
        print("=" * 85)
        return

    if ablation_type == "activations":
        # All polynomial basis variants from Table 4:
        models = ["monomial", "chebyshev", "legendre", "baseline", "proposed"]
        for m in models:
            head = "64ch" if m != "proposed" else "4ch"
            print(f"\n---> Training Activation Variant: {m.upper()} ({head} head)...")
            run_study_experiment(model_type=m, head_type=head, epochs=epochs, seed=seed, data_root=data_root,
                                 max_train_samples=max_train_samples, max_val_samples=max_val_samples)

    elif ablation_type == "losses":
        # All loss function variants from Table 6:
        losses = ["mae", "ciou", "smooth_l1", "mse", "logcosh"]
        for l in losses:
            print(f"\n---> Training Loss Variant: {l.upper()}...")
            run_study_experiment(model_type="proposed", loss_type=l, head_type="4ch", epochs=epochs, seed=seed, data_root=data_root,
                                 max_train_samples=max_train_samples, max_val_samples=max_val_samples)

    elif ablation_type == "heads":
        # Table 5 2x2 Factorial of Activation x Head/Loss
        combos = [
            ("baseline", "64ch", "ciou"),
            ("baseline", "4ch", "logcosh"),
            ("proposed", "64ch", "ciou"),
            ("proposed", "4ch", "logcosh")
        ]
        for m, h, l in combos:
            print(f"\n---> Training 2x2 Factorial Combo: {m.upper()} + {h} Head ({l})...")
            run_study_experiment(model_type=m, head_type=h, loss_type=l, epochs=epochs, seed=seed, data_root=data_root,
                                 max_train_samples=max_train_samples, max_val_samples=max_val_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Study Experiment and Architecture Training Harness")
    parser.add_argument("--model", type=str, default="proposed", choices=["proposed", "baseline", "monomial", "chebyshev", "legendre"])
    parser.add_argument("--loss", type=str, default="logcosh", choices=["logcosh", "mse", "mae", "smooth_l1", "ciou"])
    parser.add_argument("--head", type=str, default="4ch", choices=["4ch", "64ch"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-root", type=str, default="")
    parser.add_argument("--ablation", type=str, default="")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap training slices per epoch (for rapid smoke tests; default: all)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Cap validation slices per epoch (for rapid smoke tests; default: all)")
    parser.add_argument("--synthetic", action="store_true", help="Explicitly allow the synthetic stand-in dataset when --data-root is not given (smoke tests only; results are NOT BraTS results)")
    args = parser.parse_args()

    # Guard: never start a full-length run on the synthetic stand-in by accident.
    if args.data_root and not os.path.isdir(args.data_root):
        parser.error(f"--data-root '{args.data_root}' is not a directory (refusing to fall back to synthetic data).")
    if not args.data_root and not args.synthetic and args.max_train_samples is None:
        parser.error("no --data-root given. Pass --data-root /path/to/BraTS2021 for a real run, or --synthetic "
                     "(preferably with --max-train-samples/--max-val-samples) for a smoke test on synthetic stand-in data.")

    if args.ablation:
        run_ablation_study(
            ablation_type=args.ablation,
            epochs=args.epochs,
            seed=args.seed,
            data_root=args.data_root,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples
        )
    else:
        run_study_experiment(
            model_type=args.model,
            loss_type=args.loss,
            head_type=args.head,
            epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            data_root=args.data_root,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples
        )
