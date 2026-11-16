#!/usr/bin/env python3
"""
Checkpoint Evaluation & Metric Export to CSV
===========================================
Paper: "Learnable Gegenbauer Activations and Direct Coordinate Regression for Lightweight Lesion Detection"
Authors: M. Deepthi, G. N. V. R. Vikram (2026)

This script provides the evaluation pipeline that takes trained model checkpoints
and evaluates them across test cohort slices under clean, attenuated, or severity-sweep
conditions, exporting per-seed benchmark rows matching:
  - 'experiments/per_seed_results.csv' (Standard 155-run benchmark schema)
  - 'experiments/severity_sweep_per_seed.csv' (Table 3 Parametric Severity Sweep schema)

Capabilities:
  1. Multi-architecture evaluation: Proposed Gegenbauer YOLOv8n or Baseline YOLOv8n (--arch / --model).
  2. Parametric severity attenuation: Evaluates at any minimum bias gain B_min in [0.35, 1.00] (--b-min).
  3. Parametric severity sweep: Evaluates checkpoints across all 5 Table 3 severity levels (--sweep).
  4. Full cohort evaluation: Evaluates all 3,180 slices across the 187 test volumes by default (--max-slices all).
  5. Strict evaluation standards:
     - Non-Maximum Suppression (NMS) per image.
     - Strictly per-image matching.
     - 101-point COCO AP metric computation.
     - Zero synthetic fallbacks (exits with FileNotFoundError if --data-root is missing).
     - No fake zeros on partial sweeps (refuses export or writes blanks).
     - Live measured inference latency timing without hardcoded fallbacks.
"""

import os
import sys
import csv
import time
import argparse
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from models.yolov8_gegenbauer import GegenbauerYOLOv8, BaselineYOLOv8
    from experiments.train_gegenbauer_detector import nms_boxes, evaluate_cohort_detections, box_iou_matrix
    from data.mri_dataset import BraTSPatientLevelDataset
    from data.bias_field_simulation import (
        generate_canonical_bias_field,
        generate_severity_sweep_bias_field,
        apply_bias_field_and_rician_noise
    )
except (ImportError, ValueError) as err:
    print(f"[!] Warning: module import note: {err}")


SWEEP_LEVELS = [
    ("Clean (unattenuated)", 1.00, "0%", "0.0%"),
    ("Mild attenuation", 0.85, "15%", "4.2%"),
    ("Moderate attenuation", 0.70, "30%", "8.9%"),
    ("Strong attenuation", 0.50, "50%", "13.5%"),
    ("Severe stress", 0.35, "65%", "18.4%")
]


def parse_max_slices(val: Optional[str], dataset_len: int) -> int:
    """Parses --max-slices argument into an integer count."""
    if val is None or str(val).strip().lower() in ("all", "none", "-1"):
        return dataset_len
    try:
        n = int(val)
        if n <= 0:
            return dataset_len
        return min(n, dataset_len)
    except ValueError:
        raise ValueError(f"Invalid --max-slices value '{val}'. Must be an integer or 'all'.")


def load_model_from_checkpoint(checkpoint_path: str,
                               arch: str = "auto",
                               model_name: str = "Proposed Model",
                               device: str = "cpu"):
    """Loads weights from checkpoint into Baseline or Proposed YOLOv8 architecture."""
    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch is not installed in the current environment. "
            "Evaluating a neural network checkpoint requires PyTorch (torch >= 2.0)."
        )

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: '{checkpoint_path}'. "
            f"Please train a detector first with 'experiments/train_gegenbauer_detector.py'."
        )

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    # Select architecture
    if arch == "baseline" or (arch == "auto" and "baseline" in model_name.lower()):
        model = BaselineYOLOv8(num_classes=1).to(dev)
        arch_type = "Baseline YOLOv8n"
    else:
        model = GegenbauerYOLOv8(num_classes=1).to(dev)
        arch_type = "Proposed Gegenbauer YOLOv8n"

    state = torch.load(checkpoint_path, map_location=dev)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Loaded {arch_type} from '{checkpoint_path}' ({n_params:,} parameters) onto {dev}.")
    return model, dev


def run_cohort_inference(model,
                         dataset,
                         dev,
                         bias_field: Optional[np.ndarray] = None,
                         n_eval: int = 50,
                         add_rician: bool = False,
                         seed: int = 101) -> Tuple[Dict, List[float]]:
    """Runs inference across slices, applies NMS per image, and computes COCO AP metrics."""
    preds_per_img = []
    gts_per_img = []
    latencies = []

    with torch.no_grad():
        for idx in range(n_eval):
            img_np, box_np, _ = dataset[idx]
            gt_box = box_np if isinstance(box_np, np.ndarray) else box_np.numpy()

            # Apply bias field if requested
            if bias_field is not None:
                if img_np.ndim == 3:
                    img_hwc = np.transpose(img_np, (1, 2, 0))
                    atten_hwc = apply_bias_field_and_rician_noise(img_hwc, bias_field=bias_field,
                                                                 add_rician=add_rician, seed=seed + idx)
                    proc_np = np.transpose(atten_hwc, (2, 0, 1))
                else:
                    proc_np = apply_bias_field_and_rician_noise(img_np, bias_field=bias_field,
                                                                add_rician=add_rician, seed=seed + idx)
            else:
                proc_np = img_np

            img_t = torch.from_numpy(proc_np).float().unsqueeze(0).to(dev)
            if img_t.shape[1] == 1:
                img_t = img_t.repeat(1, 3, 1, 1)

            # Measure pure forward pass timing (excluding dataset loading, numpy conversion and NMS)
            if dev.type == "cuda":
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds = model(img_t)
                torch.cuda.synchronize()
                t_elapsed = (time.perf_counter() - t0) * 1000.0
            else:
                t0 = time.perf_counter()
                preds = model(img_t)
                t_elapsed = (time.perf_counter() - t0) * 1000.0

            latencies.append(t_elapsed)

            boxes = preds["boxes"][0].cpu().numpy()
            scores = preds["scores"][0, :, 0].cpu().numpy()

            # Score filtering + Non-Maximum Suppression (NMS)
            conf_mask = scores >= 0.05
            filt_b = boxes[conf_mask]
            filt_s = scores[conf_mask]
            nms_b, nms_s = nms_boxes(filt_b, filt_s, iou_thresh=0.50, max_det=100)

            preds_per_img.append((nms_b, nms_s))
            gts_per_img.append(gt_box)

    metrics = evaluate_cohort_detections(preds_per_img, gts_per_img)
    return metrics, latencies


def evaluate_model_on_test_cohort(model_name: str = "Proposed Model",
                                  arch: str = "auto",
                                  seed: int = 101,
                                  checkpoint_path: str = "",
                                  data_root: str = "",
                                  b_min: float = 0.35,
                                  add_rician: bool = False,
                                  device: str = "cpu",
                                  max_slices_arg: Optional[str] = "all") -> Dict:
    """Evaluates checkpoint across clean and attenuated cohorts for per_seed_results schema."""
    if not data_root or not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"A valid BraTS dataset directory must be provided via --data-root to evaluate model checkpoints "
            f"on the authentic test cohort. Specified path '{data_root}' was not found or is invalid. "
            f"Synthetic fallback is strictly disabled for benchmark evaluation."
        )

    model, dev = load_model_from_checkpoint(checkpoint_path, arch=arch, model_name=model_name, device=device)

    print(f"[*] Loading test cohort from: {data_root}")
    test_ds = BraTSPatientLevelDataset(data_root=data_root, split="test", run_seed=seed + 2000)

    n_eval = parse_max_slices(max_slices_arg, len(test_ds))

    print("=" * 85)
    print(f"[*] EVALUATING CHECKPOINT: {checkpoint_path}")
    print(f"[*] Model: {model_name} | Arch: {arch} | Seed: {seed} | B_min: {b_min:.2f}")
    print(f"[*] Evaluation Slices:    {n_eval} / {len(test_ds)} total test slices")
    print(f"[*] Execution Hardware:   {dev}")
    print("=" * 85)

    # 1. Clean Inference
    print(f"[*] Evaluating Clean Cohort ({n_eval} slices)...")
    clean_metrics, clean_lats = run_cohort_inference(model, test_ds, dev, bias_field=None,
                                                     n_eval=n_eval, add_rician=False, seed=seed)

    # 2. Attenuated Inference
    print(f"[*] Evaluating Attenuated Cohort (B_min = {b_min:.2f}, Rician = {add_rician})...")
    atten_field = generate_severity_sweep_bias_field(height=640, width=640, b_min=b_min)
    atten_metrics, atten_lats = run_cohort_inference(model, test_ds, dev, bias_field=atten_field,
                                                     n_eval=n_eval, add_rician=add_rician, seed=seed)

    clean_map50 = round(float(clean_metrics["map50"]), 4)
    clean_map75 = round(float(clean_metrics["map75"]), 4)
    clean_map50_95 = round(float(clean_metrics["map50_95"]), 4)

    atten_map50 = round(float(atten_metrics["map50"]), 4)
    atten_map75 = round(float(atten_metrics["map75"]), 4)
    atten_map50_95 = round(float(atten_metrics["map50_95"]), 4)

    delta_drop = round(float(atten_map50 - clean_map50), 4)
    median_lat = round(float(np.median(atten_lats)), 2) if len(atten_lats) > 0 else float("nan")

    row = {
        "model": model_name,
        "seed": seed,
        "precision": round(float(atten_metrics["precision"]), 4),
        "recall": round(float(atten_metrics["recall"]), 4),
        "map50": atten_map50,
        "map75": atten_map75,
        "map50_95": atten_map50_95,
        "clean_map50": clean_map50,
        "clean_map75": clean_map75,
        "clean_map50_95": clean_map50_95,
        "delta_drop": delta_drop,
        "latency_ms": median_lat
    }
    return row


def evaluate_severity_sweep(proposed_ckpt: str = "",
                            baseline_ckpt: str = "",
                            seed: int = 101,
                            data_root: str = "",
                            device: str = "cpu",
                            max_slices_arg: Optional[str] = "all",
                            require_both_for_export: bool = False) -> List[Dict]:
    """
    Evaluates checkpoints across all 5 Table 3 severity levels:
    B_min in {1.00 (Clean), 0.85 (Mild), 0.70 (Moderate), 0.50 (Strong), 0.35 (Severe)}.
    Returns list of rows matching 'severity_sweep_per_seed.csv' schema.
    """
    if not data_root or not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"A valid BraTS dataset directory must be provided via --data-root to evaluate severity sweeps. "
            f"Specified path '{data_root}' was not found or is invalid. "
            f"Synthetic fallback is strictly disabled for benchmark evaluation."
        )

    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to evaluate severity sweep checkpoints.")

    test_ds = BraTSPatientLevelDataset(data_root=data_root, split="test", run_seed=seed + 2000)
    n_eval = parse_max_slices(max_slices_arg, len(test_ds))

    prop_model, base_model = None, None
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    if proposed_ckpt and os.path.isfile(proposed_ckpt):
        prop_model, _ = load_model_from_checkpoint(proposed_ckpt, arch="proposed", device=device)
    if baseline_ckpt and os.path.isfile(baseline_ckpt):
        base_model, _ = load_model_from_checkpoint(baseline_ckpt, arch="baseline", device=device)

    if prop_model is None and base_model is None:
        raise ValueError("At least one valid checkpoint (--checkpoint, --proposed-checkpoint, or --baseline-checkpoint) must be supplied.")

    if require_both_for_export and (prop_model is None or base_model is None):
        raise ValueError(
            "Exporting comparative rows to --sweep-csv requires BOTH --checkpoint (proposed) and "
            "--baseline-checkpoint (baseline). Partial export refused to prevent registry corruption."
        )

    print("\n" + "=" * 95)
    print("PARAMETRIC ATTENUATION SEVERITY SWEEP EVALUATION (TABLE 3)")
    print(f"Proposed Checkpoint: {proposed_ckpt or '(None)'}")
    print(f"Baseline Checkpoint: {baseline_ckpt or '(None)'}")
    print(f"Evaluation Slices:   {n_eval} / {len(test_ds)} total test slices")
    print(f"Evaluation Seed:     {seed}")
    print("=" * 95)

    sweep_rows = []
    print(f"\n{'Severity Level':<24} {'B_min':<8} {'Canvas Falloff':<16} {'Mean Target Atten':<20} {'Baseline':<12} {'Proposed':<12} {'Margin':<12}")
    print("-" * 105)

    for name, bmin, falloff, atten in SWEEP_LEVELS:
        field = generate_severity_sweep_bias_field(height=640, width=640, b_min=bmin) if bmin < 0.999 else None

        base_map50: Optional[float] = None
        prop_map50: Optional[float] = None

        if base_model is not None:
            b_metrics, _ = run_cohort_inference(base_model, test_ds, dev, bias_field=field, n_eval=n_eval, seed=seed)
            base_map50 = round(float(b_metrics["map50"]), 4)

        if prop_model is not None:
            p_metrics, _ = run_cohort_inference(prop_model, test_ds, dev, bias_field=field, n_eval=n_eval, seed=seed)
            prop_map50 = round(float(p_metrics["map50"]), 4)

        margin: Optional[float] = round(float(prop_map50 - base_map50), 4) if (base_map50 is not None and prop_map50 is not None) else None

        b_str = f"{base_map50:.3f}" if base_map50 is not None else "---"
        p_str = f"{prop_map50:.3f}" if prop_map50 is not None else "---"
        m_str = f"{margin:+.3f}" if margin is not None else "---"

        print(f"{name:<24} {bmin:<8.2f} {falloff:<16} {atten:<20} {b_str:<12} {p_str:<12} {m_str:<12}")

        sweep_rows.append({
            "severity_level": name,
            "b_min": f"{bmin:.2f}",
            "canvas_falloff": falloff,
            "mean_target_atten": atten,
            "seed": seed,
            "baseline_map50": f"{base_map50:.4f}" if base_map50 is not None else "",
            "proposed_map50": f"{prop_map50:.4f}" if prop_map50 is not None else "",
            "margin": f"{margin:+.4f}" if margin is not None else ""
        })

    print("-" * 105)
    return sweep_rows


def export_row_to_csv(row: dict, output_csv: str):
    """Appends an evaluated run row to target CSV file matching per_seed_results schema."""
    file_exists = os.path.exists(output_csv)
    fieldnames = [
        "model", "seed", "precision", "recall", "map50", "map75",
        "map50_95", "clean_map50", "clean_map75", "clean_map50_95",
        "delta_drop", "latency_ms"
    ]
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[+] Exported benchmark row for '{row['model']}' (Seed {row['seed']}) to: {output_csv}")


def export_sweep_rows_to_csv(rows: List[dict], output_csv: str):
    """Appends severity sweep rows to target CSV file matching severity_sweep_per_seed schema."""
    file_exists = os.path.exists(output_csv)
    fieldnames = [
        "severity_level", "b_min", "canvas_falloff", "mean_target_atten",
        "seed", "baseline_map50", "proposed_map50", "margin"
    ]
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[+] Exported {len(rows)} severity sweep rows to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate checkpoint and export per-seed CSV rows")
    parser.add_argument("--checkpoint", type=str, default="", help="Model weights checkpoint (.pt)")
    parser.add_argument("--arch", type=str, choices=["auto", "proposed", "baseline"], default="auto",
                        help="Model architecture: 'proposed' (Gegenbauer) or 'baseline' (SiLU DFL)")
    parser.add_argument("--model", type=str, default="Proposed Model", help="Descriptive model name for CSV registry")
    parser.add_argument("--seed", type=int, default=101, help="Random evaluation seed")
    parser.add_argument("--b-min", type=float, default=0.35, help="Minimum bias gain B_min in [0.35, 1.00]")
    parser.add_argument("--add-rician", action="store_true", help="Add Rician sensor noise")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to BraTS 2021 dataset root directory (synthetic fallback disabled)")
    parser.add_argument("--device", type=str, default="cpu", help="Execution device: cpu or cuda")
    parser.add_argument("--max-slices", type=str, default="all",
                        help="Number of cohort slices to evaluate: an integer (e.g. 50) or 'all' (default: full test split)")
    parser.add_argument("--output-csv", type=str, default="", help="Path to export per-seed result CSV row")
    parser.add_argument("--sweep", action="store_true", help="Run Table 3 Parametric Severity Sweep across 5 levels")
    parser.add_argument("--baseline-checkpoint", type=str, default="", help="Optional Baseline checkpoint for sweep comparison")
    parser.add_argument("--sweep-csv", type=str, default="", help="Path to export severity sweep CSV rows")
    args = parser.parse_args()

    if args.sweep:
        proposed_ckpt = args.checkpoint if args.arch != "baseline" else ""
        baseline_ckpt = args.baseline_checkpoint or (args.checkpoint if args.arch == "baseline" else "")
        sweep_rows = evaluate_severity_sweep(
            proposed_ckpt=proposed_ckpt,
            baseline_ckpt=baseline_ckpt,
            seed=args.seed,
            data_root=args.data_root,
            device=args.device,
            max_slices_arg=args.max_slices,
            require_both_for_export=bool(args.sweep_csv)
        )
        if args.sweep_csv:
            export_sweep_rows_to_csv(sweep_rows, args.sweep_csv)
    else:
        row = evaluate_model_on_test_cohort(
            model_name=args.model,
            arch=args.arch,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            data_root=args.data_root,
            b_min=args.b_min,
            add_rician=args.add_rician,
            device=args.device,
            max_slices_arg=args.max_slices
        )

        print("\n" + "=" * 80)
        print("DERIVED BENCHMARK ROW (Matches 'per_seed_results.csv' Schema):")
        print("=" * 80)
        for k, v in row.items():
            print(f"  {k:<18}: {v}")
        print("=" * 80)

        if args.output_csv:
            export_row_to_csv(row, args.output_csv)
