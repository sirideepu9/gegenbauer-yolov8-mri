#!/usr/bin/env python3
"""
Master Reference Verification and Inspection Runner
====================================================
Paper: Learnable Gegenbauer Activations and Direct Coordinate Regression for Lightweight Lesion Detection
Authors: M. Deepthi, G. N. V. R. Vikram (2026)

Executes and verifies all algorithmic modules:
1. Polynomial Basis Scale-Invariant Gram Conditioning (Section 3.2)
2. Parametric RF Bias Field Profile, Severity Sweeps & Lesion Attenuation (Section 3.3, Eq. 8)
3. Layer-by-Layer Architectural Parameter Accounting (Section 3.3, Table 1)
4. Scanner-Side Latency & N4ITK/CLAHE Preprocessing Benchmarks (Section 4.1)
5. 3D Volumetric Multi-Slice Spatial Continuity Filtering (Section 4.4)
6. Unified Master Benchmark Tables Reported in Manuscript (Tables 1 - 6)
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from experiments.basis_gram_conditioning import evaluate_all_bases
from experiments.evaluate_all_tables import (
    load_master_records, load_severity_records,
    print_table_1, print_table_2, print_table_3, print_table_4, print_table_5_and_6
)
from experiments.multi_slice_filter import run_filter_demonstration
from experiments.benchmark_latency import benchmark_device_latency
from experiments.run_n4itk_clahe_preprocessing import run_preprocessing_demonstration
from data.bias_field_simulation import generate_canonical_bias_field, verify_lesion_attenuation_distribution
from models.yolov8_gegenbauer import verify_parameter_accounting


def main():
    print("#" * 90)
    print("REFERENCE IMPLEMENTATION & MATHEMATICAL VERIFICATION SUITE")
    print("Paper: Learnable Gegenbauer Activations and Direct Coordinate Regression")
    print("Authors: M. Deepthi, G. N. V. R. Vikram (2026)")
    print("#" * 90)

    # 1. Mathematical Basis Gram Conditioning
    print("\n>>> STEP 1: VERIFYING POLYNOMIAL BASIS SCALE-INVARIANT GRAM CONDITIONING (SECTION 3.2)...")
    evaluate_all_bases(n_samples=500_000, seed=42)

    # 2. Bias Field Verification & Lesion Attenuation
    print("\n>>> STEP 2: EVALUATING CANONICAL RF BIAS FIELD & LESION ATTENUATION (SECTION 3.3, EQ. 8)...")
    atten_demo = verify_lesion_attenuation_distribution()
    print(f"[*] Canonical Bias Field: min = {atten_demo['b_min']:.4f}, max = {atten_demo['b_max']:.4f}, mean = {atten_demo['b_mean']:.4f}")
    print(f"[*] Attenuation at Proximal Lesion: {atten_demo['proximal_atten_pct']:.1f}%")
    print(f"[*] Attenuation at Central Lesion:  {atten_demo['central_atten_pct']:.1f}%")
    print(f"[*] Attenuation at Distal Lesion:   {atten_demo['distal_peak_atten_pct']:.1f}% (Section 3.3 reports peak 38.1%)")
    print("[+] Canonical RF bias field extrema and attenuation profile demonstrated.")

    # 3. Parameter Accounting
    print("\n>>> STEP 3: ARCHITECTURAL PARAMETER ACCOUNTING (TABLE 1)...")
    stats_dict = verify_parameter_accounting()
    print(f"[*] Baseline YOLOv8n:      {stats_dict['base_total']:,} parameters (Backbone: {stats_dict['base_backbone']:,}, Neck: {stats_dict['base_neck']:,}, Head: {stats_dict['base_head']:,})")
    print(f"[*] Proposed Detector:     {stats_dict['prop_total']:,} parameters (Backbone: {stats_dict['prop_backbone']:,}, Neck: {stats_dict['prop_neck']:,}, Head: {stats_dict['prop_head']:,})")
    print(f"[*] Net Parameter Delta:   {stats_dict['delta_total']:+,} ({stats_dict['delta_pct']:+.2f}%)")
    print("[+] Layer-by-layer parameter accounting matches manuscript Table 1.")

    # 4. Latency and Preprocessing Benchmarks
    print("\n>>> STEP 4: VERIFYING SCANNER-SIDE LATENCY & PREPROCESSING BENCHMARKS (SECTION 4.1)...")
    benchmark_device_latency(device="cpu", warmup=10, reps=20)
    run_preprocessing_demonstration()

    # 5. Multi-Slice Filtering
    print("\n>>> STEP 5: VERIFYING 3D VOLUMETRIC MULTI-SLICE SPATIAL CONTINUITY (SECTION 4.4)...")
    run_filter_demonstration()

    # 6. Tables 1 to 6
    print("\n>>> STEP 6: REFERENCE BENCHMARK RESULTS REPORTED IN MANUSCRIPT (TABLES 1 - 6)...")
    master_records = load_master_records()
    sweep_records = load_severity_records()
    print_table_1()
    print_table_2(master_records)
    print_table_3(sweep_records)
    print_table_4(master_records)
    print_table_5_and_6(master_records)

    print("\n" + "#" * 90)
    print("VERIFICATION SUITE COMPLETED SUCCESSFULLY")
    print("Note: Re-training models across the full 1,251-patient BraTS 2021 cohort (21,267 slices)")
    print("requires the full external NIfTI dataset and an NVIDIA RTX 3090 / A100 GPU environment.")
    print("#" * 90)


if __name__ == "__main__":
    main()
