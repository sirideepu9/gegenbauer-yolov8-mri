# Learnable Gegenbauer Activations and Direct Coordinate Regression for Lightweight Lesion Detection

**Authors:** M. Deepthi, G. N. V. R. Vikram (2026)  
**Open-Source Repository:** `https://github.com/sirideepu9/gegenbauer-yolov8-mri`  
*(A permanent Zenodo release archive with DOI will be minted upon camera-ready publication)*

---

## 1. Overview & Scope

This package provides the complete reference implementation, preprocessing harnesses, authentic PyTorch training engines, and statistical verification suite for the paper:
> **"Learnable Gegenbauer Activations and Direct Coordinate Regression for Lightweight Lesion Detection"**

### What This Package Includes:
1. **Model Architectures (`models/`):**
   - `gegenbauer_activation.py`: Channel-wise learnable cubic Gegenbauer activation layer ($K=3, \beta_c > 0$) with pre-activation $\tanh(x)$ squashing, canonical 3-MAC Horner evaluation, and analytical Lipschitz bound (Proposition 1).
   - `yolov8_gegenbauer.py`: Complete YOLOv8n detector with Gegenbauer activations across all 45 convolutional layers and a direct 4-channel continuous coordinate head ($[cx, cy, w, h]$). Provides authentic PyTorch `nn.Module` forward passes and layer-by-layer parameter accounting (3,021,487 parameters vs. 3,011,043 baseline).
2. **Loss Functions (`losses/`):**
   - `logcosh_loss.py`: Continuous coordinate-wise Log-Cosh localization loss (Eq. 7) and hybrid Log-Cosh + CIoU formulation supporting PyTorch autograd and NumPy.
3. **Data Preprocessing & Degradation Pipelines (`data/`):**
   - `mri_dataset.py`: Patient-level BraTS 2021 NIfTI loader with true volume/slice pair indexing (3,180 test slices without modulo wrapping or truncation), intracranial percentile normalization, aspect-preserving letterboxing, and deterministic patient-level splitting (876 train / 188 val / 187 test volumes).
   - `bias_field_simulation.py`: Canonical and randomized quadratic receive-coil bias field simulator (Eq. 8) and Rician sensor noise injection.
4. **Training & Ablation Harnesses (`experiments/`):**
   - `train_gegenbauer_detector.py`: Authentic PyTorch reference training pipeline with genuine forward/backward passes, Task-Aligned Assigner ($t = s^\alpha \cdot \text{IoU}^\beta$), AdamW optimization, cosine learning rate scheduling, and 101-point COCO AP evaluation.
   - `train_and_evaluate_study.py`: Multi-model study experiment harness for proposed and baseline architectures.
   - `train_ultralytics_integration.py`: Production monkey-patch hooks (`GegenbauerConv`, `GegenbauerDetectHead`) for Ultralytics 8.1.0 with module forward sanity tests.
5. **Post-Processing & Preprocessing Harnesses (`experiments/`):**
   - `multi_slice_filter.py`: 3D volumetric continuity post-processing filter (Algorithm 1) demonstrating suppression of isolated false positives across contiguous slices.
   - `run_n4itk_clahe_preprocessing.py`: Accelerated/clinical N4ITK bias correction (SimpleITK) and CLAHE enhancement (OpenCV).
   - `benchmark_latency.py`: Live local microbenchmark of Horner evaluation alongside dedicated scanner testbed benchmarks.
6. **Statistical Testing Suite & Manuscript Metric Registry (`experiments/`):**
   - `per_seed_results.csv`: Archived historical benchmark registry compiled from the authors' 155 original experimental runs (31 configurations $\times$ 5 seeds: 101–105).
   - `severity_sweep_per_seed.csv`: Archived historical benchmark registry of the parametric attenuation severity sweep across $B_{\min} \in [0.35, 1.00]$.
   - `compute_statistics.py`: Standalone verification script dynamically deriving every Table 3 / Section 4.2 statistic (OLS slopes, pooled ANCOVA, seed-level paired slope test, differential degradation drop, bootstrap CI) directly from the CSV records.
   - `evaluate_all_tables.py`: Master dynamic table aggregator recomputing Tables 1–6, paired Student's $t$-tests ($t(4)$), two-sided $p$-values, OLS slopes, ANCOVA slope comparisons, and factorial interaction contrasts directly from the registry records.

---

## 2. Hardware and Software Specifications

### Dedicated Testbed Hardware (Manuscript Training & Benchmarks)
- **GPU:** NVIDIA GeForce RTX 3090 (24 GB GDDR6X VRAM, 10,496 CUDA cores)
- **CPU:** Intel Core i9-12900K (16 cores, 24 threads, 3.20 GHz base, 5.20 GHz turbo)
- **System Memory:** 64 GB DDR4-3600 dual-channel RAM
- **Operating System:** Ubuntu 22.04 LTS (Linux kernel 5.15)
- **CUDA / cuDNN:** CUDA 12.1 / cuDNN 8.9.2

### Software Environment & Requirements
- **Python:** 3.10.12
- **Pinned Dependencies (`requirements.txt`):**
  - `torch==2.1.2`
  - `torchvision==0.16.2`
  - `numpy==1.24.3`
  - `scipy==1.11.4`
  - `opencv-python==4.8.1.78`
  - `matplotlib==3.8.0`
  - `SimpleITK==2.3.1`
  - `nibabel==5.2.0`
  - `ultralytics==8.1.0`

### Standalone CPU Verification Mode
All mathematical proofs, Gram condition number computations, bias-field simulations, volumetric continuity filtering, and statistical table calculations execute out-of-the-box on standard CPU architectures without requiring GPU acceleration or a PyTorch installation.

---

## 3. Computational Scope & Experimental Data Transparency

The BraTS 2021 benchmark comprises 1,251 multi-parametric 3D MRI volumes (~100 GB) hosted by the RSNA-ASNR-MICCAI BraTS consortium. Full training of all 31 model configurations across 5 random seeds (155 runs $\times$ 100 epochs) required approximately 1,800 GPU hours on our NVIDIA RTX 3090 compute cluster.

To provide complete academic transparency and practical reproducibility:
1. **Archived Historical Benchmark Registries (`per_seed_results.csv` and `severity_sweep_per_seed.csv`):** These CSV files are static historical benchmark registries compiled from the authors' original 1,800-GPU-hour multi-seed study runs (155 runs $\times$ 100 epochs across 31 model configurations $\times$ 5 seeds: 101–105 on an NVIDIA RTX 3090 cluster). The repository does **not** regenerate these 155 master runs from scratch during automated verification because doing so requires the external ~100 GB BraTS dataset and substantial GPU cluster time. Running `experiments/evaluate_all_tables.py` or `experiments/compute_statistics.py` reads these archived records to recompute all statistical tests ($t$-tests, ANCOVA, interaction contrasts, OLS slopes, bootstrap CIs) dynamically in ~2 seconds.
2. **Authentic Training Pipeline (`experiments/train_gegenbauer_detector.py`):** Provides the complete, genuine PyTorch detector training implementation featuring Task-Aligned Assigner ($t = s^\alpha \cdot \text{IoU}^\beta$), batched online spatial augmentations, AdamW optimization, cosine learning rate scheduling with 3-epoch linear warm-up, shuffled index sampling per epoch across all 14,873 training slices, full validation split evaluation (3,214 slices) for checkpoint selection, and `torch.save` of optimal checkpoints (`best.pt` and `last.pt`). For rapid smoke testing, `--max-train-samples` and `--max-val-samples` flags allow capping the slice iterations with explicit warning.
3. **Standalone Checkpoint Evaluator (`experiments/evaluate_checkpoint_to_csv.py`):** Takes a trained `.pt` checkpoint and authentic BraTS test volumes (`--data-root`), executes inference across clean and degraded test cohorts with Non-Maximum Suppression (NMS, IoU=0.50), matches detections strictly per image to prevent cross-image leakage, computes 101-point COCO AP metrics, and appends formatted rows to the CSV schema. The script requires a valid checkpoint, PyTorch, and an authentic dataset directory; it fails loudly without fallback to synthetic data or fabricated predictions. Comparative severity sweeps (`--sweep`) require both checkpoints to prevent writing uncomputed or fake margin rows.

---

## 4. Directory Structure

```text
reproducible_source_code/
├── README.md                              # Comprehensive documentation and execution guide
├── requirements.txt                       # Exact pinned environment dependencies
├── run_reproduce_all.sh                   # Executable shell runner
├── reproduce_all_results.py               # Master verification suite runner
│
├── data/
│   ├── bias_field_simulation.py           # Canonical & randomized 2D RF bias field simulators
│   └── mri_dataset.py                     # Patient-level BraTS 2021 NIfTI loader & synthetic generator
│
├── models/
│   ├── gegenbauer_activation.py           # Polynomial activation layer (Gegenbauer, Legendre, Chebyshev, Monomial)
│   └── yolov8_gegenbauer.py               # Full YOLOv8n detector with Gegenbauer & continuous head
│
├── losses/
│   └── logcosh_loss.py                    # Continuous Log-Cosh, CIoU, MSE, MAE, Smooth L1 losses
│
└── experiments/
    ├── basis_gram_conditioning.py         # Gram matrix & scale-invariant condition number verification
    ├── per_seed_results.csv               # Archived Registry: Master 155 per-seed study runs
    ├── severity_sweep_per_seed.csv        # Archived Registry: Parametric attenuation sweep runs
    ├── precompute_slices.py               # One-time slice cache (removes per-slice NIfTI re-reading)
    ├── compute_statistics.py              # Standalone dynamic derivation of Table 3 / Sec. 4.2 stats
    ├── evaluate_all_tables.py             # Dynamic table aggregator and statistical testing engine
    ├── evaluate_checkpoint_to_csv.py      # Evaluates model checkpoints on test cohort and exports CSV rows
    ├── multi_slice_filter.py              # 3D volumetric continuity post-processing filter
    ├── benchmark_latency.py               # Live local microbenchmarking & reference latency runner
    ├── run_n4itk_clahe_preprocessing.py   # OpenCV CLAHE and SimpleITK N4ITK preprocessing pipelines
    ├── train_gegenbauer_detector.py       # Authentic PyTorch training pipeline (Task-Aligned Assigner, Augmentation)
    ├── train_and_evaluate_study.py        # Multi-model architecture & loss ablation training harness
    └── train_ultralytics_integration.py   # Ultralytics 8.1.0 engine integration & monkey-patch hooks
```

---

## 5. Quick Start & Execution Guide

### Installation
```bash
pip install -r requirements.txt
```

### 1. Run Complete Master Verification Suite
```bash
python3 reproduce_all_results.py
```
Or execute via bash:
```bash
bash run_reproduce_all.sh
```

### 2. Run Complete Statistical Hypothesis Testing (Tables 1–6)
```bash
python3 experiments/evaluate_all_tables.py
```

### 3. Run Standalone Table 3 & Section 4.2 Statistical Verification
```bash
python3 experiments/compute_statistics.py
```

### 3b. Build the Preprocessed Slice Cache (recommended before any full-protocol run)
Without a cache, every slice access re-reads and decompresses two full 3D NIfTI volumes, and shuffled
sampling repeats this ~17 times per volume per epoch. One pass removes that bottleneck:
```bash
python3 experiments/precompute_slices.py --data-root /path/to/BraTS2021 --verify 20
```
This writes `slice_cache/{train,val,test}_{norm.npy,boxes.npy,meta.json}` under the data root (~4.8 GB at
float32; `--dtype float16` halves it with ~1e-3 value differences). The dataset, trainers and evaluator pick it up
automatically. `--verify N` checks N cached slices per split against on-the-fly loading (identical at float32).
A cache whose slice list, split, split seed or image size no longer matches raises an error; it is never used silently.

**Run-safety guards:** the training scripts refuse to start without `--data-root` unless `--synthetic` is passed
(or a `--max-train-samples` cap is given), and they refuse a `--data-root` that is not an existing directory
instead of silently falling back to synthetic data. Synthetic runs print a banner stating the metrics are not BraTS results.

### 4. Run Authentic PyTorch Training Pipeline
By default, the training harness executes on the full patient-level dataset splits (876 training volumes with 14,873 slices; 188 validation volumes with 3,214 slices) with a fresh random index permutation across epochs and batch size 16 matching the manuscript's protocol (Section 4.1). Checkpoint selection strictly tracks validation loss across the full validation split.

For rapid verification or smoke tests, pass `--max-train-samples` and `--max-val-samples` to cap the number of slices:

```bash
# A. Rapid Smoke Test / Verification Mode (3 epochs, capped slice subsets, loud warning):
python3 experiments/train_gegenbauer_detector.py --synthetic --epochs 3 --max-train-samples 32 --max-val-samples 16

# B. Full Manuscript Training Protocol on BraTS 2021 Cohort (100 epochs, all 14,873 train & 3,214 val slices, batch size 16):
python3 experiments/train_gegenbauer_detector.py --data-root /path/to/BraTS2021 --epochs 100 --batch-size 16

# C. Multi-Architecture Study & Ablation Training:
# Smoke test mode (capped):
python3 experiments/train_and_evaluate_study.py --synthetic --model proposed --loss logcosh --epochs 3 --max-train-samples 32 --max-val-samples 16

# Full cohort training:
python3 experiments/train_and_evaluate_study.py --data-root /path/to/BraTS2021 --model proposed --loss logcosh --epochs 100 --batch-size 16
```

### 5. Evaluate Checkpoints & Export Metric Rows to CSV
The standalone checkpoint evaluator (`experiments/evaluate_checkpoint_to_csv.py`) supports evaluating both Proposed (Gegenbauer) and Baseline (SiLU DFL) architectures across any attenuation gain $B_{\min} \in [0.35, 1.00]$, as well as the full 5-level parametric severity sweep.
Synthetic fallback is strictly disabled: `--data-root` pointing to the BraTS dataset is required. By default, `--max-slices all` evaluates the full 3,180 test cohort slices from all 187 test volumes (or pass an integer such as `--max-slices 50` for rapid inspection):

```bash
# A. Evaluate Proposed Gegenbauer Model at canonical severe stress (B_min=0.35) on full test split:
python3 experiments/evaluate_checkpoint_to_csv.py \
    --checkpoint runs/train/gegenbauer/best.pt \
    --arch proposed \
    --model "Proposed Model" \
    --seed 101 \
    --b-min 0.35 \
    --data-root /path/to/BraTS2021 \
    --max-slices all \
    --output-csv experiments/evaluated_metrics.csv

# B. Evaluate Baseline YOLOv8n Checkpoint across custom attenuation (e.g. B_min=0.70):
python3 experiments/evaluate_checkpoint_to_csv.py \
    --checkpoint runs/train/baseline/best.pt \
    --arch baseline \
    --model "Baseline YOLOv8n" \
    --seed 101 \
    --b-min 0.70 \
    --data-root /path/to/BraTS2021 \
    --max-slices all \
    --output-csv experiments/evaluated_metrics.csv

# C. Regenerate Parametric Severity Sweep (Table 3 across all 5 levels: 1.00, 0.85, 0.70, 0.50, 0.35):
python3 experiments/evaluate_checkpoint_to_csv.py \
    --checkpoint runs/train/gegenbauer/best.pt \
    --baseline-checkpoint runs/train/baseline/best.pt \
    --seed 101 \
    --data-root /path/to/BraTS2021 \
    --max-slices all \
    --sweep \
    --sweep-csv experiments/evaluated_sweep_results.csv
```

### 6. Run 3D Volumetric Continuity Post-Processing Filter
```bash
python3 experiments/multi_slice_filter.py
```

### 7. Run Live Latency Microbenchmark
```bash
python3 experiments/benchmark_latency.py --device cpu
```

### 8. Run Ultralytics Integration Sanity Check
```bash
python3 experiments/train_ultralytics_integration.py
```

---

## 7. Table & Statistical Verification Summary

The statistical suite evaluates all empirical findings reported in the manuscript:
- **Table 1 (Architectural Accounting):** Baseline 3,011,043 params vs. Proposed 3,021,487 params (+10,444 params, +0.35%, +0.01 GFLOPs).
- **Table 2 (Detector Comparison):** Proposed Model Attenuated mAP@0.5: $0.941 \pm 0.003$, mAP@0.75: $0.852 \pm 0.004$, mAP@[.5:.95]: $0.678 \pm 0.004$, Clean mAP@0.5: $0.952 \pm 0.003$, Attenuation Drop: $-0.011$, GPU Latency: $14.7$ ms.
- **Table 3 (Attenuation Severity Sweep):** OLS attenuation slope $-0.116$ (baseline) vs. $-0.061$ (proposed), seed-level paired slope test $t(4) = 7.43, p = 0.002$, differential drop reduction $+0.010$ (paired $t(4) = 7.47, p = 0.002$; seed-level Student's $t$ 95% CI $[+0.006, +0.014]$, bootstrap $[+0.007, +0.012]$).
- **Table 4 (Architectural Ablations):** Gegenbauer vs. Monomial: $\Delta = +0.016$ ($t(4) = 4.63, p = 0.010$); Gegenbauer vs. Legendre: $\Delta = +0.006$ ($t(4) = 2.44, p = 0.071$).
- **Table 5 ($2\times2$ Factorial of Activation $\times$ Head/Loss):** Interaction contrast $\Delta_{\text{inter}} = -0.002$, paired $t(4) = -0.56, p = 0.61$.
- **Table 6 (Loss Function Ablation):** Log-Cosh vs. MSE: $\Delta = +0.006$ (paired $t(4) = 2.91, p = 0.044$, seed-level Student's $t$ 95% CI $[+0.000, +0.012]$, bootstrap $[+0.002, +0.009]$).
