#!/usr/bin/env python3
"""
Precompute Slice Cache for BraTS 2021
=====================================
One-time preprocessing pass that removes the per-slice NIfTI re-reading bottleneck.

Without a cache, BraTSPatientLevelDataset.__getitem__ loads and decompresses two full 3D volumes
(T1ce and segmentation) for EVERY slice. With shuffled sampling this happens ~17x per volume per
epoch. This script reads each volume once and writes, per split, to <data-root>/slice_cache/:

    {split}_norm.npy   (N, H, W) float32  intracranial-percentile-normalized slices in [0, 1]
    {split}_boxes.npy  (N, 4)    float32  letterboxed normalized [cx, cy, w, h] from the mask
    {split}_meta.json  slice list (volume basename, z), split, split_seed, imgsz, shapes

The dataset class picks the cache up automatically. Letterbox resizing is still applied at load time
(cheap), and the normalization/box code is the SAME shared functions used by the on-the-fly path, so
cached and uncached loading return identical tensors (verified by --verify). A cache whose slice list,
split, split seed, or image size no longer matches raises an error instead of being used silently.

Disk use (float32, full BraTS 2021 T1ce, 240x240 slices): about 3.4 GB train + 0.7 GB val + 0.7 GB test.
Use --dtype float16 to halve this (values then differ from the uncached path by ~1e-3).

Usage:
    python3 experiments/precompute_slices.py --data-root /path/to/BraTS2021
    python3 experiments/precompute_slices.py --data-root /path/to/BraTS2021 --splits val test --verify 20
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.mri_dataset import (  # noqa: E402
    BraTSPatientLevelDataset,
    find_nifti_pair,
    normalize_intracranial,
    letterbox_and_box,
)


def build_split_cache(data_root: str, split: str, imgsz: int, split_seed: int, dtype: str):
    import nibabel as nib

    cache_dir = os.path.join(data_root, "slice_cache")
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, f"{split}_meta.json")
    if os.path.isfile(meta_path):
        os.remove(meta_path)  # meta is written last, so an interrupted run is never picked up

    # Index only (no cache attached) so the cache is built from the exact slice list the dataset uses.
    ds = BraTSPatientLevelDataset(data_root=data_root, split=split, imgsz=imgsz,
                                  split_seed=split_seed, use_cache=False)
    entries = ds.slice_entries
    n = len(entries)
    if n == 0:
        raise RuntimeError(f"No tumor-bearing slices indexed for split '{split}'.")

    # Group consecutive entries by volume (the index is built volume by volume).
    groups = []
    for i, (pat_dir, z) in enumerate(entries):
        if groups and groups[-1][0] == pat_dir:
            groups[-1][1].append((i, z))
        else:
            groups.append((pat_dir, [(i, z)]))

    np_dtype = np.float16 if dtype == "float16" else np.float32
    norm_mm, boxes, shape_hw = None, np.zeros((n, 4), dtype=np.float32), None
    t0 = time.time()

    for g_idx, (pat_dir, items) in enumerate(groups):
        t1ce_path, seg_path = find_nifti_pair(pat_dir)
        vol = np.asarray(nib.load(t1ce_path).dataobj)
        seg = np.asarray(nib.load(seg_path).dataobj)
        for i, z in items:
            img_slice = vol[:, :, z].astype(np.float32)
            seg_slice = seg[:, :, z].astype(np.float32)
            img_norm = normalize_intracranial(img_slice)
            if shape_hw is None:
                shape_hw = img_norm.shape
                norm_mm = np.lib.format.open_memmap(
                    os.path.join(cache_dir, f"{split}_norm.npy"), mode="w+",
                    dtype=np_dtype, shape=(n,) + shape_hw)
            elif img_norm.shape != shape_hw:
                raise RuntimeError(f"Non-uniform slice shape {img_norm.shape} vs {shape_hw} in {pat_dir}; "
                                   f"cache requires uniform in-plane size.")
            _, bbox = letterbox_and_box(img_norm, seg_slice, imgsz)
            norm_mm[i] = img_norm.astype(np_dtype)
            boxes[i] = bbox[0]
        if (g_idx + 1) % 25 == 0 or g_idx + 1 == len(groups):
            print(f"    [{split}] {g_idx + 1}/{len(groups)} volumes, "
                  f"{items[-1][0] + 1}/{n} slices, {time.time() - t0:.0f}s", flush=True)

    norm_mm.flush()
    np.save(os.path.join(cache_dir, f"{split}_boxes.npy"), boxes)
    meta = {
        "split": split,
        "split_seed": split_seed,
        "imgsz": imgsz,
        "dtype": dtype,
        "n_slices": n,
        "slice_shape": list(shape_hw),
        "entries": [[os.path.basename(p), int(z)] for p, z in entries],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"[+] {split}: wrote {n:,} slices ({n * shape_hw[0] * shape_hw[1] * np.dtype(np_dtype).itemsize / 1e9:.2f} GB) "
          f"to {cache_dir}")
    return n


def verify(data_root: str, split: str, imgsz: int, split_seed: int, n_check: int, dtype: str):
    """Compares cached vs on-the-fly loading on evenly spaced slices."""
    ds_raw = BraTSPatientLevelDataset(data_root=data_root, split=split, imgsz=imgsz,
                                      split_seed=split_seed, use_cache=False)
    ds_cached = BraTSPatientLevelDataset(data_root=data_root, split=split, imgsz=imgsz,
                                         split_seed=split_seed, use_cache=True)
    if ds_cached._cache_norm is None:
        raise RuntimeError(f"Cache for split '{split}' was not picked up by the dataset.")
    idxs = np.unique(np.linspace(0, len(ds_raw) - 1, min(n_check, len(ds_raw))).astype(int))
    tol = 5e-3 if dtype == "float16" else 0.0
    worst = 0.0
    for i in idxs:
        a_img, a_box, a_id = ds_raw[int(i)]
        b_img, b_box, b_id = ds_cached[int(i)]
        a_img, b_img = np.asarray(a_img), np.asarray(b_img)
        a_box, b_box = np.asarray(a_box), np.asarray(b_box)
        err = max(float(np.abs(a_img - b_img).max()), float(np.abs(a_box - b_box).max()))
        worst = max(worst, err)
        if a_id != b_id or err > tol:
            raise RuntimeError(f"Cache mismatch at {split}[{i}]: max abs diff {err:.3e} (tol {tol}), ids {a_id}/{b_id}")
    print(f"[+] verify {split}: {len(idxs)} slices identical within tol={tol} (worst abs diff {worst:.3e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Precompute normalized slice cache for BraTS 2021")
    ap.add_argument("--data-root", required=True, help="BraTS 2021 root containing BraTS2021_* patient folders")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="After building, compare N evenly spaced cached slices per split with on-the-fly loading")
    args = ap.parse_args()

    if not os.path.isdir(args.data_root):
        ap.error(f"--data-root '{args.data_root}' is not a directory")

    for sp in args.splits:
        print(f"[*] Building slice cache for split '{sp}' ...")
        build_split_cache(args.data_root, sp, args.imgsz, args.split_seed, args.dtype)
        if args.verify > 0:
            verify(args.data_root, sp, args.imgsz, args.split_seed, args.verify, args.dtype)
    print("[+] Done. Training and evaluation will use slice_cache/ automatically.")
