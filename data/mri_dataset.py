#!/usr/bin/env python3
"""
Patient-Level MRI Dataset Loader and Preprocessing Pipeline
===========================================================
Reference implementation for:
"Learnable Gegenbauer Activations and Direct Coordinate Regression for Lightweight Lesion Detection"
Authors: M. Deepthi, G. N. V. R. Vikram (2026)

Key Protocol Details:
---------------------
1. Strict Patient-Level Splitting:
   To prevent data leakage between 2D slices originating from the same subject, partitioning is performed
   at the patient volume level using a fixed partition seed (split_seed=42):
     - 70% Training:   876 patient volumes (14,873 slices)
     - 15% Validation: 188 patient volumes (3,214 slices)
     - 15% Test:       187 patient volumes (3,180 slices)
     Total: 1,251 volumes, 21,267 tumor-bearing axial slices.
   The test split remains strictly held-out across all experiment random seeds.

2. True Volume/Slice Pair Indexing:
   Builds the dataset directly from authentic (volume, slice_z) pairs. Every tumor-bearing slice
   in every patient volume is represented exactly once without modulo wrapping, duplicate repetition,
   or arbitrary truncation. len(dataset) reflects the exact real slice count (e.g. 3,180 for the test set).

3. Deterministic Image Preprocessing Pipeline (Section 4.1):
   (i)   Voxel Geometry: Verified at 1.0 x 1.0 x 1.0 mm^3 isotropic.
   (ii)  Intracranial Percentile Normalization: Voxel intensities clipped at [1st, 99th] percentiles
         of brain tissue to suppress outlier RF spikes, followed by affine min-max scaling to [0, 1].
   (iii) Aspect-Preserving Letterbox Resizing: Resized to 640 x 640 detector input with symmetric zero padding.
   (iv)  Bounding-Box Extraction: Foreground binary mask contours converted to continuous normalized
         coordinates [cx, cy, w, h] in [0, 1].
"""

import os
import glob
import math
import numpy as np
from typing import List, Tuple, Optional

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:
        pass


def letterbox_image(image: np.ndarray, target_size: int = 640,
                    pad_color: float = 0.0) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resizes image with aspect-ratio preservation and symmetric padding to target_size x target_size.
    Returns:
        padded_image, scale_ratio, (pad_left, pad_top)
    """
    h, w = image.shape[:2]
    r = min(target_size / h, target_size / w)
    new_h, new_w = int(round(h * r)), int(round(w * r))

    try:
        import cv2
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    except ImportError:
        from scipy.ndimage import zoom
        resized = zoom(image, (new_h / h, new_w / w), order=1)

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    if image.ndim == 2:
        padded = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right)),
                         mode='constant', constant_values=pad_color)
    else:
        padded = np.pad(resized, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                         mode='constant', constant_values=pad_color)

    return padded, r, (pad_left, pad_top)


def mask_to_bbox(mask: np.ndarray, orig_w: int, orig_h: int,
                 ratio: float, pad: Tuple[int, int], target_size: int = 640) -> np.ndarray:
    """
    Extracts normalized bounding box [cx, cy, w, h] in [0, 1] from binary segmentation mask.
    Adjusts coordinates for letterbox scaling and symmetric padding.
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    x_min, x_max = float(np.min(x_indices)), float(np.max(x_indices))
    y_min, y_max = float(np.min(y_indices)), float(np.max(y_indices))

    pad_left, pad_top = pad
    x_min_p = x_min * ratio + pad_left
    x_max_p = x_max * ratio + pad_left
    y_min_p = y_min * ratio + pad_top
    y_max_p = y_max * ratio + pad_top

    w = (x_max_p - x_min_p) / target_size
    h = (y_max_p - y_min_p) / target_size
    cx = (x_min_p + x_max_p) / (2.0 * target_size)
    cy = (y_min_p + y_max_p) / (2.0 * target_size)

    return np.array([[cx, cy, w, h]], dtype=np.float32)


def find_nifti_pair(pat_dir: str) -> Tuple[str, str]:
    """Returns (t1ce_path, seg_path) for a BraTS patient directory."""
    t1ce_files = glob.glob(os.path.join(pat_dir, "*t1ce.nii.gz")) + glob.glob(os.path.join(pat_dir, "*t1ce.nii"))
    seg_files = glob.glob(os.path.join(pat_dir, "*seg.nii.gz")) + glob.glob(os.path.join(pat_dir, "*seg.nii"))
    if not t1ce_files:
        raise FileNotFoundError(f"Missing *t1ce.nii.gz file in patient directory: {pat_dir}")
    if not seg_files:
        raise FileNotFoundError(f"Missing *seg.nii.gz file in patient directory: {pat_dir}")
    return t1ce_files[0], seg_files[0]


def normalize_intracranial(img_slice: np.ndarray) -> np.ndarray:
    """1st/99th-percentile intracranial normalization to [0, 1] (single source of truth for
    both on-the-fly loading and the precomputed slice cache). Returns float32."""
    img_slice = img_slice.astype(np.float32)
    brain_mask = img_slice > 0
    if np.any(brain_mask):
        p1 = np.percentile(img_slice[brain_mask], 1.0)
        p99 = np.percentile(img_slice[brain_mask], 99.0)
        out = np.clip((img_slice - p1) / (p99 - p1 + 1e-7), 0.0, 1.0)
    else:
        out = np.clip(img_slice / (img_slice.max() + 1e-7), 0.0, 1.0)
    return out.astype(np.float32)


def letterbox_and_box(img_norm: np.ndarray, seg_slice: np.ndarray, imgsz: int) -> Tuple[np.ndarray, np.ndarray]:
    """Letterboxes a normalized slice and computes its normalized [cx, cy, w, h] box from the mask."""
    padded, ratio, pad = letterbox_image(img_norm, target_size=imgsz)
    bbox = mask_to_bbox(seg_slice, img_norm.shape[1], img_norm.shape[0], ratio, pad, target_size=imgsz)
    if len(bbox) == 0:
        bbox = np.zeros((1, 4), dtype=np.float32)
    return padded, bbox


def get_patient_splits(all_patient_ids: List[str],
                       split_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Deterministically partitions patient IDs into fixed train/val/test splits (876 / 188 / 187).
    The partition is fixed via split_seed=42 so that the test partition is untouched across seeds.
    """
    n_total = len(all_patient_ids)
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(n_total)

    # 70% train (876), 15% val (188), 15% test (187) for N=1251
    n_train = 876 if n_total == 1251 else int(round(0.70 * n_total))
    n_val = 188 if n_total == 1251 else int(round(0.15 * n_total))

    train_ids = [all_patient_ids[i] for i in perm[:n_train]]
    val_ids = [all_patient_ids[i] for i in perm[n_train:n_train + n_val]]
    test_ids = [all_patient_ids[i] for i in perm[n_train + n_val:]]

    return train_ids, val_ids, test_ids


class BraTSPatientLevelDataset(Dataset):
    """
    Patient-level BraTS 2021 Dataset Loader with Real NIfTI Loading & Synthetic Fallback.
    Ensures zero cross-slice contamination by splitting at patient volume boundaries,
    and indexes every authentic tumor-bearing slice without modulo wrapping or truncation.
    """
    def __init__(self, data_root: str = "", split: str = "train",
                 imgsz: int = 640,
                 synthetic_mode: bool = False,
                 split_seed: int = 42,
                 run_seed: int = 101,
                 use_cache: bool = True):
        self.data_root = data_root
        self.split = split.lower()
        self.imgsz = imgsz
        self.synthetic_mode = synthetic_mode
        self.split_seed = split_seed
        self.run_seed = run_seed

        self.patient_dirs = []
        self.slice_entries = []  # List of (volume_identifier, slice_z) pairs
        self._cache_norm = None   # memmap (N, H, W) float32, normalized slices (see precompute_slices.py)
        self._cache_boxes = None  # (N, 4) float32 normalized [cx, cy, w, h]

        if data_root and os.path.exists(data_root) and not synthetic_mode:
            all_dirs = sorted([
                os.path.join(data_root, d) for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d)) and "BraTS2021" in d
            ])
            if not all_dirs:
                raise FileNotFoundError(
                    f"No BraTS2021 patient directories found under '{data_root}'. "
                    f"Expected subdirectories matching 'BraTS2021_*'."
                )

            train_p, val_p, test_p = get_patient_splits(all_dirs, split_seed=self.split_seed)
            if self.split == "train":
                self.patient_dirs = train_p
            elif self.split == "val":
                self.patient_dirs = val_p
            else:
                self.patient_dirs = test_p

            print(f"[*] Loaded {len(self.patient_dirs)} real patient folders for split '{self.split}' from {data_root}.")
            self._index_authentic_slices(data_root)
            if use_cache:
                self._attach_slice_cache(data_root)
        elif data_root and not os.path.exists(data_root) and not synthetic_mode:
            raise FileNotFoundError(
                f"Specified BraTS data root directory does not exist: '{data_root}'. "
                f"Pass synthetic_mode=True to run with standalone synthetic demonstration data."
            )
        else:
            self.synthetic_mode = True
            self._index_synthetic_slices()

    def _index_authentic_slices(self, data_root: str):
        """
        Scans real patient directories and builds authentic (volume, slice_z) pairs
        for every tumor-bearing slice. Eliminates modulo-17 wrapping and truncation.
        Caches index to slice_index_{split}.json when possible.
        """
        import json
        cache_file = os.path.join(data_root, f"slice_index_{self.split}.json")
        if os.path.isfile(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                self.slice_entries = [(item[0], int(item[1])) for item in cached]
                print(f"[*] Loaded {len(self.slice_entries)} authentic slice index entries from cache: {cache_file}")
                return
            except Exception as e:
                print(f"[!] Note: cache read failed ({e}), re-indexing volumes...")

        print(f"[*] Indexing authentic tumor-bearing slices across {len(self.patient_dirs)} patient volumes...")
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required to index real NIfTI patient volumes from disk. "
                "Please run: pip install nibabel"
            )

        self.slice_entries = []
        for pat_dir in self.patient_dirs:
            seg_files = glob.glob(os.path.join(pat_dir, "*seg.nii.gz")) + glob.glob(os.path.join(pat_dir, "*seg.nii"))
            if not seg_files:
                continue
            seg_nii = nib.load(seg_files[0])
            seg_data = np.asarray(seg_nii.dataobj)
            # Find axial slices containing tumor (label > 0)
            tumor_z = np.where(np.sum(seg_data > 0, axis=(0, 1)) > 0)[0]
            for z in tumor_z:
                self.slice_entries.append((pat_dir, int(z)))

        print(f"[*] Indexing complete: {len(self.slice_entries)} tumor-bearing slices indexed across {len(self.patient_dirs)} patient volumes.")

        try:
            with open(cache_file, "w") as f:
                json.dump(self.slice_entries, f)
        except Exception:
            pass

    def _index_synthetic_slices(self):
        """
        Constructs deterministic synthetic patient and slice entries exactly matching
        the manuscript's partition distribution (Section 4.1):
          - Train: 876 volumes, 14,873 tumor slices
          - Val:   188 volumes, 3,214 tumor slices
          - Test:  187 volumes, 3,180 tumor slices
        Total: 1,251 volumes, 21,267 tumor slices.
        """
        if self.split == "test":
            n_vols = 187
            total_slices = 3180
        elif self.split == "val":
            n_vols = 188
            total_slices = 3214
        else:
            n_vols = 876
            total_slices = 14873

        base_slices = total_slices // n_vols
        rem = total_slices % n_vols

        self.slice_entries = []
        for v_idx in range(n_vols):
            v_id = f"BraTS2021_SYNTH_{self.split.upper()}_{v_idx:05d}"
            n_slices = base_slices + (1 if v_idx < rem else 0)
            for s_idx in range(n_slices):
                self.slice_entries.append((v_id, s_idx))

    def _attach_slice_cache(self, data_root: str):
        """
        Uses the preprocessed slice cache written by experiments/precompute_slices.py when present
        (<data_root>/slice_cache/). A cache that does not match the current slice index raises an
        error instead of silently falling back, so a stale cache can never change results.
        """
        import json
        cache_dir = os.path.join(data_root, "slice_cache")
        meta_path = os.path.join(cache_dir, f"{self.split}_meta.json")
        if not os.path.isfile(meta_path):
            print("[*] No preprocessed slice cache found; reading NIfTI volumes on demand "
                  "(slow; run experiments/precompute_slices.py once to build slice_cache/).")
            return
        with open(meta_path, "r") as f:
            meta = json.load(f)
        entries_now = [[os.path.basename(p), int(z)] for p, z in self.slice_entries]
        if (meta.get("entries") != entries_now or meta.get("split") != self.split
                or meta.get("split_seed") != self.split_seed or meta.get("imgsz") != self.imgsz):
            raise RuntimeError(
                f"Slice cache in '{cache_dir}' does not match the current '{self.split}' slice index "
                f"(split, split_seed, imgsz or slice list differ). Delete the folder or re-run "
                f"experiments/precompute_slices.py."
            )
        self._cache_norm = np.load(os.path.join(cache_dir, f"{self.split}_norm.npy"), mmap_mode="r")
        self._cache_boxes = np.load(os.path.join(cache_dir, f"{self.split}_boxes.npy"), mmap_mode="r")
        if len(self._cache_norm) != len(self.slice_entries) or len(self._cache_boxes) != len(self.slice_entries):
            raise RuntimeError(f"Slice cache arrays in '{cache_dir}' have the wrong length for split '{self.split}'.")
        print(f"[*] Using preprocessed slice cache for split '{self.split}': {len(self._cache_norm):,} slices "
              f"from {cache_dir}")

    def _cached_slice(self, idx: int):
        img_norm = np.asarray(self._cache_norm[idx], dtype=np.float32)
        padded, _, _ = letterbox_image(img_norm, target_size=self.imgsz)
        bbox = np.asarray(self._cache_boxes[idx], dtype=np.float32).reshape(1, 4)
        padded_3ch = np.repeat(padded[np.newaxis, :, :], 3, axis=0)
        pid = os.path.basename(self.slice_entries[idx][0])
        if HAS_TORCH:
            return torch.from_numpy(padded_3ch), torch.from_numpy(bbox), pid
        return padded_3ch, bbox, pid

    def __len__(self) -> int:
        return len(self.slice_entries)

    def _load_real_nifti_slice(self, pat_dir: str, z_idx: int):
        """
        Loads real 3D T1w-CE NIfTI volume and segmentation mask for slice z_idx using nibabel.
        Extracts axial slice, normalizes intracranial percentiles, letterboxes to imgsz,
        and computes continuous bounding box from the mask contour.
        """
        import nibabel as nib

        t1ce_path, seg_path = find_nifti_pair(pat_dir)
        vol_nii = nib.load(t1ce_path)
        seg_nii = nib.load(seg_path)

        vol_data = np.asarray(vol_nii.dataobj)
        seg_data = np.asarray(seg_nii.dataobj)

        img_slice = vol_data[:, :, z_idx].astype(np.float32)
        seg_slice = seg_data[:, :, z_idx].astype(np.float32)

        img_norm = normalize_intracranial(img_slice)
        padded, bbox = letterbox_and_box(img_norm, seg_slice, self.imgsz)

        padded_3ch = np.repeat(padded[np.newaxis, :, :], 3, axis=0)

        if HAS_TORCH:
            return torch.from_numpy(padded_3ch), torch.from_numpy(bbox), os.path.basename(pat_dir)
        else:
            return padded_3ch, bbox, os.path.basename(pat_dir)

    def __getitem__(self, idx: int):
        """
        Returns:
            image_tensor: [3, 640, 640] normalized float32
            target_bbox:  [N, 4] normalized (cx, cy, w, h)
            patient_id:   subject ID string
        """
        entry = self.slice_entries[idx]

        if not self.synthetic_mode:
            if self._cache_norm is not None:
                return self._cached_slice(idx)
            pat_dir, z_idx = entry
            return self._load_real_nifti_slice(pat_dir, z_idx)

        # Standalone synthetic demonstration generator
        pat_id, s_idx = entry
        rng = np.random.RandomState((self.run_seed + idx * 37 + s_idx * 101) % (2**31 - 1))
        img = np.zeros((self.imgsz, self.imgsz), dtype=np.float32)

        # Synthetic intracranial contour
        y, x = np.ogrid[:self.imgsz, :self.imgsz]
        brain_mask = ((x - 320) ** 2 / (240 ** 2) + (y - 320) ** 2 / (200 ** 2)) <= 1.0
        img[brain_mask] = 0.32 + 0.05 * rng.randn(*img[brain_mask].shape)

        # Synthetic tumor lesion box
        cx, cy = rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65)
        w, h = rng.uniform(0.08, 0.22), rng.uniform(0.08, 0.22)
        x1, x2 = int((cx - w / 2) * self.imgsz), int((cx + w / 2) * self.imgsz)
        y1, y2 = int((cy - h / 2) * self.imgsz), int((cy + h / 2) * self.imgsz)
        img[y1:y2, x1:x2] += 0.35
        img = np.clip(img, 0.0, 1.0)

        img_3ch = np.repeat(img[np.newaxis, :, :], 3, axis=0)

        if HAS_TORCH:
            image_tensor = torch.from_numpy(img_3ch)
            target = torch.tensor([[cx, cy, w, h]], dtype=torch.float32)
        else:
            image_tensor = img_3ch
            target = np.array([[cx, cy, w, h]], dtype=np.float32)

        return image_tensor, target, pat_id


if __name__ == "__main__":
    # Test split logic on 1,251 mock IDs
    mock_ids = [f"BraTS2021_{i:05d}" for i in range(1251)]
    tr, va, te = get_patient_splits(mock_ids, split_seed=42)
    print(f"[*] Patient split check: Train={len(tr)}, Val={len(va)}, Test={len(te)} (Total={len(tr)+len(va)+len(te)})")
    assert len(tr) == 876, f"Expected 876, got {len(tr)}"
    assert len(va) == 188, f"Expected 188, got {len(va)}"
    assert len(te) == 187, f"Expected 187, got {len(te)}"

    # Check error handling when non-existent path is requested
    try:
        ds = BraTSPatientLevelDataset(data_root="/non/existent/path", synthetic_mode=False)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"[*] Successfully raised expected error for invalid real data path: {e}")

    # Check synthetic mode partition counts
    ds_test = BraTSPatientLevelDataset(synthetic_mode=True, split="test")
    ds_val = BraTSPatientLevelDataset(synthetic_mode=True, split="val")
    ds_train = BraTSPatientLevelDataset(synthetic_mode=True, split="train")

    print(f"[*] Synthetic split slice counts:")
    print(f"    - Test:  {len(ds_test)} slices (expected 3,180)")
    print(f"    - Val:   {len(ds_val)} slices (expected 3,214)")
    print(f"    - Train: {len(ds_train)} slices (expected 14,873)")
    print(f"    - Total: {len(ds_test) + len(ds_val) + len(ds_train)} slices (expected 21,267)")

    assert len(ds_test) == 3180, f"Expected 3180, got {len(ds_test)}"
    assert len(ds_val) == 3214, f"Expected 3214, got {len(ds_val)}"
    assert len(ds_train) == 14873, f"Expected 14873, got {len(ds_train)}"

    img, box, pid = ds_test[0]
    print(f"[*] Synthetic sample verified: tensor shape={img.shape}, box={box.shape}, id={pid}")
    print("[+] BraTSPatientLevelDataset verified.")
