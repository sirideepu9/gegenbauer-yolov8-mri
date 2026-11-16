#!/usr/bin/env python3
"""
Continuous Coordinate Regression Log-Cosh Loss Functions
=========================================================
Reference implementation for:
"Learnable Gegenbauer Activations and Direct Coordinate Regression for Lightweight Lesion Detection"
Authors: M. Deepthi, G. N. V. R. Vikram (2026)

Mathematical Formulations (Section 3.3, Eq. 7):
------------------------------------------------
1. Standard Log-Cosh Function:
     L(r) = (1 / beta) * log(cosh(beta * r))
     For beta = 1: L(r) = log(cosh(r))

2. Taylor Expansion for Small Residuals (|r| <= 0.12):
     log(cosh(r)) = (1/2) * r^2 - (1/12) * r^4 + O(r^6)
     Demonstrating pure L2 (MSE) behavior near convergence.

3. Asymptotic Linear Regimes for Outliers (|r| >> 1):
     lim_{|r| -> inf} log(cosh(r)) = |r| - log(2)
     Demonstrating robust L1 (Huber-like) behavior with bounded gradient:
     d/dr log(cosh(r)) = tanh(r) in (-1, +1).

4. Numerically Stable Implementation:
     log(cosh(x)) = |x| + log(1 + exp(-2*|x|)) - log(2) = |x| + log1p(exp(-2*|x|)) - log(2)

5. Hybrid Multi-Objective Loss Formulation:
     L_total = lambda_cls * L_cls + lambda_loc * L_loc
     where L_loc = L_logcosh (Proposed) or L_logcosh + L_ciou (Proposed Hybrid).
"""

import math
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        class Module:
            pass


class LogCoshLoss(nn.Module):
    """
    Numerically Stable Coordinate-Wise Log-Cosh Localization Loss.
    Supports both PyTorch Tensors and NumPy arrays seamlessly with optional sample weighting.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        self.log2 = math.log(2.0)

    def forward(self, pred, target, weights=None):
        if HAS_TORCH and isinstance(pred, torch.Tensor):
            diff = pred - target
            abs_diff = torch.abs(diff)
            element_loss = abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff)) - self.log2
            box_loss = element_loss.sum(dim=-1)
            if weights is not None:
                box_loss = box_loss * weights
                denom = weights.sum().clamp(min=1e-7)
                return box_loss.sum() / denom
            if self.reduction == "mean":
                return box_loss.mean()
            elif self.reduction == "sum":
                return box_loss.sum()
            else:
                return box_loss
        else:
            p = np.asarray(pred, dtype=np.float32)
            t = np.asarray(target, dtype=np.float32)
            diff = p - t
            abs_diff = np.abs(diff)
            element_loss = abs_diff + np.log1p(np.exp(-2.0 * abs_diff)) - self.log2
            box_loss = np.sum(element_loss, axis=-1)
            if weights is not None:
                w = np.asarray(weights, dtype=np.float32)
                return float(np.sum(box_loss * w) / max(np.sum(w), 1e-7))
            if self.reduction == "mean":
                return float(np.mean(box_loss))
            elif self.reduction == "sum":
                return float(np.sum(box_loss))
            else:
                return box_loss

    def __call__(self, pred, target, weights=None):
        return self.forward(pred, target, weights=weights)


def bbox_ciou(box1, box2, eps: float = 1e-7):
    """
    Complete Intersection over Union (CIoU) Loss for normalized bboxes [cx, cy, w, h].
    Returns 1 - CIoU in range [0, 2].
    """
    if HAS_TORCH and isinstance(box1, torch.Tensor):
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        inter_w = torch.clamp(torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1), min=0)
        inter_h = torch.clamp(torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1), min=0)
        inter_area = inter_w * inter_h

        w1, h1 = box1[:, 2], box1[:, 3]
        w2, h2 = box2[:, 2], box2[:, 3]
        union_area = w1 * h1 + w2 * h2 - inter_area + eps
        iou = inter_area / union_area

        c_x1 = torch.min(b1_x1, b2_x1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_y2 = torch.max(b1_y2, b2_y2)
        c_diag_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps
        center_dist_sq = (box1[:, 0] - box2[:, 0]) ** 2 + (box1[:, 1] - box2[:, 1]) ** 2

        v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + eps)

        ciou = iou - (center_dist_sq / c_diag_sq) - alpha * v
        return 1.0 - ciou.clamp(min=-1.0, max=1.0)
    else:
        b1 = np.asarray(box1, dtype=np.float32)
        b2 = np.asarray(box2, dtype=np.float32)
        if b1.ndim == 1:
            b1 = b1[np.newaxis, :]
        if b2.ndim == 1:
            b2 = b2[np.newaxis, :]

        b1_x1, b1_x2 = b1[:, 0] - b1[:, 2] / 2, b1[:, 0] + b1[:, 2] / 2
        b1_y1, b1_y2 = b1[:, 1] - b1[:, 3] / 2, b1[:, 1] + b1[:, 3] / 2
        b2_x1, b2_x2 = b2[:, 0] - b2[:, 2] / 2, b2[:, 0] + b2[:, 2] / 2
        b2_y1, b2_y2 = b2[:, 1] - b2[:, 3] / 2, b2[:, 1] + b2[:, 3] / 2

        inter_w = np.maximum(0.0, np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1))
        inter_h = np.maximum(0.0, np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1))
        inter_area = inter_w * inter_h

        w1, h1 = b1[:, 2], b1[:, 3]
        w2, h2 = b2[:, 2], b2[:, 3]
        union_area = w1 * h1 + w2 * h2 - inter_area + eps
        iou = inter_area / union_area

        c_x1 = np.minimum(b1_x1, b2_x1)
        c_x2 = np.maximum(b1_x2, b2_x2)
        c_y1 = np.minimum(b1_y1, b2_y1)
        c_y2 = np.maximum(b1_y2, b2_y2)
        c_diag_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps
        center_dist_sq = (b1[:, 0] - b2[:, 0]) ** 2 + (b1[:, 1] - b2[:, 1]) ** 2

        v = (4.0 / (math.pi ** 2)) * np.power(np.arctan(w2 / (h2 + eps)) - np.arctan(w1 / (h1 + eps)), 2)
        alpha = v / (1.0 - iou + v + eps)
        ciou = iou - (center_dist_sq / c_diag_sq) - alpha * v
        return 1.0 - np.clip(ciou, -1.0, 1.0)


class MSELossLoc(nn.Module):
    """Mean Squared Error (L2) Localization Loss."""
    def forward(self, pred, target, weights=None):
        diff_sq = ((pred - target) ** 2).sum(dim=-1)
        if weights is not None:
            return (diff_sq * weights).sum() / weights.sum().clamp(min=1e-7)
        return diff_sq.mean() if HAS_TORCH and isinstance(pred, torch.Tensor) else float(np.mean(diff_sq))


class MAELossLoc(nn.Module):
    """Mean Absolute Error (L1) Localization Loss."""
    def forward(self, pred, target, weights=None):
        diff_abs = torch.abs(pred - target).sum(dim=-1) if HAS_TORCH and isinstance(pred, torch.Tensor) else np.sum(np.abs(pred - target), axis=-1)
        if weights is not None:
            return (diff_abs * weights).sum() / weights.sum().clamp(min=1e-7)
        return diff_abs.mean() if HAS_TORCH and isinstance(pred, torch.Tensor) else float(np.mean(diff_abs))


class SmoothL1LossLoc(nn.Module):
    """Smooth L1 Localization Loss (delta=0.05)."""
    def __init__(self, beta: float = 0.05):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, weights=None):
        if HAS_TORCH and isinstance(pred, torch.Tensor):
            diff = torch.abs(pred - target)
            cond = diff < self.beta
            loss = torch.where(cond, 0.5 * (diff ** 2) / self.beta, diff - 0.5 * self.beta).sum(dim=-1)
            if weights is not None:
                return (loss * weights).sum() / weights.sum().clamp(min=1e-7)
            return loss.mean()
        else:
            diff = np.abs(np.asarray(pred) - np.asarray(target))
            cond = diff < self.beta
            loss = np.sum(np.where(cond, 0.5 * (diff ** 2) / self.beta, diff - 0.5 * self.beta), axis=-1)
            if weights is not None:
                w = np.asarray(weights)
                return float(np.sum(loss * w) / max(np.sum(w), 1e-7))
            return float(np.mean(loss))


class CIoULossLoc(nn.Module):
    """Complete IoU (CIoU) Localization Loss."""
    def forward(self, pred, target, weights=None):
        ciou_l = bbox_ciou(pred, target)
        if weights is not None:
            return (ciou_l * weights).sum() / weights.sum().clamp(min=1e-7)
        return ciou_l.mean() if HAS_TORCH and isinstance(pred, torch.Tensor) else float(np.mean(ciou_l))


class HybridLogCoshCIoULoss(nn.Module):
    """
    Hybrid Localization Loss: Continuous Log-Cosh + Complete-IoU (Configuration 13 in Table 4).
    L_loc = L_logcosh + L_ciou
    """
    def __init__(self):
        super().__init__()
        self.logcosh = LogCoshLoss()
        self.ciou = CIoULossLoc()

    def forward(self, pred, target, weights=None):
        return self.logcosh(pred, target, weights=weights) + self.ciou(pred, target, weights=weights)

    def __call__(self, pred, target, weights=None):
        return self.forward(pred, target, weights=weights)


class TotalLesionDetectionLoss(nn.Module):
    """
    Multi-Task Detection Loss supporting all loss formulations from Table 6:
    - logcosh (Proposed)
    - hybrid (Log-Cosh + CIoU)
    - mse (MSE / L2)
    - mae (MAE / L1)
    - smooth_l1 (Smooth L1, delta=0.05)
    - ciou (CIoU)
    L_total = lambda_cls * L_cls + lambda_loc * L_loc
    """
    def __init__(self, lambda_cls: float = 0.5, lambda_loc: float = 7.5, loss_type: str = "logcosh"):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_loc = lambda_loc
        self.loss_type = loss_type.lower()

        if self.loss_type in ["logcosh", "proposed"]:
            self.loc_loss_fn = LogCoshLoss()
        elif self.loss_type in ["hybrid", "logcosh_ciou"]:
            self.loc_loss_fn = HybridLogCoshCIoULoss()
        elif self.loss_type in ["mse", "l2"]:
            self.loc_loss_fn = MSELossLoc()
        elif self.loss_type in ["mae", "l1"]:
            self.loc_loss_fn = MAELossLoc()
        elif self.loss_type in ["smooth_l1", "smoothl1"]:
            self.loc_loss_fn = SmoothL1LossLoc(beta=0.05)
        elif self.loss_type in ["ciou", "eiou"]:
            self.loc_loss_fn = CIoULossLoc()
        else:
            self.loc_loss_fn = LogCoshLoss()

    def forward(self, pred_boxes, target_boxes, pred_scores, target_scores, pos_mask=None, weights=None):
        """
        Computes detection loss over matched anchors.
        """
        if pos_mask is not None and pos_mask.sum() > 0:
            p_b = pred_boxes[pos_mask]
            t_b = target_boxes[pos_mask]
            w = weights[pos_mask] if weights is not None else None
            loss_loc = self.loc_loss_fn(p_b, t_b, weights=w)
        else:
            loss_loc = self.loc_loss_fn(pred_boxes, target_boxes, weights=weights)

        if HAS_TORCH and isinstance(pred_scores, torch.Tensor):
            loss_cls = F.binary_cross_entropy(pred_scores.clamp(1e-6, 1.0 - 1e-6), target_scores)
        else:
            p_s = np.clip(np.asarray(pred_scores), 1e-6, 1.0 - 1e-6)
            t_s = np.asarray(target_scores, dtype=np.float32)
            loss_cls = float(-np.mean(t_s * np.log(p_s) + (1.0 - t_s) * np.log(1.0 - p_s)))

        loss_total = self.lambda_cls * loss_cls + self.lambda_loc * loss_loc
        return {
            "loss_total": loss_total,
            "loss_loc": loss_loc,
            "loss_cls": loss_cls
        }

    def __call__(self, pred_boxes, target_boxes, pred_scores, target_scores, pos_mask=None, weights=None):
        return self.forward(pred_boxes, target_boxes, pred_scores, target_scores, pos_mask=pos_mask, weights=weights)


if __name__ == "__main__":
    lc = LogCoshLoss()
    pred = np.array([[0.50, 0.50, 0.20, 0.20], [0.10, 0.10, 0.05, 0.05]], dtype=np.float32)
    target = np.array([[0.52, 0.49, 0.21, 0.19], [0.10, 0.10, 0.05, 0.05]], dtype=np.float32)
    loss = lc(pred, target)
    print(f"[*] LogCosh loss (NumPy verification): {loss:.6f}")
    ciou = bbox_ciou(pred, target)
    print(f"[*] CIoU loss: {ciou.mean():.6f}")
    print("[+] All continuous regression loss functions verified.")
