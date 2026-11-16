#!/usr/bin/env python3
"""
Gegenbauer-YOLOv8 Architecture Reference Model
=============================================
Reference implementation for:
"Learnable Gegenbauer Activations and Direct Coordinate Regression for Lightweight Lesion Detection"
Authors: M. Deepthi, G. N. V. R. Vikram (2026)

Key Architectural Specifications (Section 3):
----------------------------------------------
1. Channel-Wise Gegenbauer Polynomial Activations (Section 3.1):
   Replaces all static scalar activations throughout the Backbone (26 layers, 2,512 channels)
   and Neck (19 layers, 1,920 channels) with learnable cubic Gegenbauer expansions (K=3, beta_c > 0).
   Total activation channels: 4,432. Added parameters: 5 * 4,432 = +22,160 (+0.74%).

2. Direct 4-Channel Continuous Coordinate Regression Head (Section 3.3):
   Replaces the 64-channel discrete Distribution Focal Loss (DFL) head (16 probability bins per coordinate)
   with a direct 4-coordinate continuous regression head (cx, cy, w, h).
   Saves 11,700 projection parameters + 16 DFL convolution parameters = -11,716 parameters across scales P3, P4, P5.
   Net parameter delta: +22,160 - 11,716 = +10,444 parameters (+0.35%).
   Total parameters: 3,021,487 (vs. 3,011,043 baseline YOLOv8n, nc=1).

3. Grid-Relative Continuous Bounding Box Decoding (Eqs. 5-6):
     (b_hat_x, b_hat_y) = ((x_grid + 0.5 + t_hat_x) * s_k / W_in, (y_grid + 0.5 + t_hat_y) * s_k / H_in)
     (b_hat_w, b_hat_h) = clip(s_k * (exp(t_hat_w), exp(t_hat_h)) / (W_in, H_in), 1e-4, 1.0)
   where s_k in {8, 16, 32} is the scale stride.
"""

import math
from typing import Tuple, List, Dict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from .gegenbauer_activation import GegenbauerActivation
except (ImportError, ValueError):
    from gegenbauer_activation import GegenbauerActivation


# ==============================================================================
# Structural Layer Abstractions for Standalone & PyTorch Execution
# ==============================================================================

class ParameterRef:
    """Lightweight parameter representation holding name and number of elements."""
    def __init__(self, name: str, numel: int):
        self.name = name
        self._numel = numel

    def numel(self) -> int:
        return self._numel


class ModuleRef:
    """Base module providing recursive parameter registration and counting."""
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def register_parameter(self, name: str, numel: int):
        self._parameters[name] = ParameterRef(name, numel)

    def __setattr__(self, name, value):
        if isinstance(value, ModuleRef):
            if '_modules' not in self.__dict__:
                self.__dict__['_modules'] = {}
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> List[ParameterRef]:
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ==============================================================================
# Gegenbauer-YOLOv8 Components (Dual-Mode: PyTorch nn.Module & Parameter Accounting)
# ==============================================================================

if HAS_TORCH:
    class Conv_GB_Layer(nn.Module):
        """Convolution + BatchNorm + Gegenbauer Activation Block (PyTorch)."""
        def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, basis_type: str = "gegenbauer"):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, padding=k//2, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = GegenbauerActivation(num_channels=c2, basis_type=basis_type)

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

    class Conv_SiLU_Layer(nn.Module):
        """Standard Convolution + BatchNorm + SiLU (PyTorch)."""
        def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1):
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, padding=k//2, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())

    class Bottleneck_GB_Layer(nn.Module):
        """Residual Bottleneck with Gegenbauer Activations (PyTorch)."""
        def __init__(self, c1: int, c2: int, shortcut: bool = True, basis_type: str = "gegenbauer"):
            super().__init__()
            self.cv1 = Conv_GB_Layer(c1, c2, k=3, basis_type=basis_type)
            self.cv2 = Conv_GB_Layer(c2, c2, k=3, basis_type=basis_type)
            self.shortcut = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))

        def count_parameters(self) -> int:
            return self.cv1.count_parameters() + self.cv2.count_parameters()

    class Bottleneck_SiLU_Layer(nn.Module):
        """Residual Bottleneck with SiLU Activations (PyTorch)."""
        def __init__(self, c1: int, c2: int, shortcut: bool = True):
            super().__init__()
            self.cv1 = Conv_SiLU_Layer(c1, c2, k=3)
            self.cv2 = Conv_SiLU_Layer(c2, c2, k=3)
            self.shortcut = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))

        def count_parameters(self) -> int:
            return self.cv1.count_parameters() + self.cv2.count_parameters()

    class C2f_GB_Layer(nn.Module):
        """C2f Module with Learnable Gegenbauer Activations (PyTorch)."""
        def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, basis_type: str = "gegenbauer"):
            super().__init__()
            self.c = int(c2 * 0.5)
            self.cv1 = Conv_GB_Layer(c1, 2 * self.c, 1, basis_type=basis_type)
            self.cv2 = Conv_GB_Layer((2 + n) * self.c, c2, 1, basis_type=basis_type)
            self.m = nn.ModuleList([Bottleneck_GB_Layer(self.c, self.c, shortcut=shortcut, basis_type=basis_type) for _ in range(n)])

        def forward(self, x):
            y = list(self.cv1(x).chunk(2, 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))

        def count_parameters(self) -> int:
            return self.cv1.count_parameters() + self.cv2.count_parameters() + sum(b.count_parameters() for b in self.m)

    class C2f_SiLU_Layer(nn.Module):
        """C2f Module with SiLU Activations (PyTorch)."""
        def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False):
            super().__init__()
            self.c = int(c2 * 0.5)
            self.cv1 = Conv_SiLU_Layer(c1, 2 * self.c, 1)
            self.cv2 = Conv_SiLU_Layer((2 + n) * self.c, c2, 1)
            self.m = nn.ModuleList([Bottleneck_SiLU_Layer(self.c, self.c, shortcut=shortcut) for _ in range(n)])

        def forward(self, x):
            y = list(self.cv1(x).chunk(2, 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))

        def count_parameters(self) -> int:
            return self.cv1.count_parameters() + self.cv2.count_parameters() + sum(b.count_parameters() for b in self.m)

    class SPPF_GB_Layer(nn.Module):
        """Spatial Pyramid Pooling Fast with Gegenbauer Activations (PyTorch)."""
        def __init__(self, c1: int, c2: int, k: int = 5, basis_type: str = "gegenbauer"):
            super().__init__()
            c_ = c1 // 2
            self.cv1 = Conv_GB_Layer(c1, c_, 1, basis_type=basis_type)
            self.cv2 = Conv_GB_Layer(c_ * 4, c2, 1, basis_type=basis_type)
            self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        def forward(self, x):
            x = self.cv1(x)
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

        def count_parameters(self) -> int:
            return self.cv1.count_parameters() + self.cv2.count_parameters()

    class SPPF_SiLU_Layer(nn.Module):
        """Spatial Pyramid Pooling Fast with SiLU Activations (PyTorch)."""
        def __init__(self, c1: int, c2: int, k: int = 5):
            super().__init__()
            c_ = c1 // 2
            self.cv1 = Conv_SiLU_Layer(c1, c_, 1)
            self.cv2 = Conv_SiLU_Layer(c_ * 4, c2, 1)
            self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        def forward(self, x):
            x = self.cv1(x)
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

        def count_parameters(self) -> int:
            return self.cv1.count_parameters() + self.cv2.count_parameters()
        def __init__(self, c1: int, c2: int, k: int = 5):
            super().__init__()
            c_ = c1 // 2
            self.cv1 = Conv_SiLU_Layer(c1, c_, 1)
            self.cv2 = Conv_SiLU_Layer(c_ * 4, c2, 1)
            self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        def forward(self, x):
            x = self.cv1(x)
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

        def count_parameters(self) -> int:
            return self.cv1.count_parameters() + self.cv2.count_parameters()

    class DirectCoordinateHeadLayer(nn.Module):
        """
        Direct 4-Channel Continuous Coordinate Regression Head (PyTorch).
        Outputs normalized bounding boxes [cx, cy, w, h] and classification logits.
        """
        def __init__(self, in_channels: int, stride: int = 8, num_classes: int = 1, hidden_channels: int = 64):
            super().__init__()
            self.stride = stride
            self.num_classes = num_classes

            # Coordinate regression subnet -> 4 continuous outputs
            self.reg_cv0 = Conv_SiLU_Layer(in_channels, hidden_channels, 3)
            self.reg_cv1 = Conv_SiLU_Layer(hidden_channels, hidden_channels, 3)
            self.reg_out = nn.Conv2d(hidden_channels, 4, 1)

            # Classification subnet -> 1 lesion logit
            self.cls_cv0 = Conv_SiLU_Layer(in_channels, hidden_channels, 3)
            self.cls_cv1 = Conv_SiLU_Layer(hidden_channels, hidden_channels, 3)
            self.cls_out = nn.Conv2d(hidden_channels, num_classes, 1)

        def forward(self, x):
            b, _, h, w = x.shape
            reg = self.reg_out(self.reg_cv1(self.reg_cv0(x)))
            cls_logits = self.cls_out(self.cls_cv1(self.cls_cv0(x)))

            # Grid relative decoding (Eqs. 5-6)
            yv, xv = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
            grid = torch.stack((xv, yv), 2).float().unsqueeze(0)  # (1, h, w, 2)

            t_xy = reg[:, :2].permute(0, 2, 3, 1)
            t_wh = reg[:, 2:].permute(0, 2, 3, 1)

            cx = (grid[..., 0:1] + 0.5 + t_xy[..., 0:1]) * (self.stride / (w * self.stride))
            cy = (grid[..., 1:2] + 0.5 + t_xy[..., 1:2]) * (self.stride / (h * self.stride))
            bw = torch.clamp((self.stride * torch.exp(t_wh[..., 0:1])) / (w * self.stride), 1e-4, 1.0)
            bh = torch.clamp((self.stride * torch.exp(t_wh[..., 1:2])) / (h * self.stride), 1e-4, 1.0)

            boxes = torch.cat((cx, cy, bw, bh), dim=-1).reshape(b, -1, 4)
            scores = torch.sigmoid(cls_logits.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes))

            return boxes, scores, reg

        def count_parameters(self) -> int:
            return (self.reg_cv0.count_parameters() + self.reg_cv1.count_parameters() +
                    sum(p.numel() for p in self.reg_out.parameters()) +
                    self.cls_cv0.count_parameters() + self.cls_cv1.count_parameters() +
                    sum(p.numel() for p in self.cls_out.parameters()))

    class DFLHeadLayer(nn.Module):
        """Standard 64-Channel Discrete DFL Head (PyTorch)."""
        def __init__(self, in_channels: int, stride: int = 8, num_classes: int = 1, hidden_channels: int = 64):
            super().__init__()
            self.stride = stride
            self.num_classes = num_classes
            self.reg_cv0 = Conv_SiLU_Layer(in_channels, hidden_channels, 3)
            self.reg_cv1 = Conv_SiLU_Layer(hidden_channels, hidden_channels, 3)
            self.reg_out = nn.Conv2d(hidden_channels, 64, 1)
            self.cls_cv0 = Conv_SiLU_Layer(in_channels, hidden_channels, 3)
            self.cls_cv1 = Conv_SiLU_Layer(hidden_channels, hidden_channels, 3)
            self.cls_out = nn.Conv2d(hidden_channels, num_classes, 1)

        def forward(self, x):
            b, _, h, w = x.shape
            reg = self.reg_out(self.reg_cv1(self.reg_cv0(x)))  # (b, 64, h, w)
            cls_logits = self.cls_out(self.cls_cv1(self.cls_cv0(x)))  # (b, num_classes, h, w)

            # DFL expectation: 16 probability bins per coordinate
            reg_4x16 = reg.view(b, 4, 16, h, w).permute(0, 1, 3, 4, 2)  # (b, 4, h, w, 16)
            prob = F.softmax(reg_4x16, dim=-1)
            proj = torch.arange(16, dtype=torch.float32, device=x.device)
            dist = torch.sum(prob * proj, dim=-1)  # (b, 4, h, w)

            yv, xv = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
            grid = torch.stack((xv, yv), 2).float().unsqueeze(0)  # (1, h, w, 2)

            t_xy = dist[:, :2].permute(0, 2, 3, 1)
            t_wh = dist[:, 2:].permute(0, 2, 3, 1)

            cx = (grid[..., 0:1] + 0.5 + t_xy[..., 0:1]) * (self.stride / (w * self.stride))
            cy = (grid[..., 1:2] + 0.5 + t_xy[..., 1:2]) * (self.stride / (h * self.stride))
            bw = torch.clamp((self.stride * torch.exp(t_wh[..., 0:1] * 0.10)) / (w * self.stride), 1e-4, 1.0)
            bh = torch.clamp((self.stride * torch.exp(t_wh[..., 1:2] * 0.10)) / (h * self.stride), 1e-4, 1.0)

            boxes = torch.cat((cx, cy, bw, bh), dim=-1).reshape(b, -1, 4)
            scores = torch.sigmoid(cls_logits.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes))
            return boxes, scores, reg

        def count_parameters(self) -> int:
            return (self.reg_cv0.count_parameters() + self.reg_cv1.count_parameters() +
                    sum(p.numel() for p in self.reg_out.parameters()) +
                    self.cls_cv0.count_parameters() + self.cls_cv1.count_parameters() +
                    sum(p.numel() for p in self.cls_out.parameters()))

else:
    # Standalone structural representations for environments without PyTorch
    class Conv_GB_Layer(ModuleRef):
        def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1):
            super().__init__()
            self.register_parameter('conv_w', c2 * c1 * k * k)
            self.register_parameter('bn_w', c2)
            self.register_parameter('bn_b', c2)
            self.register_parameter('act_coeffs', c2 * 4)
            self.register_parameter('act_beta', c2)

    class Conv_SiLU_Layer(ModuleRef):
        def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1):
            super().__init__()
            self.register_parameter('conv_w', c2 * c1 * k * k)
            self.register_parameter('bn_w', c2)
            self.register_parameter('bn_b', c2)

    class Bottleneck_GB_Layer(ModuleRef):
        def __init__(self, c1: int, c2: int, shortcut: bool = True):
            super().__init__()
            self.cv1 = Conv_GB_Layer(c1, c2, k=3)
            self.cv2 = Conv_GB_Layer(c2, c2, k=3)

    class Bottleneck_SiLU_Layer(ModuleRef):
        def __init__(self, c1: int, c2: int, shortcut: bool = True):
            super().__init__()
            self.cv1 = Conv_SiLU_Layer(c1, c2, k=3)
            self.cv2 = Conv_SiLU_Layer(c2, c2, k=3)

    class C2f_GB_Layer(ModuleRef):
        def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False):
            super().__init__()
            c_ = int(c2 * 0.5)
            self.cv1 = Conv_GB_Layer(c1, 2 * c_, 1)
            self.cv2 = Conv_GB_Layer((2 + n) * c_, c2, 1)
            for i in range(n):
                setattr(self, f'b_{i}', Bottleneck_GB_Layer(c_, c_, shortcut=shortcut))

    class C2f_SiLU_Layer(ModuleRef):
        def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False):
            super().__init__()
            c_ = int(c2 * 0.5)
            self.cv1 = Conv_SiLU_Layer(c1, 2 * c_, 1)
            self.cv2 = Conv_SiLU_Layer((2 + n) * c_, c2, 1)
            for i in range(n):
                setattr(self, f'b_{i}', Bottleneck_SiLU_Layer(c_, c_, shortcut=shortcut))

    class SPPF_GB_Layer(ModuleRef):
        def __init__(self, c1: int, c2: int, k: int = 5):
            super().__init__()
            c_ = c1 // 2
            self.cv1 = Conv_GB_Layer(c1, c_, 1)
            self.cv2 = Conv_GB_Layer(c_ * 4, c2, 1)

    class SPPF_SiLU_Layer(ModuleRef):
        def __init__(self, c1: int, c2: int, k: int = 5):
            super().__init__()
            c_ = c1 // 2
            self.cv1 = Conv_SiLU_Layer(c1, c_, 1)
            self.cv2 = Conv_SiLU_Layer(c_ * 4, c2, 1)

    class DirectCoordinateHeadLayer(ModuleRef):
        def __init__(self, in_channels: int, stride: int = 8, num_classes: int = 1, hidden_channels: int = 64):
            super().__init__()
            self.stride = stride
            self.num_classes = num_classes
            self.reg_cv0 = Conv_SiLU_Layer(in_channels, hidden_channels, 3)
            self.reg_cv1 = Conv_SiLU_Layer(hidden_channels, hidden_channels, 3)
            self.register_parameter('reg_cv2_w', hidden_channels * 4 * 1 * 1)
            self.register_parameter('reg_cv2_b', 4)
            self.cls_cv0 = Conv_SiLU_Layer(in_channels, hidden_channels, 3)
            self.cls_cv1 = Conv_SiLU_Layer(hidden_channels, hidden_channels, 3)
            self.register_parameter('cls_cv2_w', hidden_channels * num_classes * 1 * 1)
            self.register_parameter('cls_cv2_b', num_classes)

    class DFLHeadLayer(ModuleRef):
        def __init__(self, in_channels: int, stride: int = 8, num_classes: int = 1, hidden_channels: int = 64):
            super().__init__()
            self.stride = stride
            self.num_classes = num_classes
            self.reg_cv0 = Conv_SiLU_Layer(in_channels, hidden_channels, 3)
            self.reg_cv1 = Conv_SiLU_Layer(hidden_channels, hidden_channels, 3)
            self.register_parameter('reg_cv2_w', hidden_channels * 64 * 1 * 1)
            self.register_parameter('reg_cv2_b', 64)
            self.cls_cv0 = Conv_SiLU_Layer(in_channels, hidden_channels, 3)
            self.cls_cv1 = Conv_SiLU_Layer(hidden_channels, hidden_channels, 3)
            self.register_parameter('cls_cv2_w', hidden_channels * num_classes * 1 * 1)
            self.register_parameter('cls_cv2_b', num_classes)


# ==============================================================================
# Complete Detectors
# ==============================================================================

class GegenbauerYOLOv8(nn.Module if HAS_TORCH else ModuleRef):
    """
    Proposed Detector: YOLOv8n with Channel-Wise Gegenbauer / Polynomial Activations
    and Direct 4-Coordinate Continuous Regression Head (or 64-channel discrete DFL head for ablation).
    Total Parameters: 3,021,487 (+10,444 params over baseline YOLOv8n).
    """
    def __init__(self, num_classes: int = 1, basis_type: str = "gegenbauer", head_type: str = "4ch"):
        super().__init__()
        self.num_classes = num_classes
        self.basis_type = basis_type.lower()
        self.head_type = head_type.lower()

        # Backbone (26 layers, 2,512 channels, Polynomial activations)
        if HAS_TORCH:
            self.b0 = Conv_GB_Layer(3, 16, 3, 2, basis_type=self.basis_type)
            self.b1 = Conv_GB_Layer(16, 32, 3, 2, basis_type=self.basis_type)
            self.b2 = C2f_GB_Layer(32, 32, n=1, shortcut=True, basis_type=self.basis_type)
            self.b3 = Conv_GB_Layer(32, 64, 3, 2, basis_type=self.basis_type)
            self.b4 = C2f_GB_Layer(64, 64, n=2, shortcut=True, basis_type=self.basis_type)
            self.b5 = Conv_GB_Layer(64, 128, 3, 2, basis_type=self.basis_type)
            self.b6 = C2f_GB_Layer(128, 128, n=2, shortcut=True, basis_type=self.basis_type)
            self.b7 = Conv_GB_Layer(128, 256, 3, 2, basis_type=self.basis_type)
            self.b8 = C2f_GB_Layer(256, 256, n=1, shortcut=True, basis_type=self.basis_type)
            self.b9 = SPPF_GB_Layer(256, 256, 5, basis_type=self.basis_type)

            # Neck (19 layers, 1,920 channels, Polynomial activations)
            self.n12 = C2f_GB_Layer(128 + 256, 128, n=1, shortcut=False, basis_type=self.basis_type)
            self.n15 = C2f_GB_Layer(64 + 128, 64, n=1, shortcut=False, basis_type=self.basis_type)
            self.n16 = Conv_GB_Layer(64, 64, 3, 2, basis_type=self.basis_type)
            self.n18 = C2f_GB_Layer(64 + 128, 128, n=1, shortcut=False, basis_type=self.basis_type)
            self.n19 = Conv_GB_Layer(128, 128, 3, 2, basis_type=self.basis_type)
            self.n21 = C2f_GB_Layer(128 + 256, 256, n=1, shortcut=False, basis_type=self.basis_type)

            # Detection Heads (P3, P4, P5)
            if self.head_type == "64ch":
                self.head_p3 = DFLHeadLayer(64, stride=8, num_classes=num_classes)
                self.head_p4 = DFLHeadLayer(128, stride=16, num_classes=num_classes)
                self.head_p5 = DFLHeadLayer(256, stride=32, num_classes=num_classes)
            else:
                self.head_p3 = DirectCoordinateHeadLayer(64, stride=8, num_classes=num_classes)
                self.head_p4 = DirectCoordinateHeadLayer(128, stride=16, num_classes=num_classes)
                self.head_p5 = DirectCoordinateHeadLayer(256, stride=32, num_classes=num_classes)
        else:
            self.backbone = ModuleRef()
            self.backbone.b0 = Conv_GB_Layer(3, 16, 3, 2)
            self.backbone.b1 = Conv_GB_Layer(16, 32, 3, 2)
            self.backbone.b2 = C2f_GB_Layer(32, 32, n=1, shortcut=True)
            self.backbone.b3 = Conv_GB_Layer(32, 64, 3, 2)
            self.backbone.b4 = C2f_GB_Layer(64, 64, n=2, shortcut=True)
            self.backbone.b5 = Conv_GB_Layer(64, 128, 3, 2)
            self.backbone.b6 = C2f_GB_Layer(128, 128, n=2, shortcut=True)
            self.backbone.b7 = Conv_GB_Layer(128, 256, 3, 2)
            self.backbone.b8 = C2f_GB_Layer(256, 256, n=1, shortcut=True)
            self.backbone.b9 = SPPF_GB_Layer(256, 256, 5)

            self.neck = ModuleRef()
            self.neck.n12 = C2f_GB_Layer(128 + 256, 128, n=1, shortcut=False)
            self.neck.n15 = C2f_GB_Layer(64 + 128, 64, n=1, shortcut=False)
            self.neck.n16 = Conv_GB_Layer(64, 64, 3, 2)
            self.neck.n18 = C2f_GB_Layer(64 + 128, 128, n=1, shortcut=False)
            self.neck.n19 = Conv_GB_Layer(128, 128, 3, 2)
            self.neck.n21 = C2f_GB_Layer(128 + 256, 256, n=1, shortcut=False)

            self.head = ModuleRef()
            if self.head_type == "64ch":
                self.head.p3 = DFLHeadLayer(64, stride=8, num_classes=num_classes)
                self.head.p4 = DFLHeadLayer(128, stride=16, num_classes=num_classes)
                self.head.p5 = DFLHeadLayer(256, stride=32, num_classes=num_classes)
            else:
                self.head.p3 = DirectCoordinateHeadLayer(64, stride=8, num_classes=num_classes)
                self.head.p4 = DirectCoordinateHeadLayer(128, stride=16, num_classes=num_classes)
                self.head.p5 = DirectCoordinateHeadLayer(256, stride=32, num_classes=num_classes)

    def forward(self, x):
        if not HAS_TORCH:
            raise RuntimeError("Live forward pass requires PyTorch. Install PyTorch to execute model inference.")

        # Backbone forward
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        p3 = self.b4(x)  # stride 8, 64 channels
        x = self.b5(p3)
        p4 = self.b6(x)  # stride 16, 128 channels
        x = self.b7(p4)
        x = self.b8(x)
        p5 = self.b9(x)  # stride 32, 256 channels

        # Neck forward (PANet)
        u5 = F.interpolate(p5, scale_factor=2, mode="nearest")
        n12 = self.n12(torch.cat([u5, p4], dim=1))
        u4 = F.interpolate(n12, scale_factor=2, mode="nearest")
        n15 = self.n15(torch.cat([u4, p3], dim=1))  # P3 feature
        d3 = self.n16(n15)
        n18 = self.n18(torch.cat([d3, n12], dim=1))  # P4 feature
        d4 = self.n19(n18)
        n21 = self.n21(torch.cat([d4, p5], dim=1))  # P5 feature

        # Direct heads forward
        b3, s3, r3 = self.head_p3(n15)
        b4, s4, r4 = self.head_p4(n18)
        b5, s5, r5 = self.head_p5(n21)

        all_boxes = torch.cat((b3, b4, b5), dim=1)
        all_scores = torch.cat((s3, s4, s5), dim=1)

        return {
            "boxes": all_boxes,
            "scores": all_scores,
            "raw_reg": (r3, r4, r5)
        }

    def backbone_parameters(self) -> int:
        if HAS_TORCH:
            b_mods = [self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8, self.b9]
            return sum(m.count_parameters() for m in b_mods)
        return self.backbone.count_parameters()

    def neck_parameters(self) -> int:
        if HAS_TORCH:
            n_mods = [self.n12, self.n15, self.n16, self.n18, self.n19, self.n21]
            return sum(m.count_parameters() for m in n_mods)
        return self.neck.count_parameters()

    def head_parameters(self) -> int:
        if HAS_TORCH:
            h_mods = [self.head_p3, self.head_p4, self.head_p5]
            return sum(m.count_parameters() for m in h_mods)
        return self.head.count_parameters()

    def count_parameters(self) -> int:
        return self.backbone_parameters() + self.neck_parameters() + self.head_parameters()


class BaselineYOLOv8(nn.Module if HAS_TORCH else ModuleRef):
    """
    Standard Baseline YOLOv8n Detector (SiLU activations + 64-channel discrete DFL head or 4-channel direct head).
    Total Parameters: 3,011,043.
    """
    def __init__(self, num_classes: int = 1, head_type: str = "64ch"):
        super().__init__()
        self.num_classes = num_classes
        self.head_type = head_type.lower()

        if HAS_TORCH:
            self.b0 = Conv_SiLU_Layer(3, 16, 3, 2)
            self.b1 = Conv_SiLU_Layer(16, 32, 3, 2)
            self.b2 = C2f_SiLU_Layer(32, 32, n=1, shortcut=True)
            self.b3 = Conv_SiLU_Layer(32, 64, 3, 2)
            self.b4 = C2f_SiLU_Layer(64, 64, n=2, shortcut=True)
            self.b5 = Conv_SiLU_Layer(64, 128, 3, 2)
            self.b6 = C2f_SiLU_Layer(128, 128, n=2, shortcut=True)
            self.b7 = Conv_SiLU_Layer(128, 256, 3, 2)
            self.b8 = C2f_SiLU_Layer(256, 256, n=1, shortcut=True)
            self.b9 = SPPF_SiLU_Layer(256, 256, 5)

            self.n12 = C2f_SiLU_Layer(128 + 256, 128, n=1, shortcut=False)
            self.n15 = C2f_SiLU_Layer(64 + 128, 64, n=1, shortcut=False)
            self.n16 = Conv_SiLU_Layer(64, 64, 3, 2)
            self.n18 = C2f_SiLU_Layer(64 + 128, 128, n=1, shortcut=False)
            self.n19 = Conv_SiLU_Layer(128, 128, 3, 2)
            self.n21 = C2f_SiLU_Layer(128 + 256, 256, n=1, shortcut=False)

            if self.head_type == "4ch":
                self.head_p3 = DirectCoordinateHeadLayer(64, stride=8, num_classes=num_classes)
                self.head_p4 = DirectCoordinateHeadLayer(128, stride=16, num_classes=num_classes)
                self.head_p5 = DirectCoordinateHeadLayer(256, stride=32, num_classes=num_classes)
            else:
                self.head_p3 = DFLHeadLayer(64, stride=8, num_classes=num_classes)
                self.head_p4 = DFLHeadLayer(128, stride=16, num_classes=num_classes)
                self.head_p5 = DFLHeadLayer(256, stride=32, num_classes=num_classes)
                self.dfl_conv_weight = nn.Parameter(torch.zeros(16))
        else:
            self.backbone = ModuleRef()
            self.backbone.b0 = Conv_SiLU_Layer(3, 16, 3, 2)
            self.backbone.b1 = Conv_SiLU_Layer(16, 32, 3, 2)
            self.backbone.b2 = C2f_SiLU_Layer(32, 32, n=1, shortcut=True)
            self.backbone.b3 = Conv_SiLU_Layer(32, 64, 3, 2)
            self.backbone.b4 = C2f_SiLU_Layer(64, 64, n=2, shortcut=True)
            self.backbone.b5 = Conv_SiLU_Layer(64, 128, 3, 2)
            self.backbone.b6 = C2f_SiLU_Layer(128, 128, n=2, shortcut=True)
            self.backbone.b7 = Conv_SiLU_Layer(128, 256, 3, 2)
            self.backbone.b8 = C2f_SiLU_Layer(256, 256, n=1, shortcut=True)
            self.backbone.b9 = SPPF_SiLU_Layer(256, 256, 5)

            self.neck = ModuleRef()
            self.neck.n12 = C2f_SiLU_Layer(128 + 256, 128, n=1, shortcut=False)
            self.neck.n15 = C2f_SiLU_Layer(64 + 128, 64, n=1, shortcut=False)
            self.neck.n16 = Conv_SiLU_Layer(64, 64, 3, 2)
            self.neck.n18 = C2f_SiLU_Layer(64 + 128, 128, n=1, shortcut=False)
            self.neck.n19 = Conv_SiLU_Layer(128, 128, 3, 2)
            self.neck.n21 = C2f_SiLU_Layer(128 + 256, 256, n=1, shortcut=False)

            self.head = ModuleRef()
            if self.head_type == "4ch":
                self.head.p3 = DirectCoordinateHeadLayer(64, stride=8, num_classes=num_classes)
                self.head.p4 = DirectCoordinateHeadLayer(128, stride=16, num_classes=num_classes)
                self.head.p5 = DirectCoordinateHeadLayer(256, stride=32, num_classes=num_classes)
            else:
                self.head.p3 = DFLHeadLayer(64, stride=8, num_classes=num_classes)
                self.head.p4 = DFLHeadLayer(128, stride=16, num_classes=num_classes)
                self.head.p5 = DFLHeadLayer(256, stride=32, num_classes=num_classes)

            self.neck = ModuleRef()
            self.neck.n12 = C2f_SiLU_Layer(128 + 256, 128, n=1, shortcut=False)
            self.neck.n15 = C2f_SiLU_Layer(64 + 128, 64, n=1, shortcut=False)
            self.neck.n16 = Conv_SiLU_Layer(64, 64, 3, 2)
            self.neck.n18 = C2f_SiLU_Layer(64 + 128, 128, n=1, shortcut=False)
            self.neck.n19 = Conv_SiLU_Layer(128, 128, 3, 2)
            self.neck.n21 = C2f_SiLU_Layer(128 + 256, 256, n=1, shortcut=False)

            self.head = ModuleRef()
            self.head.p3 = DFLHeadLayer(64, stride=8, num_classes=num_classes)
            self.head.p4 = DFLHeadLayer(128, stride=16, num_classes=num_classes)
            self.head.p5 = DFLHeadLayer(256, stride=32, num_classes=num_classes)
            self.head.register_parameter('dfl_conv_weight', 16)

    def backbone_parameters(self) -> int:
        if HAS_TORCH:
            b_mods = [self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8, self.b9]
            return sum(m.count_parameters() for m in b_mods)
        return self.backbone.count_parameters()

    def neck_parameters(self) -> int:
        if HAS_TORCH:
            n_mods = [self.n12, self.n15, self.n16, self.n18, self.n19, self.n21]
            return sum(m.count_parameters() for m in n_mods)
        return self.neck.count_parameters()

    def head_parameters(self) -> int:
        if HAS_TORCH:
            h_mods = [self.head_p3, self.head_p4, self.head_p5]
            return sum(m.count_parameters() for m in h_mods) + 16
        return self.head.count_parameters()

    def count_parameters(self) -> int:
        return self.backbone_parameters() + self.neck_parameters() + self.head_parameters()


def verify_parameter_accounting() -> Dict[str, int]:
    """
    Instantiates both GegenbauerYOLOv8 and BaselineYOLOv8 models and verifies
    exact layer-by-layer architectural parameter sums against Table 1 of the manuscript.
    """
    model = GegenbauerYOLOv8(num_classes=1)
    base_model = BaselineYOLOv8(num_classes=1)

    prop_backbone = model.backbone_parameters()
    prop_neck = model.neck_parameters()
    prop_head = model.head_parameters()
    prop_total = model.count_parameters()

    base_backbone = base_model.backbone_parameters()
    base_neck = base_model.neck_parameters()
    base_head = base_model.head_parameters()
    base_total = base_model.count_parameters()

    delta_total = prop_total - base_total
    delta_pct = (delta_total / base_total) * 100.0

    return {
        "base_backbone": base_backbone,
        "base_neck": base_neck,
        "base_head": base_head,
        "base_total": base_total,
        "prop_backbone": prop_backbone,
        "prop_neck": prop_neck,
        "prop_head": prop_head,
        "prop_total": prop_total,
        "delta_total": delta_total,
        "delta_pct": delta_pct
    }


if __name__ == "__main__":
    stats = verify_parameter_accounting()
    print("=" * 85)
    print("ARCHITECTURAL PARAMETER ACCOUNTING (TABLE 1)")
    print("=" * 85)
    print(f"{'Component':<30} {'Baseline (SiLU)':<18} {'Proposed (GB)':<18} {'Net Delta':<15}")
    print("-" * 85)
    print(f"{'Backbone (26 layers)':<30} {stats['base_backbone']:,} {stats['prop_backbone']:,} {stats['prop_backbone'] - stats['base_backbone']:+,}")
    print(f"{'Neck PANet (19 layers)':<30} {stats['base_neck']:,} {stats['prop_neck']:,} {stats['prop_neck'] - stats['base_neck']:+,}")
    print(f"{'Detection Head (3 scales)':<30} {stats['base_head']:,} {stats['prop_head']:,} {stats['prop_head'] - stats['base_head']:+,}")
    print("-" * 85)
    print(f"{'TOTAL DETECTOR':<30} {stats['base_total']:,} (3.011M)  {stats['prop_total']:,} (3.021M)  {stats['delta_total']:+,} (+{stats['delta_pct']:.2f}%)")
    print("=" * 85)

    assert stats['base_total'] == 3_011_043, f"Baseline count mismatch: {stats['base_total']} != 3,011,043"
    assert stats['prop_total'] == 3_021_487, f"Proposed count mismatch: {stats['prop_total']} != 3,021,487"
    assert stats['delta_total'] == 10_444, f"Delta mismatch: {stats['delta_total']} != 10,444"
    print("[+] All model parameters verified strictly via layer-by-layer programmatic summation.")
