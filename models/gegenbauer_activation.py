#!/usr/bin/env python3
"""
Channel-Wise Learnable Gegenbauer Polynomial Activation Function
================================================================
Reference implementation for:
"Learnable Gegenbauer Activations and Direct Coordinate Regression for Lightweight Lesion Detection"
Authors: M. Deepthi, G. N. V. R. Vikram (2026)

Mathematical Formulations:
--------------------------
1. Hyperbolic Tangent Squashing:
     x_hat = tanh(x) in (-1, 1)
   Ensures polynomial inputs remain strictly bounded within the orthogonality interval [-1, 1].

2. Ultraspherical Gegenbauer Polynomial Basis (Order K=3):
     GB_0^(beta)(x_hat) = 1
     GB_1^(beta)(x_hat) = 2 * beta * x_hat
     GB_2^(beta)(x_hat) = 2 * beta * (beta + 1) * x_hat^2 - beta
     GB_3^(beta)(x_hat) = (4 * beta * (beta + 1) * (beta + 2) * x_hat^3 - 6 * beta * (beta + 1) * x_hat) / 3

3. Linear-in-Parameters (LIP) Channel-Wise Expansion:
     f_tilde_c(x_hat) = sum_{k=0}^3 a_{c,k} * GB_k^(beta_c)(x_hat)
   where a_{c,k} are learnable coefficients and beta_c > 0 is a learnable shape parameter.

4. Initialization (Section 3.1):
   beta_c is initialized to 0.5 (Legendre baseline).
   Coefficients a_{c,k} are initialized from N(0, 1 / (K + 1)), with K = 3 (std = 0.50).

5. Canonical Horner's Polynomial Representation:
   To evaluate efficiently in 3 multiply-accumulates without recursive overhead:
     f_tilde_c(x_hat) = A_0 + x_hat * (A_1 + x_hat * (A_2 + x_hat * A_3))
   where:
     A_0 = a_0 - a_2 * beta
     A_1 = 2 * beta * a_1 - 2 * beta * (beta + 1) * a_3
     A_2 = 2 * beta * (beta + 1) * a_2
     A_3 = (4/3) * beta * (beta + 1) * (beta + 2) * a_3

6. Analytical Lipschitz Derivative Bound (Proposition 1):
     Lip(f_tilde_c) <= 2 * beta_c * sum_{k=1}^3 |a_{c,k}| * binom(k + 2*beta_c, k - 1)
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


class GegenbauerActivation(nn.Module if HAS_TORCH else object):
    """
    Channel-wise Polynomial Activation Layer supporting:
    - Gegenbauer (GB3): Learnable shape parameter beta_c > 0 (Proposed)
    - Legendre (P3): Fixed beta_c = 0.5
    - Chebyshev (T3): First-kind Chebyshev orthogonal polynomials
    - Monomial: Standard cubic power basis (1, x, x^2, x^3)

    Args:
        num_channels (int): Number of feature channels C.
        init_beta (float): Initial value of shape parameter beta (default: 0.5, Legendre baseline).
        clamp_beta (tuple): Min/max projection bounds for beta (default: (0.01, 5.0)).
        init_mode (str): 'paper_gaussian' (default: N(0, 1/(K+1)) as in Sec 3.1) or 'identity_slope'.
        basis_type (str): 'gegenbauer', 'legendre', 'chebyshev', or 'monomial'.
    """
    def __init__(self, num_channels: int = None, init_beta: float = 0.5,
                 clamp_beta: tuple = (0.01, 5.0), init_mode: str = "paper_gaussian",
                 channels: int = None, K: int = 3, basis_type: str = "gegenbauer"):
        if HAS_TORCH:
            super().__init__()
        if num_channels is None and channels is not None:
            num_channels = channels
        elif num_channels is None:
            num_channels = 64
        self.num_channels = num_channels
        self.clamp_beta = clamp_beta
        self.k_order = K
        self.init_mode = init_mode
        self.basis_type = basis_type.lower()

        if HAS_TORCH:
            if init_mode == "paper_gaussian":
                # Section 3.1: a_{c, k} ~ N(0, 1 / (K + 1))
                std = math.sqrt(1.0 / (self.k_order + 1))  # 0.50
                a_init = torch.randn(num_channels, self.k_order + 1) * std
            else:
                a_init = torch.zeros(num_channels, self.k_order + 1)
                a_init[:, 1] = 1.0 / (2.0 * init_beta)

            self.coeffs = nn.Parameter(a_init)

            if self.basis_type == "gegenbauer":
                # Unconstrained beta_raw parameterization: beta = softplus(beta_raw) + 1e-4
                raw_val = math.log(math.exp(init_beta - 1e-4) - 1.0) if init_beta > 1e-4 else 0.0
                self.beta_raw = nn.Parameter(torch.full((num_channels,), raw_val, dtype=torch.float32))
            else:
                self.register_buffer("fixed_beta", torch.full((num_channels,), init_beta, dtype=torch.float32))
        else:
            std = math.sqrt(1.0 / (self.k_order + 1))
            self.coeffs = np.random.randn(num_channels, self.k_order + 1).astype(np.float32) * std
            self.beta_val = np.full((num_channels,), init_beta, dtype=np.float32)

    @property
    def beta(self):
        """Returns the strictly positive, bounded shape parameter beta > 0."""
        if HAS_TORCH:
            if self.basis_type == "gegenbauer":
                b = F.softplus(self.beta_raw) + 1e-4
                if self.clamp_beta:
                    b = torch.clamp(b, self.clamp_beta[0], self.clamp_beta[1])
                return b
            else:
                return self.fixed_beta
        else:
            return np.clip(self.beta_val, self.clamp_beta[0], self.clamp_beta[1])

    def get_canonical_coeffs(self):
        """
        Computes canonical polynomial coefficients [A0, A1, A2, A3] for Horner evaluation:
        f_tilde(x_hat) = A0 + x_hat * (A1 + x_hat * (A2 + x_hat * A3))
        """
        a0, a1, a2, a3 = self.coeffs[:, 0], self.coeffs[:, 1], self.coeffs[:, 2], self.coeffs[:, 3]
        basis = self.basis_type

        if basis == "monomial":
            return a0, a1, a2, a3

        if basis == "chebyshev":
            A0 = a0 - a2
            A1 = a1 - 3.0 * a3
            A2 = 2.0 * a2
            A3 = 4.0 * a3
            return A0, A1, A2, A3

        if basis == "legendre":
            A0 = a0 - 0.5 * a2
            A1 = a1 - 1.5 * a3
            A2 = 1.5 * a2
            A3 = 2.5 * a3
            return A0, A1, A2, A3

        # Default: Gegenbauer (beta_c learnable)
        beta = self.beta
        if HAS_TORCH:
            beta_p1 = beta + 1.0
            beta_p2 = beta + 2.0
            A0 = a0 - a2 * beta
            A1 = 2.0 * beta * a1 - 2.0 * beta * beta_p1 * a3
            A2 = 2.0 * beta * beta_p1 * a2
            A3 = (4.0 / 3.0) * beta * beta_p1 * beta_p2 * a3
            return A0, A1, A2, A3
        else:
            b_p1 = beta + 1.0
            b_p2 = beta + 2.0
            A0 = a0 - a2 * beta
            A1 = 2.0 * beta * a1 - 2.0 * beta * b_p1 * a3
            A2 = 2.0 * beta * b_p1 * a2
            A3 = (4.0 / 3.0) * beta * b_p1 * b_p2 * a3
            return A0, A1, A2, A3

    def forward(self, x):
        """
        Forward activation using Horner's polynomial evaluation method.
        """
        if HAS_TORCH:
            x_hat = torch.tanh(x)
            A0, A1, A2, A3 = self.get_canonical_coeffs()

            if x.dim() == 4:
                shape = (1, self.num_channels, 1, 1)
            elif x.dim() == 2:
                shape = (1, self.num_channels)
            else:
                shape = (1, self.num_channels) + (1,) * (x.dim() - 2)

            a0 = A0.view(shape)
            a1 = A1.view(shape)
            a2 = A2.view(shape)
            a3 = A3.view(shape)

            return a0 + x_hat * (a1 + x_hat * (a2 + x_hat * a3))
        else:
            x_hat = np.tanh(x)
            A0, A1, A2, A3 = self.get_canonical_coeffs()
            shape = [1] * x_hat.ndim
            shape[1] = self.num_channels
            a0 = A0.reshape(shape)
            a1 = A1.reshape(shape)
            a2 = A2.reshape(shape)
            a3 = A3.reshape(shape)
            return a0 + x_hat * (a1 + x_hat * (a2 + x_hat * a3))

    def compute_lipschitz_bound(self):
        """
        Computes the theoretical upper bound on the activation derivative (Proposition 1):
        Lip(f_tilde_c) <= 2 * beta_c * sum_{k=1}^3 |a_{c,k}| * binom(k + 2*beta_c, k - 1)
        """
        beta = self.beta
        a = self.coeffs
        if HAS_TORCH:
            b1 = torch.ones_like(beta)
            b2 = 2.0 + 2.0 * beta
            b3 = (3.0 + 2.0 * beta) * (1.0 + beta)

            term1 = torch.abs(a[:, 1]) * b1
            term2 = torch.abs(a[:, 2]) * b2
            term3 = torch.abs(a[:, 3]) * b3
            return 2.0 * beta * (term1 + term2 + term3)
        else:
            b1 = np.ones_like(beta)
            b2 = 2.0 + 2.0 * beta
            b3 = (3.0 + 2.0 * beta) * (1.0 + beta)
            term1 = np.abs(a[:, 1]) * b1
            term2 = np.abs(a[:, 2]) * b2
            term3 = np.abs(a[:, 3]) * b3
            return 2.0 * beta * (term1 + term2 + term3)


if __name__ == "__main__":
    layer = GegenbauerActivation(num_channels=64, init_beta=0.5, init_mode="paper_gaussian")
    dummy = np.random.randn(2, 64, 40, 40).astype(np.float32)
    out = layer.forward(dummy)
    print(f"[*] Layer test: input={dummy.shape} -> output={out.shape}")
    lip = layer.compute_lipschitz_bound()
    print(f"[*] Mean analytical Lipschitz bound (Proposition 1): {float(np.mean(lip)):.4f}")
    print("[+] GegenbauerActivation validated.")
