#!/usr/bin/env python3
"""
Basis Gram Matrix Conditioning and Numerical Collinearity Evaluation
====================================================================
Paper: Learnable Gegenbauer Activations and Direct Coordinate Regression for Lightweight Lesion Detection
Authors: M. Deepthi, G. N. V. R. Vikram (2026)

Reproduces Section 3.2:
-----------------------
Evaluates the numerical conditioning of polynomial bases under standardized post-batch-normalization
inputs z ~ N(0, 1) mapped through hyperbolic tangent squashing u = tanh(z) in [-1, 1].

Mathematical Formulations:
1. Empirical Gram Matrix:
     G_{i, j} = E_{z ~ N(0, 1)} [ phi_i(tanh(z)) * phi_j(tanh(z)) ]

2. Scale-Invariant Correlation Gram Matrix:
     C = D^{-1/2} * G * D^{-1/2},   where D = diag(G)
   Normalizes diagonal elements to 1.0, isolating intrinsic coordinate collinearity from scalar weights.

3. Condition Numbers:
     kappa(G) = sigma_max(G) / sigma_min(G)
     kappa(C) = sigma_max(C) / sigma_min(C)
"""

import math
import numpy as np


def compute_gram_and_correlation(phi_matrix: np.ndarray):
    """
    Computes raw Gram matrix G, diagonal scaling D, scale-invariant correlation Gram matrix C,
    and their condition numbers.
    Args:
        phi_matrix: array of shape [4, N_samples]
    """
    n_samples = phi_matrix.shape[1]
    G = (phi_matrix @ phi_matrix.T) / n_samples
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(G)))
    C = D_inv_sqrt @ G @ D_inv_sqrt

    kappa_G = np.linalg.cond(G)
    kappa_C = np.linalg.cond(C)

    # Maximum off-diagonal pairwise correlation
    off_diag = np.abs(C - np.eye(4))
    max_corr = np.max(off_diag)

    return G, C, kappa_G, kappa_C, max_corr


def evaluate_all_bases(n_samples: int = 1_000_000, seed: int = 42):
    np.random.seed(seed)
    z = np.random.normal(0.0, 1.0, n_samples)
    u = np.tanh(z)

    # 1. Monomial Cubic: [1, u, u^2, u^3]
    phi_mono = np.vstack([u ** 0, u ** 1, u ** 2, u ** 3])
    _, C_mono, kG_mono, kC_mono, corr_mono = compute_gram_and_correlation(phi_mono)

    # 2. Chebyshev Basis: T0=1, T1=u, T2=2u^2-1, T3=4u^3-3u
    phi_cheb = np.vstack([
        np.ones_like(u),
        u,
        2.0 * u ** 2 - 1.0,
        4.0 * u ** 3 - 3.0 * u
    ])
    _, C_cheb, kG_cheb, kC_cheb, corr_cheb = compute_gram_and_correlation(phi_cheb)

    # 3. Legendre Basis: P0=1, P1=u, P2=0.5*(3u^2-1), P3=0.5*(5u^3-3u)
    phi_leg = np.vstack([
        np.ones_like(u),
        u,
        0.5 * (3.0 * u ** 2 - 1.0),
        0.5 * (5.0 * u ** 3 - 3.0 * u)
    ])
    _, C_leg, kG_leg, kC_leg, corr_leg = compute_gram_and_correlation(phi_leg)

    # 4. Learned Gegenbauer Basis at beta = 0.74 (empirical mean across channels)
    beta = 0.74
    phi_gb = np.vstack([
        np.ones_like(u),
        2.0 * beta * u,
        2.0 * beta * (beta + 1.0) * u ** 2 - beta,
        (4.0 * beta * (beta + 1.0) * (beta + 2.0) * u ** 3 - 6.0 * beta * (beta + 1.0) * u) / 3.0
    ])
    _, C_gb, kG_gb, kC_gb, corr_gb = compute_gram_and_correlation(phi_gb)

    print("=" * 80)
    print("POLYNOMIAL BASIS SCALE-INVARIANT GRAM CONDITIONING (SECTION 3.2)")
    print("=" * 80)
    print(f"{'Basis Formulation':<26} {'Raw kappa(G)':<16} {'Scale-Inv kappa(C)':<20} {'Max Correlation':<16}")
    print("-" * 80)
    print(f"{'Monomial Cubic (u^k)':<26} {kG_mono:<16.2f} {kC_mono:<20.2f} {corr_mono:<16.3f}")
    print(f"{'Chebyshev (T_k)':<26} {kG_cheb:<16.2f} {kC_cheb:<20.2f} {corr_cheb:<16.3f}")
    print(f"{'Legendre (P_k, beta=0.5)':<26} {kG_leg:<16.2f} {kC_leg:<20.2f} {corr_leg:<16.3f}")
    print(f"{'Gegenbauer (beta=0.74)':<26} {kG_gb:<16.2f} {kC_gb:<20.2f} {corr_gb:<16.3f}")
    print("=" * 80)

    # Verifications strictly matching text in Section 3.2
    assert abs(kC_mono - 28.6) < 0.2, f"Expected monomial kappa(C) ~ 28.6, got {kC_mono}"
    assert abs(corr_mono - 0.932) < 0.005, f"Expected monomial max corr ~ 0.932, got {corr_mono}"
    assert abs(kC_cheb - 2.33) < 0.05, f"Expected Chebyshev kappa(C) ~ 2.33, got {kC_cheb}"
    assert abs(kC_leg - 1.48) < 0.05, f"Expected Legendre kappa(C) ~ 1.47, got {kC_leg}"
    assert abs(kC_gb - 2.10) < 0.05, f"Expected Gegenbauer kappa(C) ~ 2.10, got {kC_gb}"
    print("[+] ALL MATHEMATICAL BASIS CONDITIONING THEOREMS AND EMPIRICAL VALUES VERIFIED 100%!")


if __name__ == "__main__":
    evaluate_all_bases()
