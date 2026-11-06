"""
Covariance health checks and Tikhonov regularization for DESI BAO tables.
"""

from __future__ import annotations
from typing import Any
import numpy as np


COV_COND_THRESHOLD: float = 1e10


COV_KAPPA_WARN: float = 1e3


TIKHONOV_EPS: float = 1e-8


def check_covariance_health(cov: np.ndarray, *, name: str = "") -> dict[str, Any]:
    """
    Diagnose DESI Gaussian covariance matrix before χ² fits.

    Returns condition number, eigenvalue span, symmetry error, and warnings.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square 2D, got shape {cov.shape}")

    n = int(cov.shape[0])
    sym_err = float(np.max(np.abs(cov - cov.T))) if n else 0.0
    trace = float(np.trace(cov))
    diag = np.diag(cov) if n else np.array([])
    min_diag = float(np.min(diag)) if n else float("nan")

    warnings: list[str] = []
    if sym_err > 1e-6 * max(np.max(np.abs(cov)), 1.0):
        warnings.append(f"asymmetry max|C-C.T|={sym_err:.2e}")

    if n == 0:
        return {
            "name": name,
            "n": 0,
            "trace": 0.0,
            "condition_number": float("inf"),
            "min_eigenvalue": float("nan"),
            "max_eigenvalue": float("nan"),
            "min_diagonal": float("nan"),
            "symmetry_error": sym_err,
            "is_positive_definite": False,
            "warnings": ["empty covariance matrix"],
            "healthy": False,
        }

    try:
        eigs = np.linalg.eigvalsh((cov + cov.T) * 0.5)
        min_eig = float(eigs.min())
        max_eig = float(eigs.max())
        cond = float(np.linalg.cond(cov))
    except np.linalg.LinAlgError as exc:
        warnings.append(f"eigen-analysis failed: {exc}")
        min_eig = float("nan")
        max_eig = float("nan")
        cond = float("inf")

    if min_diag <= 0:
        warnings.append(f"non-positive diagonal entry (min={min_diag:.2e})")
    if min_eig <= 0:
        warnings.append(f"not positive definite (λ_min={min_eig:.2e})")
    if cond > COV_COND_THRESHOLD:
        warnings.append(f"ill-conditioned (κ={cond:.2e} > {COV_COND_THRESHOLD:.0e})")
    elif np.isfinite(cond) and cond > COV_KAPPA_WARN:
        warnings.append(
            f"high condition number (κ={cond:.2e} > {COV_KAPPA_WARN:.0e}); "
            "do not trust p-values without regularization"
        )

    healthy = not warnings
    return {
        "name": name,
        "n": n,
        "trace": trace,
        "condition_number": cond,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "min_diagonal": min_diag,
        "symmetry_error": sym_err,
        "is_positive_definite": min_eig > 0 if np.isfinite(min_eig) else False,
        "warnings": warnings,
        "healthy": healthy,
    }


def apply_tikhonov_regularization(
    cov: np.ndarray,
    *,
    eps: float = TIKHONOV_EPS,
) -> tuple[np.ndarray, float]:
    """
    Light Tikhonov jitter: C_reg = C + ε (tr C / n) I.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 0:
        return cov.copy(), 0.0
    reg_strength = float(eps * np.trace(cov) / n)
    cov_reg = cov + reg_strength * np.eye(n)
    return cov_reg, reg_strength


def prepare_covariance(
    cov: np.ndarray,
    *,
    name: str = "",
    cond_threshold: float = COV_COND_THRESHOLD,
    eps: float = TIKHONOV_EPS,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Check covariance health and apply Tikhonov regularization only when κ is high.
    """
    health = check_covariance_health(cov, name=name)
    cov_out = np.asarray(cov, dtype=float).copy()
    reg_applied = False
    reg_strength = 0.0

    cond = float(health.get("condition_number", 0.0))
    if cond > cond_threshold or not health.get("is_positive_definite", True):
        if verbose:
            print("Applying light Tikhonov regularization to covariance...")
        cov_out, reg_strength = apply_tikhonov_regularization(cov_out, eps=eps)
        reg_applied = True
        if verbose:
            print(f"Regularization strength: {reg_strength:.2e}")

    health["regularization_applied"] = reg_applied
    health["regularization_strength"] = reg_strength
    health["cond_threshold"] = float(cond_threshold)
    if reg_applied:
        post = check_covariance_health(cov_out, name=name)
        health["condition_number_after"] = post["condition_number"]
        health["min_eigenvalue_after"] = post["min_eigenvalue"]

    return cov_out, health
