"""Numerical quadrature independent of the SymPy proof producer."""

import math

import numpy as np
from scipy.special import ndtr


def gaussian_expectation(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(weights, values) / math.sqrt(math.pi))


def check() -> dict:
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    rows = []
    max_error = 0.0
    for name, a_plus, a_minus in (
        ("ReLU", 1.0, 0.0),
        ("LeakyReLU(alpha=0.1)", 1.0, 0.1),
        ("identity", 1.0, 1.0),
    ):
        A = (a_plus**2 + a_minus**2) / 2
        for K in (0.25, 1.0, 4.0):
            z = np.sqrt(2 * K) * nodes
            slopes = np.where(z >= 0, a_plus, a_minus)
            sigma = slopes * z
            observed_sigma2 = gaussian_expectation(sigma**2, weights)
            observed_dsigma2 = gaussian_expectation(slopes**2, weights)
            error = max(abs(observed_sigma2 - A * K), abs(observed_dsigma2 - A))
            max_error = max(max_error, error)
            rows.append(
                {
                    "activation": name,
                    "K": K,
                    "E_sigma2": observed_sigma2,
                    "expected_E_sigma2": A * K,
                    "E_sigma_prime2": observed_dsigma2,
                    "expected_E_sigma_prime2": A,
                }
            )

    gelu_dsigma2 = []
    for K in (0.25, 1.0, 4.0):
        z = np.sqrt(2 * K) * nodes
        phi = np.exp(-(z**2) / 2) / math.sqrt(2 * math.pi)
        derivative = ndtr(z) + z * phi
        gelu_dsigma2.append(gaussian_expectation(derivative**2, weights))
    gelu_range = max(gelu_dsigma2) - min(gelu_dsigma2)
    checks = {
        "piecewise_linear_quadrature_matches_closed_form": max_error < 1e-12,
        "gelu_control_is_K_dependent": gelu_range > 0.02,
    }
    return {
        "rows": rows,
        "max_abs_error": max_error,
        "gelu_control": {"K": [0.25, 1.0, 4.0], "E_sigma_prime2": gelu_dsigma2},
        "checks": checks,
        "passed": all(checks.values()),
    }
