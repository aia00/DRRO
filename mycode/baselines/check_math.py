#!/usr/bin/env python3
"""Lightweight math checks for baseline formulas."""

from __future__ import annotations

import numpy as np


def check_ensemble_aggregations() -> None:
    scores = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    mean = scores.mean(axis=1)
    var = scores.var(axis=1)
    wco = scores.min(axis=1)
    uwo = mean - 0.5 * var

    assert np.allclose(mean, np.array([2.0, 0.0]))
    assert np.allclose(wco, np.array([1.0, -1.0]))
    assert np.allclose(var, np.array([2.0 / 3.0, 2.0 / 3.0]))
    assert np.allclose(uwo, np.array([2.0 - 1.0 / 3.0, -1.0 / 3.0]))


def check_dual_updates() -> None:
    # mu-PPO: lambda >= 0, update with theta - mean
    lam_mu = 0.2
    theta = 1.0
    mean_val = 0.8
    lr = 0.5
    lam_mu = max(0.0, lam_mu + lr * (theta - mean_val))
    assert abs(lam_mu - 0.3) < 1e-8

    # xi-PPO: signed lambda, update with mean - theta
    lam_xi = 0.2
    lam_xi = lam_xi + lr * (mean_val - theta)
    assert abs(lam_xi - 0.1) < 1e-8


def main() -> None:
    check_ensemble_aggregations()
    check_dual_updates()
    print("math checks passed")


if __name__ == "__main__":
    main()
