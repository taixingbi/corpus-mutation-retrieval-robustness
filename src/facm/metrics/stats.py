"""Bootstrap / paired significance helpers for Paper 4 reports."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int | None]:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots[i] = float(sample.mean())
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {"n": n, "mean": float(arr.mean()), "ci_low": lo, "ci_high": hi}


def paired_bootstrap_mean_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int | None]:
    """Mean(a - b) with bootstrap CI; pairs by index."""
    if len(a) != len(b) or not a:
        return {
            "n": len(a),
            "mean_diff": None,
            "ci_low": None,
            "ci_high": None,
            "p_two_sided": None,
        }
    diffs = np.asarray([float(x) - float(y) for x, y in zip(a, b)], dtype=float)
    n = len(diffs)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boots[i] = float(sample.mean())
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    # two-sided p: fraction of bootstrap means on the other side of 0
    mean_diff = float(diffs.mean())
    if mean_diff >= 0:
        p = 2.0 * float(np.mean(boots <= 0.0))
    else:
        p = 2.0 * float(np.mean(boots >= 0.0))
    p = min(1.0, p)
    return {
        "n": n,
        "mean_diff": mean_diff,
        "ci_low": lo,
        "ci_high": hi,
        "p_two_sided": p,
    }


def wilcoxon_signed_rank(
    a: Sequence[float],
    b: Sequence[float],
) -> dict[str, float | int | None]:
    """Approximate two-sided Wilcoxon signed-rank via normal approximation (no SciPy)."""
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    nz = [d for d in diffs if abs(d) > 1e-12]
    n = len(nz)
    if n < 5:
        return {"n": n, "statistic": None, "p_two_sided": None, "note": "n<5"}
    # rank absolute diffs
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if nz[i] < 0)
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return {"n": n, "statistic": float(w), "p_two_sided": None}
    z = (w - mean_w) / (var_w**0.5)
    # two-sided normal p
    p = 2.0 * (0.5 * (1.0 - _erf(abs(z) / (2**0.5))))
    return {"n": n, "statistic": float(w), "p_two_sided": float(min(1.0, max(0.0, p)))}


def _erf(x: float) -> float:
    # Abramowitz & Stegun approximation
    sign = 1.0 if x >= 0 else -1.0
    ax = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * float(y)
