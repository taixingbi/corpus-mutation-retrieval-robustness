from __future__ import annotations

import math
import random
from typing import Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Percentile bootstrap CI for the mean."""
    xs = list(values)
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(max(1, n_boot)):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo_i = int(math.floor(alpha / 2 * len(means)))
    hi_i = int(math.ceil((1 - alpha / 2) * len(means))) - 1
    hi_i = max(0, min(hi_i, len(means) - 1))
    return {
        "n": n,
        "mean": mean(xs),
        "ci_low": means[lo_i],
        "ci_high": means[hi_i],
    }


def paired_bootstrap_mean_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Paired bootstrap CI for mean(a - b). ``a`` and ``b`` must be aligned."""
    if len(a) != len(b) or not a:
        return {
            "n": 0,
            "mean_diff": None,
            "ci_low": None,
            "ci_high": None,
            "p_two_sided": None,
        }
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(max(1, n_boot)):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot.append(mean(sample))
    boot.sort()
    lo_i = int(math.floor(alpha / 2 * len(boot)))
    hi_i = int(math.ceil((1 - alpha / 2) * len(boot))) - 1
    hi_i = max(0, min(hi_i, len(boot) - 1))
    # two-sided p: fraction of bootstrap means with opposite sign to observed
    obs = mean(diffs)
    p_le = sum(1 for m in boot if m <= 0) / len(boot)
    p_ge = sum(1 for m in boot if m >= 0) / len(boot)
    p = 1.0 if abs(obs) < 1e-15 else min(1.0, 2.0 * min(p_le, p_ge))
    return {
        "n": n,
        "mean_diff": obs,
        "ci_low": boot[lo_i],
        "ci_high": boot[hi_i],
        "p_two_sided": p,
    }


def wilcoxon_signed_rank(
    a: Sequence[float],
    b: Sequence[float],
) -> dict[str, float | int | None]:
    """Approximate two-sided Wilcoxon signed-rank p-value (normal approx, no scipy)."""
    diffs = [x - y for x, y in zip(a, b) if abs(x - y) > 1e-12]
    n = len(diffs)
    if n < 5:
        return {"n": n, "statistic": None, "p_two_sided": None}
    # rank absolute diffs
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(diffs[order[j + 1]]) - abs(diffs[order[i]])) < 1e-12:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return {"n": n, "statistic": w, "p_two_sided": None}
    z = (w - mean_w) / math.sqrt(var_w)
    # two-sided normal approx
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return {"n": n, "statistic": w, "p_two_sided": min(1.0, p)}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
