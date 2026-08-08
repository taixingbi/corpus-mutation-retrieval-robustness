from facm.metrics.ranking import recoverability_class
from facm.metrics.stats import (
    bootstrap_mean_ci,
    paired_bootstrap_mean_diff,
    wilcoxon_signed_rank,
)


def test_recoverability_missing_is_control():
    assert (
        recoverability_class("missing", ["a", "b"], []) == "unrecoverable_control"
    )


def test_recoverability_gold_in_pool():
    assert recoverability_class("noise", ["x", "gold"], ["gold"]) == "recoverable"
    assert recoverability_class("noise", ["x", "y"], ["gold"]) == "out_of_pool"


def test_bootstrap_mean_ci():
    out = bootstrap_mean_ci([0.0, 1.0, 0.5], n_boot=200, seed=1)
    assert out["n"] == 3
    assert out["mean"] == 0.5
    assert out["ci_low"] is not None and out["ci_high"] is not None


def test_paired_bootstrap_and_wilcoxon():
    a = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    b = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2]
    boot = paired_bootstrap_mean_diff(a, b, n_boot=300, seed=2)
    assert boot["n"] == 6
    assert boot["mean_diff"] is not None and boot["mean_diff"] > 0
    wil = wilcoxon_signed_rank(a, b)
    assert wil["n"] == 6
    assert wil["p_two_sided"] is not None
