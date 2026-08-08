from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from facm.metrics.stats import (
    bootstrap_mean_ci,
    mean,
    paired_bootstrap_mean_diff,
    wilcoxon_signed_rank,
)

# Primary RQ4 absolute metrics (post-rerank on shared pool)
_RQ4_ABS = (
    "hit_at_1_post",
    "recall_at_5_post",
    "mrr_at_10_post",
    "ndcg_at_10_post",
    "mrr_post",
    "ndcg_at_5_post",
)
_RQ4_DELTA = (
    ("delta_hit_at_1", "hit_at_1_post", "hit_at_1_pre"),
    ("delta_recall_at_5", "recall_at_5_post", "recall_at_5_pre"),
    ("delta_mrr_at_10", "mrr_at_10_post", "mrr_at_10_pre"),
    ("delta_ndcg_at_10", "ndcg_at_10_post", "ndcg_at_10_pre"),
)
_RQ4_RECOVERY = (
    "recovery_hit_at_1",
    "recovery_recall_at_5",
    "recovery_mrr_at_10",
    "recovery_ndcg_at_10",
    "recovery_mrr",
)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def build_rq_report(
    *,
    mutations: list[dict],
    behavior: list[dict],
    retrieval: list[dict],
    rerank: list[dict],
    n_boot: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    # RQ1
    by_fail: dict[str, list[bool]] = defaultdict(list)
    for m in mutations:
        sem = (m.get("semantic") or {}).get("passed")
        if sem is not None:
            by_fail[m["failure"]].append(bool(sem))
    rq1 = {
        failure: {"n": len(vals), "semantic_success_rate": mean([1.0 if v else 0.0 for v in vals])}
        for failure, vals in sorted(by_fail.items())
    }

    # RQ2 — unique mutations
    mut_accepted: dict[str, bool] = {}
    mut_fail: dict[str, str] = {}
    for b in behavior:
        mid = b["mutation_id"]
        mut_fail[mid] = b["failure"]
        if b.get("accepted") is not None:
            mut_accepted[mid] = bool(b["accepted"])
    rq2_by_fail: dict[str, list[bool]] = defaultdict(list)
    for mid, ok in mut_accepted.items():
        rq2_by_fail[mut_fail[mid]].append(ok)
    rq2 = {
        failure: {
            "n": len(vals),
            "behavioral_success_rate": mean([1.0 if v else 0.0 for v in vals]),
        }
        for failure, vals in sorted(rq2_by_fail.items())
    }

    # RQ3 held-out challenge on behaviorally accepted mutations
    accepted_ids = {mid for mid, ok in mut_accepted.items() if ok}
    clean = {
        (r["sample_id"], r["retriever"]): r
        for r in retrieval
        if r.get("stage") == "clean"
    }
    mutated = [r for r in retrieval if r.get("stage") == "mutated"]
    rq3: dict[str, Any] = defaultdict(lambda: defaultdict(list))
    for r in mutated:
        if r.get("retriever_role") != "held_out":
            continue
        if r.get("mutation_id") not in accepted_ids:
            continue
        key = (r["sample_id"], r["retriever"])
        c = clean.get(key)
        if not c:
            continue
        drop = c["mrr"] - r["mrr"]
        challenged = drop > 0.0 or r["recall_at_5"] < c["recall_at_5"]
        rq3[r["failure"]][r["retriever"]].append(
            {
                "challenged": challenged,
                "delta_mrr": drop,
                "delta_recall@5": c["recall_at_5"] - r["recall_at_5"],
            }
        )
    rq3_summary = {}
    for failure, by_ret in rq3.items():
        rq3_summary[failure] = {}
        for ret, rows in by_ret.items():
            rq3_summary[failure][ret] = {
                "n": len(rows),
                "challenge_rate": mean([1.0 if x["challenged"] else 0.0 for x in rows]),
                "mean_delta_mrr": mean([x["delta_mrr"] for x in rows]),
                "mean_delta_recall@5": mean([x["delta_recall@5"] for x in rows]),
            }

    # Recoverability inventory (shared Hybrid pool)
    rec_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen_mid: set[str] = set()
    for r in rerank:
        if r.get("reranker") != "none":
            continue
        mid = r["mutation_id"]
        if mid in seen_mid:
            continue
        seen_mid.add(mid)
        rec_counts[r["failure"]][r.get("recoverability") or "out_of_pool"] += 1
    recoverability_summary = {
        failure: dict(counts) for failure, counts in sorted(rec_counts.items())
    }

    # RQ4 — absolute + delta + recovery, split by recoverability
    rq4 = _build_rq4(
        rerank, n_boot=n_boot, bootstrap_seed=bootstrap_seed
    )

    # Ablation
    sem_pass = {
        m["mutation_id"]
        for m in mutations
        if (m.get("semantic") or {}).get("passed")
    }
    beh_pass = {mid for mid, ok in mut_accepted.items() if ok}
    all_mids = {m["mutation_id"] for m in mutations}

    def challenge_rate(mids: set[str]) -> float:
        flags = []
        for r in mutated:
            if r.get("retriever_role") != "held_out":
                continue
            if r.get("mutation_id") not in mids:
                continue
            c = clean.get((r["sample_id"], r["retriever"]))
            if not c:
                continue
            flags.append(
                1.0
                if (c["mrr"] - r["mrr"] > 0 or r["recall_at_5"] < c["recall_at_5"])
                else 0.0
            )
        return mean(flags)

    ablation = {
        "raw_llm": {"n_mutations": len(all_mids), "held_out_challenge_rate": challenge_rate(all_mids)},
        "plus_semantic": {
            "n_mutations": len(sem_pass),
            "held_out_challenge_rate": challenge_rate(sem_pass),
        },
        "plus_semantic_behavioral": {
            "n_mutations": len(beh_pass),
            "held_out_challenge_rate": challenge_rate(beh_pass),
        },
    }

    return {
        "rq1_semantic_success": rq1,
        "rq2_behavioral_success": rq2,
        "rq3_held_out_generalization": rq3_summary,
        "recoverability": {
            "note": (
                "missing is an unrecoverable negative control (gold removed from corpus). "
                "Other failures are recoverable iff gold evidence remains in the shared "
                "Hybrid Top-K candidate pool."
            ),
            "by_failure": recoverability_summary,
        },
        "rq4_reranker_recovery": rq4,
        "ablation_validation_filters": ablation,
    }


def _build_rq4(
    rerank: list[dict],
    *,
    n_boot: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    # Index: failure -> recoverability -> reranker -> mutation_id -> row
    by: dict[str, dict[str, dict[str, dict[str, dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    pool_name = "hybrid_top_20"
    for r in rerank:
        pool_name = r.get("candidate_pool") or pool_name
        rclass = r.get("recoverability") or "out_of_pool"
        by[r["failure"]][rclass][r["reranker"]][r["mutation_id"]] = r

    out: dict[str, Any] = {
        "candidate_pool": pool_name,
        "by_failure": {},
        "paired_comparisons": {},
    }

    for failure, by_class in sorted(by.items()):
        out["by_failure"][failure] = {}
        for rclass, by_rr in by_class.items():
            # Prefer reporting recoverable slice; always include control for missing
            class_block: dict[str, Any] = {"n_mutations": 0, "rerankers": {}}
            # use intersection of mutation ids across rerankers for fair n
            id_sets = [set(d.keys()) for d in by_rr.values()]
            common = set.intersection(*id_sets) if id_sets else set()
            class_block["n_mutations"] = len(common)
            for reranker, rows_by_mid in sorted(by_rr.items()):
                rows = [rows_by_mid[mid] for mid in sorted(common) if mid in rows_by_mid]
                if not rows:
                    rows = list(rows_by_mid.values())
                class_block["rerankers"][reranker] = _summarize_reranker_rows(
                    rows, n_boot=n_boot, bootstrap_seed=bootstrap_seed + hash(reranker) % 997
                )
            out["by_failure"][failure][rclass] = class_block

        # Paired tests on recoverable (or unrecoverable_control for missing)
        focus = (
            "unrecoverable_control"
            if failure == "missing"
            else "recoverable"
        )
        focus_rr = by_class.get(focus) or {}
        if len(focus_rr) >= 2:
            out["paired_comparisons"][failure] = _paired_reranker_tests(
                focus_rr,
                metric="ndcg_at_10_post",
                n_boot=n_boot,
                seed=bootstrap_seed,
                slice_name=focus,
            )

    return out


def _summarize_reranker_rows(
    rows: list[dict],
    *,
    n_boot: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows)}
    for key in _RQ4_ABS:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        summary[key] = bootstrap_mean_ci(vals, n_boot=n_boot, seed=bootstrap_seed)
    for name, post_k, pre_k in _RQ4_DELTA:
        vals = [
            float(r[post_k]) - float(r[pre_k])
            for r in rows
            if r.get(post_k) is not None and r.get(pre_k) is not None
        ]
        summary[name] = bootstrap_mean_ci(vals, n_boot=n_boot, seed=bootstrap_seed + 1)
    for key in _RQ4_RECOVERY:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        summary[key] = bootstrap_mean_ci(vals, n_boot=n_boot, seed=bootstrap_seed + 2)
    return summary


def _paired_reranker_tests(
    by_rr: dict[str, dict[str, dict]],
    *,
    metric: str,
    n_boot: int,
    seed: int,
    slice_name: str,
) -> dict[str, Any]:
    names = sorted(by_rr.keys())
    # Align on common mutation_ids
    common = set.intersection(*(set(by_rr[n].keys()) for n in names)) if names else set()
    mids = sorted(common)
    series = {
        n: [float(by_rr[n][mid][metric]) for mid in mids if by_rr[n][mid].get(metric) is not None]
        for n in names
    }
    # drop if lengths mismatch due to missing metric
    min_n = min((len(v) for v in series.values()), default=0)
    if min_n != len(mids):
        # re-align strictly
        series = {
            n: [float(by_rr[n][mid][metric]) for mid in mids]
            for n in names
        }

    pairs: dict[str, Any] = {"slice": slice_name, "metric": metric, "n": len(mids), "pairs": {}}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}__vs__{b}"
            boot = paired_bootstrap_mean_diff(
                series[a], series[b], n_boot=n_boot, seed=seed + i
            )
            wil = wilcoxon_signed_rank(series[a], series[b])
            pairs["pairs"][key] = {
                "bootstrap_mean_diff_a_minus_b": boot,
                "wilcoxon_signed_rank": wil,
            }
    return pairs
