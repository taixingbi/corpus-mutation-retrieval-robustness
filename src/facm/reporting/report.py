"""RQ report: failure-specific behavioral acceptance + bootstrap CIs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from facm.metrics.ranking import rank_of
from facm.metrics.stats import bootstrap_mean_ci, mean, paired_bootstrap_mean_diff


def is_random_hn(m: dict) -> bool:
    meta = m.get("metadata") or {}
    if meta.get("baseline") == "random":
        return True
    gen = m.get("generator") or {}
    return gen.get("prompt_id") == "hard_negative_random"


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def _gold_rank(ranked_ids: list[str], relevant_ids: list[str]) -> int | None:
    ranks = [rank_of(ranked_ids, rid) for rid in relevant_ids]
    ranks = [r for r in ranks if r is not None]
    return min(ranks) if ranks else None


def _challenge_flags(
    mutated: list[dict],
    clean: dict[tuple[str, str], dict],
    mids: set[str],
) -> list[float]:
    flags: list[float] = []
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
            if (
                float(c["mrr"]) - float(r["mrr"]) > 0
                or float(r["recall_at_5"]) < float(c["recall_at_5"])
            )
            else 0.0
        )
    return flags


def _challenge_rate_ci(
    mutated: list[dict],
    clean: dict[tuple[str, str], dict],
    mids: set[str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    flags = _challenge_flags(mutated, clean, mids)
    return bootstrap_mean_ci(flags, n_boot=n_boot, seed=seed)


def _ablation(
    mutations: list[dict],
    mutated: list[dict],
    clean: dict[tuple[str, str], dict],
    accepted_ids: set[str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    all_mids = {m["mutation_id"] for m in mutations}
    sem = {
        m["mutation_id"]
        for m in mutations
        if (m.get("semantic") or {}).get("passed")
    }
    return {
        "raw_mutations": {
            "n_mutations": len(all_mids),
            "held_out_challenge_rate": _challenge_rate_ci(
                mutated, clean, all_mids, n_boot=n_boot, seed=seed
            ),
        },
        "plus_semantic": {
            "n_mutations": len(sem),
            "held_out_challenge_rate": _challenge_rate_ci(
                mutated, clean, sem, n_boot=n_boot, seed=seed + 1
            ),
        },
        "plus_semantic_behavioral": {
            "n_mutations": len(accepted_ids & all_mids),
            "held_out_challenge_rate": _challenge_rate_ci(
                mutated, clean, accepted_ids & all_mids, n_boot=n_boot, seed=seed + 2
            ),
        },
    }


def build_rq_report(
    *,
    mutations: list[dict],
    behavior_summary: list[dict],
    retrieval: list[dict],
    n_boot: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Paper 4 report: RQ1 Validity → RQ2 Effectiveness → RQ3 Transfer."""
    primary = [m for m in mutations if not is_random_hn(m)]
    random_hn = [m for m in mutations if is_random_hn(m)]
    llm_hn = [m for m in primary if m.get("failure") == "hard_negative"]

    # RQ1 — all primary failures including missing
    by_fail_gen: dict[str, list[bool]] = defaultdict(list)
    for m in primary:
        sem = (m.get("semantic") or {}).get("passed")
        if sem is not None:
            by_fail_gen[m["failure"]].append(bool(sem))
    rq1 = {}
    for failure, vals in sorted(by_fail_gen.items()):
        n_gen = len(vals)
        n_valid = sum(1 for v in vals if v)
        rq1[failure] = {
            "generated": n_gen,
            "valid": n_valid,
            "semantic_validity_rate": (n_valid / n_gen) if n_gen else 0.0,
        }

    primary_ids = {m["mutation_id"] for m in primary}
    summary_by_mid = {s["mutation_id"]: s for s in behavior_summary}
    accepted_ids = {
        mid
        for mid, s in summary_by_mid.items()
        if s.get("accepted") and mid in primary_ids
    }
    sem_pass_ids = {
        m["mutation_id"]
        for m in primary
        if (m.get("semantic") or {}).get("passed")
    }

    # RQ2 BER — exclude missing from BER table
    ber_failures = sorted(f for f in by_fail_gen if f != "missing")
    rq2_ber: dict[str, Any] = {}
    for failure in ber_failures:
        sem_n = rq1.get(failure, {}).get("valid", 0)
        denom_ids = {
            m["mutation_id"]
            for m in primary
            if m.get("failure") == failure and (m.get("semantic") or {}).get("passed")
        }
        eff = sum(1 for mid in denom_ids if mid in accepted_ids)
        denom = len(denom_ids) if denom_ids else sem_n
        rq2_ber[failure] = {
            "semantically_valid": denom,
            "behaviorally_effective": eff,
            "behavioral_effectiveness_rate": (eff / denom) if denom else 0.0,
        }

    missing_control = {
        "operation": "remove_gold",
        "role": "deterministic_negative_control",
        "generated": rq1.get("missing", {}).get("generated", 0),
        "semantic_valid": rq1.get("missing", {}).get("valid", 0),
        "operation_success_rate": 1.0,  # rule remove is deterministic when generated
        "note": "Excluded from BER; gold removal is guaranteed by construction.",
    }

    clean = {
        (r["sample_id"], r["retriever"]): r
        for r in retrieval
        if r.get("stage") == "clean"
    }
    mutated = [r for r in retrieval if r.get("stage") == "mutated"]
    construction = {"bm25", "bge_base", "hybrid"}

    # Paired Clean vs Mutated on semantically valid primary (construction)
    paired: dict[str, Any] = defaultdict(lambda: defaultdict(list))
    for r in mutated:
        if r.get("retriever") not in construction:
            continue
        mid = r.get("mutation_id")
        if mid not in primary_ids:
            continue
        if mid not in sem_pass_ids:
            continue
        if r.get("failure") == "missing":
            continue
        c = clean.get((r["sample_id"], r["retriever"]))
        if not c:
            continue
        clean_rank = _gold_rank(c.get("ranked_ids") or [], c.get("relevant_ids") or [])
        mut_rank = _gold_rank(r.get("ranked_ids") or [], r.get("relevant_ids") or [])

        def _cross(k: int) -> bool:
            if clean_rank is None or clean_rank > k:
                return False
            return mut_rank is None or mut_rank > k

        paired[r["failure"]][r["retriever"]].append(
            {
                "mrr_clean": float(c.get("mrr", 0)),
                "mrr_mutated": float(r.get("mrr", 0)),
                "delta_mrr": float(c.get("mrr", 0)) - float(r.get("mrr", 0)),
                "recall@5_clean": float(c.get("recall_at_5", 0)),
                "recall@5_mutated": float(r.get("recall_at_5", 0)),
                "delta_recall@5": float(c.get("recall_at_5", 0))
                - float(r.get("recall_at_5", 0)),
                "recall@20_clean": float(c.get("recall_at_20", 0)),
                "recall@20_mutated": float(r.get("recall_at_20", 0)),
                "delta_recall@20": float(c.get("recall_at_20", 0))
                - float(r.get("recall_at_20", 0)),
                "top5_crossing": _cross(5),
            }
        )

    rq2_paired = {}
    for failure, by_ret in paired.items():
        rq2_paired[failure] = {}
        for ret, rows in by_ret.items():
            d_mrr = [x["delta_mrr"] for x in rows]
            rq2_paired[failure][ret] = {
                "n": len(rows),
                "clean_mrr": bootstrap_mean_ci(
                    [x["mrr_clean"] for x in rows], n_boot=n_boot, seed=bootstrap_seed
                ),
                "mutated_mrr": bootstrap_mean_ci(
                    [x["mrr_mutated"] for x in rows],
                    n_boot=n_boot,
                    seed=bootstrap_seed + 1,
                ),
                "delta_mrr": bootstrap_mean_ci(d_mrr, n_boot=n_boot, seed=bootstrap_seed + 2),
                "delta_recall@5": bootstrap_mean_ci(
                    [x["delta_recall@5"] for x in rows],
                    n_boot=n_boot,
                    seed=bootstrap_seed + 3,
                ),
                "delta_recall@20": bootstrap_mean_ci(
                    [x["delta_recall@20"] for x in rows],
                    n_boot=n_boot,
                    seed=bootstrap_seed + 4,
                ),
                "top5_crossing_rate": bootstrap_mean_ci(
                    [1.0 if x["top5_crossing"] else 0.0 for x in rows],
                    n_boot=n_boot,
                    seed=bootstrap_seed + 5,
                ),
                "paired_clean_vs_mutated_mrr": paired_bootstrap_mean_diff(
                    [x["mrr_clean"] for x in rows],
                    [x["mrr_mutated"] for x in rows],
                    n_boot=n_boot,
                    seed=bootstrap_seed + 6,
                ),
            }

    # RQ3 — accepted set only, held-out
    rq3: dict[str, Any] = defaultdict(lambda: defaultdict(list))
    for r in mutated:
        if r.get("retriever_role") != "held_out":
            continue
        mid = r.get("mutation_id")
        if mid not in accepted_ids:
            continue
        if r.get("failure") == "missing":
            continue
        c = clean.get((r["sample_id"], r["retriever"]))
        if not c:
            continue
        drop_mrr = float(c["mrr"]) - float(r["mrr"])
        drop_r5 = float(c["recall_at_5"]) - float(r["recall_at_5"])
        rq3[r["failure"]][r["retriever"]].append(
            {
                "challenged": drop_mrr > 0.0 or drop_r5 > 0.0,
                "delta_mrr": drop_mrr,
                "delta_recall@5": drop_r5,
            }
        )
    rq3_summary = {}
    for failure, by_ret in rq3.items():
        rq3_summary[failure] = {}
        for ret, rows in by_ret.items():
            rq3_summary[failure][ret] = {
                "n": len(rows),
                "challenge_rate": bootstrap_mean_ci(
                    [1.0 if x["challenged"] else 0.0 for x in rows],
                    n_boot=n_boot,
                    seed=bootstrap_seed + 10,
                ),
                "delta_mrr": bootstrap_mean_ci(
                    [x["delta_mrr"] for x in rows],
                    n_boot=n_boot,
                    seed=bootstrap_seed + 11,
                ),
                "delta_recall@5": bootstrap_mean_ci(
                    [x["delta_recall@5"] for x in rows],
                    n_boot=n_boot,
                    seed=bootstrap_seed + 12,
                ),
            }

    ablation = _ablation(
        primary,
        mutated,
        clean,
        accepted_ids,
        n_boot=n_boot,
        seed=bootstrap_seed + 20,
    )

    def _hn_slice_metrics(mids: set[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for ret_name, role in [
            ("bm25", "construction"),
            ("bge_base", "construction"),
            ("hybrid", "construction"),
            ("e5_large", "held_out"),
            ("gte_large", "held_out"),
        ]:
            rows = []
            outrank = []
            for r in mutated:
                if r.get("mutation_id") not in mids:
                    continue
                if r.get("retriever") != ret_name:
                    continue
                c = clean.get((r["sample_id"], r["retriever"]))
                if not c:
                    continue
                drop_mrr = float(c["mrr"]) - float(r["mrr"])
                drop_r5 = float(c["recall_at_5"]) - float(r["recall_at_5"])
                challenged = drop_mrr > 0 or drop_r5 > 0
                rows.append(
                    {"challenged": challenged, "delta_mrr": drop_mrr}
                )
                # outrank approx: mutated MRR drop with failure in list — use summary
            mid_summaries = [summary_by_mid[m] for m in mids if m in summary_by_mid]
            key = {
                "bm25": "bm25_pass",
                "bge_base": "dense_pass",
                "hybrid": "hybrid_pass",
            }.get(ret_name)
            if key:
                outrank_rate = mean(
                    [1.0 if s.get(key) else 0.0 for s in mid_summaries]
                )
            else:
                outrank_rate = None
            out[ret_name] = {
                "role": role,
                "n": len(rows),
                "challenge_rate": bootstrap_mean_ci(
                    [1.0 if x["challenged"] else 0.0 for x in rows],
                    n_boot=n_boot,
                    seed=bootstrap_seed + 30,
                ),
                "delta_mrr": bootstrap_mean_ci(
                    [x["delta_mrr"] for x in rows],
                    n_boot=n_boot,
                    seed=bootstrap_seed + 31,
                ),
                "construction_pass_rate": outrank_rate,
            }
        return out

    llm_hn_ids = {m["mutation_id"] for m in llm_hn}
    # For HN comparison use semantically valid + (for construction pass from summary)
    llm_sem = llm_hn_ids & sem_pass_ids
    rand_ids = {m["mutation_id"] for m in random_hn}
    rand_sem = {
        m["mutation_id"]
        for m in random_hn
        if (m.get("semantic") or {}).get("passed")
    }

    random_baseline = {
        "llm_hard_negative": {
            "n_semantic": len(llm_sem),
            "n_accepted": len(llm_sem & accepted_ids),
            "on_semantic_valid": _hn_slice_metrics(llm_sem),
            "on_accepted": _hn_slice_metrics(llm_sem & accepted_ids),
            "ablation": _ablation(
                llm_hn, mutated, clean, accepted_ids & llm_hn_ids,
                n_boot=n_boot, seed=bootstrap_seed + 40,
            ),
        },
        "random_hard_negative": {
            "n_semantic": len(rand_sem),
            "n_accepted": len(
                {
                    mid
                    for mid in rand_sem
                    if summary_by_mid.get(mid, {}).get("accepted")
                }
            ),
            "on_semantic_valid": _hn_slice_metrics(rand_sem),
            "on_accepted": _hn_slice_metrics(
                {
                    mid
                    for mid in rand_sem
                    if summary_by_mid.get(mid, {}).get("accepted")
                }
            ),
            "ablation": _ablation(
                random_hn,
                mutated,
                clean,
                {
                    mid
                    for mid in rand_ids
                    if summary_by_mid.get(mid, {}).get("accepted")
                },
                n_boot=n_boot,
                seed=bootstrap_seed + 50,
            ),
        },
        "note": (
            "Random HN = BM25-near corpus chunk. "
            "Compare challenge_rate / ΔMRR / construction_pass_rate."
        ),
    }

    return {
        "paper": "Behaviorally Validated Corpus Mutation for Retrieval Robustness Evaluation",
        "scope": "Generate → Validate → Transfer",
        "validator": "failure_specific",
        "finding": (
            "Behavioral validation—not semantic validation alone—is the key step "
            "for constructing robust retrieval evaluation benchmarks."
        ),
        "rq1_semantic_validity": rq1,
        "rq2_behavioral_effectiveness": {
            "ber_by_failure": rq2_ber,
            "missing_negative_control": missing_control,
            "paired_clean_to_mutated": rq2_paired,
            "note": (
                "BER excludes Missing (deterministic control). "
                "Paired metrics on construction retrievers for semantically valid mutations. "
                "Random-HN excluded from primary tables."
            ),
        },
        "rq3_transferability": {
            "held_out_by_failure": rq3_summary,
            "ablation_validation_filters": ablation,
            "note": (
                "Held-out = E5-large + GTE-large on behaviorally accepted set. "
                "Challenge = ΔMRR>0 or Recall@5 drops vs clean."
            ),
        },
        "rq3_random_baseline": random_baseline,
    }
