from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from facm.metrics.ranking import rank_of


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def _gold_rank(ranked_ids: list[str], relevant_ids: list[str]) -> int | None:
    ranks = [rank_of(ranked_ids, rid) for rid in relevant_ids]
    ranks = [r for r in ranks if r is not None]
    return min(ranks) if ranks else None


def is_random_hn(m: dict) -> bool:
    meta = m.get("metadata") or {}
    if meta.get("baseline") == "random":
        return True
    gen = m.get("generator") or {}
    return gen.get("prompt_id") == "hard_negative_random"


def _challenge_rate(
    mutated: list[dict],
    clean: dict[tuple[str, str], dict],
    mids: set[str],
) -> float:
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
            if (
                float(c["mrr"]) - float(r["mrr"]) > 0
                or float(r["recall_at_5"]) < float(c["recall_at_5"])
            )
            else 0.0
        )
    return mean(flags)


def _ablation_block(
    mutations: list[dict],
    mutated: list[dict],
    clean: dict[tuple[str, str], dict],
    mut_accepted: dict[str, bool],
) -> dict[str, Any]:
    all_mids = {m["mutation_id"] for m in mutations}
    sem_pass = {
        m["mutation_id"]
        for m in mutations
        if (m.get("semantic") or {}).get("passed")
    }
    beh_pass = {mid for mid in all_mids if mut_accepted.get(mid)}
    return {
        "raw_mutations": {
            "n_mutations": len(all_mids),
            "held_out_challenge_rate": _challenge_rate(mutated, clean, all_mids),
        },
        "plus_semantic": {
            "n_mutations": len(sem_pass),
            "held_out_challenge_rate": _challenge_rate(mutated, clean, sem_pass),
        },
        "plus_semantic_behavioral": {
            "n_mutations": len(beh_pass),
            "held_out_challenge_rate": _challenge_rate(mutated, clean, beh_pass),
        },
    }


def build_rq_report(
    *,
    mutations: list[dict],
    behavior: list[dict],
    retrieval: list[dict],
) -> dict[str, Any]:
    """Paper 4 report: RQ1 Validity → RQ2 Effectiveness → RQ3 Transfer (+ ablation).

    Primary RQ tables exclude Random-HN baseline mutations.
    """
    primary = [m for m in mutations if not is_random_hn(m)]
    random_hn = [m for m in mutations if is_random_hn(m)]
    llm_hn = [
        m
        for m in primary
        if m.get("failure") == "hard_negative"
    ]

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
    mut_accepted: dict[str, bool] = {}
    mut_fail: dict[str, str] = {}
    for b in behavior:
        mid = b["mutation_id"]
        if mid not in primary_ids and mid not in {m["mutation_id"] for m in random_hn}:
            continue
        mut_fail[mid] = b["failure"]
        if b.get("accepted") is not None:
            mut_accepted[mid] = bool(b["accepted"])

    # Primary BER uses primary mutations only
    primary_accepted = {
        mid: ok for mid, ok in mut_accepted.items() if mid in primary_ids
    }
    primary_fail = {mid: f for mid, f in mut_fail.items() if mid in primary_ids}

    sem_pass_ids = {
        m["mutation_id"]
        for m in primary
        if (m.get("semantic") or {}).get("passed")
    }
    rq2_ber: dict[str, Any] = {}
    for failure in sorted(set(list(by_fail_gen.keys()) + list(primary_fail.values()))):
        sem_n = rq1.get(failure, {}).get("valid", 0)
        beh_mids = {mid for mid, f in primary_fail.items() if f == failure}
        denom_ids = beh_mids & sem_pass_ids if sem_pass_ids else beh_mids
        if not denom_ids and sem_n:
            denom = sem_n
            eff = sum(
                1
                for mid, ok in primary_accepted.items()
                if primary_fail.get(mid) == failure and ok
            )
        else:
            denom = len(denom_ids) if denom_ids else sem_n
            eff = sum(1 for mid in denom_ids if primary_accepted.get(mid))
        rq2_ber[failure] = {
            "semantically_valid": denom,
            "behaviorally_effective": eff,
            "behavioral_effectiveness_rate": (eff / denom) if denom else 0.0,
        }

    clean = {
        (r["sample_id"], r["retriever"]): r
        for r in retrieval
        if r.get("stage") == "clean"
    }
    mutated = [r for r in retrieval if r.get("stage") == "mutated"]
    construction_names = {"bm25", "bge_base", "hybrid"}
    rq2_paired: dict[str, Any] = defaultdict(lambda: defaultdict(list))
    for r in mutated:
        if r.get("retriever") not in construction_names:
            continue
        mid = r.get("mutation_id")
        if mid not in primary_ids:
            continue
        if mid not in sem_pass_ids and sem_pass_ids:
            continue
        c = clean.get((r["sample_id"], r["retriever"]))
        if not c:
            continue
        clean_rank = _gold_rank(c.get("ranked_ids") or [], c.get("relevant_ids") or [])
        mut_rank = _gold_rank(r.get("ranked_ids") or [], r.get("relevant_ids") or [])

        def _cross(k: int) -> bool:
            if clean_rank is None:
                return False
            if clean_rank > k:
                return False
            return mut_rank is None or mut_rank > k

        rq2_paired[r["failure"]][r["retriever"]].append(
            {
                "delta_recall@5": float(c.get("recall_at_5", 0)) - float(r.get("recall_at_5", 0)),
                "delta_recall@20": float(c.get("recall_at_20", 0)) - float(r.get("recall_at_20", 0)),
                "delta_mrr": float(c.get("mrr", 0)) - float(r.get("mrr", 0)),
                "gold_rank_clean": clean_rank,
                "gold_rank_mutated": mut_rank,
                "gold_rank_delta": (
                    None
                    if clean_rank is None
                    else ((mut_rank if mut_rank is not None else 10_000) - clean_rank)
                ),
                "top5_crossing": _cross(5),
                "top20_crossing": _cross(20),
                "ndcg@5_clean": float(c.get("ndcg_at_5", 0)),
                "ndcg@5_mutated": float(r.get("ndcg_at_5", 0)),
            }
        )

    rq2_metrics = {}
    for failure, by_ret in rq2_paired.items():
        rq2_metrics[failure] = {}
        for ret, rows in by_ret.items():
            rank_deltas = [x["gold_rank_delta"] for x in rows if x["gold_rank_delta"] is not None]
            rq2_metrics[failure][ret] = {
                "n": len(rows),
                "mean_delta_recall@5": mean([x["delta_recall@5"] for x in rows]),
                "mean_delta_recall@20": mean([x["delta_recall@20"] for x in rows]),
                "mean_delta_mrr": mean([x["delta_mrr"] for x in rows]),
                "mean_gold_rank_delta": mean(rank_deltas) if rank_deltas else None,
                "top5_crossing_rate": mean([1.0 if x["top5_crossing"] else 0.0 for x in rows]),
                "top20_crossing_rate": mean(
                    [1.0 if x["top20_crossing"] else 0.0 for x in rows]
                ),
                "mean_delta_ndcg@5": mean(
                    [x["ndcg@5_clean"] - x["ndcg@5_mutated"] for x in rows]
                ),
            }

    accepted_ids = {mid for mid, ok in primary_accepted.items() if ok}
    rq3: dict[str, Any] = defaultdict(lambda: defaultdict(list))
    for r in mutated:
        if r.get("retriever_role") != "held_out":
            continue
        if r.get("mutation_id") not in accepted_ids:
            continue
        c = clean.get((r["sample_id"], r["retriever"]))
        if not c:
            continue
        drop_mrr = float(c["mrr"]) - float(r["mrr"])
        drop_r5 = float(c["recall_at_5"]) - float(r["recall_at_5"])
        challenged = drop_mrr > 0.0 or drop_r5 > 0.0
        rq3[r["failure"]][r["retriever"]].append(
            {
                "challenged": challenged,
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
                "challenge_rate": mean([1.0 if x["challenged"] else 0.0 for x in rows]),
                "mean_delta_mrr": mean([x["delta_mrr"] for x in rows]),
                "mean_delta_recall@5": mean([x["delta_recall@5"] for x in rows]),
            }

    ablation = _ablation_block(primary, mutated, clean, primary_accepted)

    report: dict[str, Any] = {
        "paper": "Behaviorally Validated Corpus Mutation for Retrieval Robustness Evaluation",
        "scope": "Generate → Validate → Transfer",
        "finding": (
            "Behavioral validation—not semantic validation alone—is the key step "
            "for constructing robust retrieval evaluation benchmarks."
        ),
        "rq1_semantic_validity": rq1,
        "rq2_behavioral_effectiveness": {
            "ber_by_failure": rq2_ber,
            "paired_clean_to_mutated": rq2_metrics,
            "note": (
                "BER = behaviorally_effective / semantically_valid. "
                "Paired metrics use construction retrievers (BM25, BGE, Hybrid) only. "
                "Primary tables exclude Random-HN baseline."
            ),
        },
        "rq3_transferability": {
            "held_out_by_failure": rq3_summary,
            "ablation_validation_filters": ablation,
            "note": (
                "Held-out = E5-large + GTE-large; never used for mutation acceptance. "
                "Challenge = ΔMRR>0 or Recall@5 drops vs clean."
            ),
        },
    }

    if random_hn or llm_hn:
        report["rq3_random_baseline"] = {
            "llm_hard_negative": _ablation_block(
                llm_hn, mutated, clean, mut_accepted
            ),
            "random_hard_negative": _ablation_block(
                random_hn, mutated, clean, mut_accepted
            ),
            "note": (
                "Random HN = BM25-related corpus chunk (no LLM). "
                "Compare held-out challenge rates Raw → +Semantic → +Behavioral."
            ),
        }

    return report
