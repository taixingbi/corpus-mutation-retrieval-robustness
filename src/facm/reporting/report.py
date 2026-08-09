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


def build_rq_report(
    *,
    mutations: list[dict],
    behavior: list[dict],
    retrieval: list[dict],
) -> dict[str, Any]:
    """Paper 4 report: RQ1 Validity → RQ2 Effectiveness → RQ3 Transfer (+ ablation)."""

    by_fail_gen: dict[str, list[bool]] = defaultdict(list)
    for m in mutations:
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

    mut_accepted: dict[str, bool] = {}
    mut_fail: dict[str, str] = {}
    for b in behavior:
        mid = b["mutation_id"]
        mut_fail[mid] = b["failure"]
        if b.get("accepted") is not None:
            mut_accepted[mid] = bool(b["accepted"])

    sem_pass_ids = {
        m["mutation_id"]
        for m in mutations
        if (m.get("semantic") or {}).get("passed")
    }
    rq2_ber: dict[str, Any] = {}
    for failure in sorted(set(list(by_fail_gen.keys()) + list(mut_fail.values()))):
        sem_n = rq1.get(failure, {}).get("valid", 0)
        beh_mids = {mid for mid, f in mut_fail.items() if f == failure}
        denom_ids = beh_mids & sem_pass_ids if sem_pass_ids else beh_mids
        if not denom_ids and sem_n:
            denom = sem_n
            eff = sum(
                1
                for mid, ok in mut_accepted.items()
                if mut_fail.get(mid) == failure and ok
            )
        else:
            denom = len(denom_ids) if denom_ids else sem_n
            eff = sum(1 for mid in denom_ids if mut_accepted.get(mid))
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

    accepted_ids = {mid for mid, ok in mut_accepted.items() if ok}
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

    all_mids = {m["mutation_id"] for m in mutations}
    beh_pass = accepted_ids

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
                if (
                    float(c["mrr"]) - float(r["mrr"]) > 0
                    or float(r["recall_at_5"]) < float(c["recall_at_5"])
                )
                else 0.0
            )
        return mean(flags)

    ablation = {
        "raw_mutations": {
            "n_mutations": len(all_mids),
            "held_out_challenge_rate": challenge_rate(all_mids),
        },
        "plus_semantic": {
            "n_mutations": len(sem_pass_ids),
            "held_out_challenge_rate": challenge_rate(sem_pass_ids),
        },
        "plus_semantic_behavioral": {
            "n_mutations": len(beh_pass),
            "held_out_challenge_rate": challenge_rate(beh_pass),
        },
    }

    return {
        "paper": "Behaviorally Validated Corpus Mutation for Retrieval Robustness Evaluation",
        "scope": "Generate → Validate → Transfer",
        "rq1_semantic_validity": rq1,
        "rq2_behavioral_effectiveness": {
            "ber_by_failure": rq2_ber,
            "paired_clean_to_mutated": rq2_metrics,
            "note": (
                "BER = behaviorally_effective / semantically_valid. "
                "Paired metrics use construction retrievers (BM25, BGE, Hybrid) only."
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
