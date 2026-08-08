from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def build_rq_report(
    *,
    mutations: list[dict],
    behavior: list[dict],
    retrieval: list[dict],
    rerank: list[dict],
) -> dict[str, Any]:
    # RQ1
    by_fail: dict[str, list[bool]] = defaultdict(list)
    for m in mutations:
        sem = (m.get("semantic") or {}).get("passed")
        if sem is not None:
            by_fail[m["failure"]].append(bool(sem))
    rq1 = {
        failure: {"n": len(vals), "semantic_success_rate": _mean([1.0 if v else 0.0 for v in vals])}
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
            "behavioral_success_rate": _mean([1.0 if v else 0.0 for v in vals]),
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
                "challenge_rate": _mean([1.0 if x["challenged"] else 0.0 for x in rows]),
                "mean_delta_mrr": _mean([x["delta_mrr"] for x in rows]),
                "mean_delta_recall@5": _mean([x["delta_recall@5"] for x in rows]),
            }

    # RQ4 recovery
    rq4: dict[str, Any] = defaultdict(lambda: defaultdict(list))
    for r in rerank:
        if r.get("recovery_mrr") is not None:
            rq4[r["failure"]][r["reranker"]].append(r["recovery_mrr"])
    rq4_summary = {
        failure: {
            reranker: {"n": len(vals), "mean_recovery_mrr": _mean(vals)}
            for reranker, vals in by_rr.items()
        }
        for failure, by_rr in rq4.items()
    }

    # Ablation: raw vs semantic vs behavioral on held-out challenge
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
        return _mean(flags)

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
        "rq4_reranker_recovery": rq4_summary,
        "ablation_validation_filters": ablation,
    }
