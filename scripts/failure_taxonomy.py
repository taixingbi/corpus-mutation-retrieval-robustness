#!/usr/bin/env python3
"""Offline failure taxonomy for held-out non-transfer (no LLM, no re-retrieval).

Uses Original criteria (ΔMRR≥0.10, majority≥2/3). Classifies behaviorally
accepted mutations by why they do / do not challenge held-out retrievers.
Also summarizes construction-reject reasons among semantic-pass failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facm.io import load_models, read_jsonl  # noqa: E402
from facm.metrics.ranking import rank_of  # noqa: E402
from facm.reporting.report import is_random_hn  # noqa: E402
from facm.schemas.models import Mutation  # noqa: E402
from facm.validation.behavioral import (  # noqa: E402
    aggregate_acceptance,
    validate_behavior_for_retriever,
)

CONSTRUCTION = ["bm25", "bge_base", "hybrid"]
HELD_OUT = ["e5_large", "gte_large"]
BER_FAILURES = ["hard_negative", "conflict", "boundary", "noise"]


def _best_rank(ranked: list[str], ids: list[str]) -> int | None:
    ranks = [rank_of(ranked, rid) for rid in ids]
    ranks = [r for r in ranks if r is not None]
    return min(ranks) if ranks else None


def _challenged(clean_row: dict, mut_row: dict) -> bool:
    return (
        float(clean_row["mrr"]) - float(mut_row["mrr"]) > 0.0
        or float(mut_row["recall_at_5"]) < float(clean_row["recall_at_5"])
    )


def _mutation_in_top20(ranked: list[str], add_ids: list[str]) -> bool:
    return any(cid in ranked for cid in add_ids)


def _gold_stable(clean_row: dict, mut_row: dict, gold_ids: list[str]) -> bool:
    """Gold ranking unchanged within top-20 (or both absent)."""
    c_rank = _best_rank(list(clean_row.get("ranked_ids") or []), gold_ids)
    m_rank = _best_rank(list(mut_row.get("ranked_ids") or []), gold_ids)
    return c_rank == m_rank


def _classify_held_out(
    *,
    mut: Mutation,
    clean_row: dict,
    mut_row: dict,
) -> str:
    """ mutually exclusive primary bucket for one held-out retriever. """
    ranked = list(mut_row.get("ranked_ids") or [])
    add_ids = list(mut.add or [])
    gold_ids = list(mut.original_gold_ids or [])
    challenged = _challenged(clean_row, mut_row)
    if challenged:
        return "challenged"
    # Non-transfer buckets (priority order)
    if add_ids and not _mutation_in_top20(ranked, add_ids):
        return "mutation_never_top20"
    if _gold_stable(clean_row, mut_row, gold_ids):
        return "enters_top20_gold_stable"
    # Mutation present / gold moved but still not challenged by ΔMRR/R@5 definition
    return "enters_top20_no_challenge_signal"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="data/runs/main")
    p.add_argument("--out-json", default="reports/main/failure_taxonomy.json")
    p.add_argument("--out-md", default="reports/main/failure_taxonomy.md")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--delta-mrr-threshold", type=float, default=0.10)
    p.add_argument("--majority", type=int, default=2)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    mut_path = run_dir / "3_semantic" / "mutation.jsonl"
    if not mut_path.exists():
        mut_path = run_dir / "2_mutations" / "mutation.jsonl"
    ret_path = run_dir / "5_benchmark" / "retrieval_result.jsonl"

    mutations = load_models(mut_path, Mutation)
    retrieval = list(read_jsonl(ret_path))
    clean_idx: dict[tuple[str, str], dict] = {}
    mut_idx: dict[tuple[str, str], dict] = {}
    for r in retrieval:
        if r.get("stage") == "clean":
            clean_idx[(r["sample_id"], r["retriever"])] = r
        elif r.get("stage") == "mutated" and r.get("mutation_id"):
            mut_idx[(r["mutation_id"], r["retriever"])] = r

    # Recompute Original acceptance + reject reasons
    accepted: dict[str, Mutation] = {}
    reject_reasons: dict[str, Counter] = defaultdict(Counter)
    construction_pass: dict[str, bool] = {}

    for mutation in mutations:
        mdict = mutation.model_dump(mode="json")
        if is_random_hn(mdict):
            continue
        per = []
        for name in CONSTRUCTION:
            c = clean_idx.get((mutation.sample_id, name))
            m = mut_idx.get((mutation.mutation_id, name))
            if c is None or m is None:
                continue
            per.append(
                validate_behavior_for_retriever(
                    mutation,
                    retriever_name=name,
                    clean_ranked=list(c.get("ranked_ids") or []),
                    mutated_ranked=list(m.get("ranked_ids") or []),
                    top_k=args.top_k,
                    delta_mrr_threshold=args.delta_mrr_threshold,
                )
            )
        if not per:
            continue
        ok, _, reason = aggregate_acceptance(per, majority=args.majority)
        sem = bool(mutation.semantic and mutation.semantic.passed)
        construction_pass[mutation.mutation_id] = bool(ok)
        if mutation.failure == "missing":
            continue
        if sem and not ok:
            reject_reasons[mutation.failure][reason] += 1
            # also per-retriever reasons
            for row in per:
                rsn = str((row.metadata or {}).get("reason") or "none")
                if not row.accepted_for_retriever:
                    reject_reasons[f"{mutation.failure}__retriever"][rsn] += 1
        if sem and ok:
            accepted[mutation.mutation_id] = mutation

    # Held-out taxonomy on accepted set
    by_fail_ret: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    by_fail_mut: dict[str, Counter] = defaultdict(Counter)
    gold_rows: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n_rows": 0, "n_clean_gold_at1": 0, "n_nontransfer_and_gold_at1": 0}
    )
    construction_only: set[str] = set()
    construction_only_by_fail: Counter = Counter()

    for mid, mut in accepted.items():
        fail = mut.failure
        gold_ids = list(mut.original_gold_ids or [])
        any_held_challenged = False
        mut_buckets = []
        for ret in HELD_OUT:
            c = clean_idx.get((mut.sample_id, ret))
            m = mut_idx.get((mid, ret))
            if c is None or m is None:
                continue
            bucket = _classify_held_out(mut=mut, clean_row=c, mut_row=m)
            by_fail_ret[fail][ret][bucket] += 1
            mut_buckets.append(bucket)
            if bucket == "challenged":
                any_held_challenged = True

            gstat = gold_rows[(fail, ret)]
            gstat["n_rows"] += 1
            clean_gold = _best_rank(list(c.get("ranked_ids") or []), gold_ids)
            if clean_gold == 1:
                gstat["n_clean_gold_at1"] += 1
                if bucket != "challenged":
                    gstat["n_nontransfer_and_gold_at1"] += 1

        if not any_held_challenged:
            construction_only.add(mid)
            construction_only_by_fail[fail] += 1
            by_fail_mut[fail]["construction_only_no_heldout_challenge"] += 1
        else:
            by_fail_mut[fail]["transfers_to_at_least_one_heldout"] += 1

        non = [b for b in mut_buckets if b != "challenged"]
        if not non:
            by_fail_mut[fail]["challenged_both_or_one"] += 1
        elif all(b == "mutation_never_top20" for b in non):
            by_fail_mut[fail]["nontransfer_never_top20"] += 1
        elif any(b == "mutation_never_top20" for b in non):
            by_fail_mut[fail]["nontransfer_mixed_never_top20"] += 1
        elif all(b == "enters_top20_gold_stable" for b in non):
            by_fail_mut[fail]["nontransfer_gold_stable"] += 1
        else:
            by_fail_mut[fail]["nontransfer_in_top20_weak_effect"] += 1

    gold_ceiling: dict[str, Any] = {}
    for fail in BER_FAILURES:
        gold_ceiling[fail] = {}
        for ret in HELD_OUT:
            rows = gold_rows[(fail, ret)]
            n_rows = rows["n_rows"]
            n_at1 = rows["n_clean_gold_at1"]
            n_nt = rows["n_nontransfer_and_gold_at1"]
            gold_ceiling[fail][ret] = {
                "n": n_rows,
                "clean_gold_at1": n_at1,
                "clean_gold_at1_rate": (n_at1 / n_rows) if n_rows else 0.0,
                "nontransfer_and_clean_gold_at1": n_nt,
                "nontransfer_and_clean_gold_at1_rate": (n_nt / n_rows) if n_rows else 0.0,
            }

    # Per-retriever bucket rates
    held_out_buckets: dict[str, Any] = {}
    for fail in BER_FAILURES:
        held_out_buckets[fail] = {}
        for ret in HELD_OUT:
            ctr = by_fail_ret[fail][ret]
            n = sum(ctr.values())
            held_out_buckets[fail][ret] = {
                "n": n,
                "counts": dict(ctr),
                "rates": {k: (v / n if n else 0.0) for k, v in ctr.items()},
            }

    # Construction reject summary (semantic pass, not accepted)
    reject_summary = {
        fail: {
            "n_rejected": sum(reject_reasons[fail].values()),
            "aggregate_reasons": dict(reject_reasons[fail].most_common()),
            "per_retriever_fail_reasons": dict(
                reject_reasons[f"{fail}__retriever"].most_common()
            ),
        }
        for fail in BER_FAILURES
    }

    report = {
        "criteria": {
            "delta_mrr_threshold": args.delta_mrr_threshold,
            "majority": args.majority,
            "note": "Original frozen criteria; top-20 only (gold outside top-20 is null).",
        },
        "n_accepted_primary": len(accepted),
        "n_construction_only": len(construction_only),
        "construction_only_by_failure": dict(construction_only_by_fail),
        "held_out_buckets_by_failure_retriever": held_out_buckets,
        "mutation_level_transfer": {
            fail: dict(by_fail_mut[fail]) for fail in BER_FAILURES
        },
        "clean_gold_at1_attribute": gold_ceiling,
        "construction_reject_reasons": reject_summary,
        "bucket_definitions": {
            "challenged": "ΔMRR>0 or Recall@5 drops vs clean",
            "mutation_never_top20": "No failure chunk in held-out top-20",
            "enters_top20_gold_stable": (
                "Failure chunk in top-20 (or N/A for boundary splits) but gold "
                "best-rank unchanged within top-20; not challenged"
            ),
            "enters_top20_no_challenge_signal": (
                "Not challenged, mutation present / gold moved, but ΔMRR/R@5 "
                "challenge definition not met"
            ),
            "construction_only_no_heldout_challenge": (
                "Behaviorally accepted on construction (≥2/3) but neither E5 "
                "nor GTE challenged"
            ),
            "clean_gold_at1": (
                "Overlapping attribute: clean gold already rank-1 on that "
                "held-out retriever (hard to displace)"
            ),
        },
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    out_md.write_text(_to_markdown(report))
    print(f"accepted={len(accepted)} construction_only={len(construction_only)}")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Failure taxonomy (offline)",
        "",
        "Original criteria (ΔMRR≥0.10, ≥2/3 construction). "
        "Ranks limited to top-20 (consistent with the experiment).",
        "",
        f"Accepted primary mutations: **{report['n_accepted_primary']}**. "
        f"Construction-only (no held-out challenge on E5 or GTE): "
        f"**{report['n_construction_only']}** "
        f"({_fmt_pct(report['n_construction_only'] / report['n_accepted_primary']) if report['n_accepted_primary'] else 'n/a'}).",
        "",
        "## Held-out non-transfer buckets (accepted set)",
        "",
        "| Failure | Retriever | Challenged | Never Top-20 | Top-20 gold stable | Top-20 no challenge | n |",
        "|---|---|---|---|---|---|---|",
    ]
    buckets = report["held_out_buckets_by_failure_retriever"]
    for fail in BER_FAILURES:
        for ret in HELD_OUT:
            b = buckets[fail][ret]
            c = b["counts"]
            n = b["n"] or 1
            lines.append(
                "| {f} | {r} | {ch} | {nt} | {gs} | {nc} | {n} |".format(
                    f=fail,
                    r=ret,
                    ch=_fmt_pct(c.get("challenged", 0) / n),
                    nt=_fmt_pct(c.get("mutation_never_top20", 0) / n),
                    gs=_fmt_pct(c.get("enters_top20_gold_stable", 0) / n),
                    nc=_fmt_pct(c.get("enters_top20_no_challenge_signal", 0) / n),
                    n=b["n"],
                )
            )

    lines += [
        "",
        "## Construction-only transfer gap (mutation-level)",
        "",
        "| Failure | Construction-only | Transfers to ≥1 held-out |",
        "|---|---|---|",
    ]
    co = report["construction_only_by_failure"]
    ml = report["mutation_level_transfer"]
    for fail in BER_FAILURES:
        lines.append(
            f"| {fail} | {co.get(fail, 0)} | "
            f"{ml.get(fail, {}).get('transfers_to_at_least_one_heldout', 0)} |"
        )

    lines += [
        "",
        "## Clean gold already @1 (attribute on accepted × held-out)",
        "",
        "| Failure | Retriever | Clean gold@1 rate | Non-transfer ∧ gold@1 |",
        "|---|---|---|---|",
    ]
    gc = report["clean_gold_at1_attribute"]
    for fail in BER_FAILURES:
        for ret in HELD_OUT:
            row = gc[fail][ret]
            lines.append(
                f"| {fail} | {ret} | {_fmt_pct(row['clean_gold_at1_rate'])} | "
                f"{_fmt_pct(row['nontransfer_and_clean_gold_at1_rate'])} |"
            )

    lines += [
        "",
        "## Construction reject reasons (semantic pass, not accepted)",
        "",
        "| Failure | n_rejected | Per-retriever fail reasons (top) |",
        "|---|---|---|",
    ]
    for fail in BER_FAILURES:
        rs = report["construction_reject_reasons"][fail]
        top = ", ".join(
            f"{k}={v}" for k, v in list(rs["per_retriever_fail_reasons"].items())[:5]
        )
        lines.append(f"| {fail} | {rs['n_rejected']} | {top} |")

    lines += [
        "",
        "## Bucket definitions",
        "",
    ]
    for k, v in report["bucket_definitions"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
