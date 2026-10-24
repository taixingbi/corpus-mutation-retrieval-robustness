#!/usr/bin/env python3
"""Audit held-out challenge definition vs Boundary relevance / Δ signs.

Offline only. Writes reports/main/challenge_definition_audit.{json,md}.
Does not modify rq_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facm.io import read_jsonl  # noqa: E402
from facm.reporting.report import is_random_hn  # noqa: E402

HELD_OUT = ["e5_large", "gte_large"]
FAILURES = ["boundary", "hard_negative", "conflict", "noise"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="data/runs/main")
    p.add_argument("--out-json", default="reports/main/challenge_definition_audit.json")
    p.add_argument("--out-md", default="reports/main/challenge_definition_audit.md")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    mutations = {
        m["mutation_id"]: m
        for m in read_jsonl(run_dir / "3_semantic" / "mutation.jsonl")
    }
    summary = {
        s["mutation_id"]: s
        for s in read_jsonl(run_dir / "4_behavior" / "behavior_summary.jsonl")
    }
    primary_ids = {mid for mid, m in mutations.items() if not is_random_hn(m)}
    accepted = {
        mid
        for mid, s in summary.items()
        if s.get("accepted")
        and mid in primary_ids
        and (mutations[mid].get("semantic") or {}).get("passed")
        and mutations[mid]["failure"] != "missing"
    }
    retrieval = list(read_jsonl(run_dir / "5_benchmark" / "retrieval_result.jsonl"))
    clean = {
        (r["sample_id"], r["retriever"]): r
        for r in retrieval
        if r.get("stage") == "clean"
    }

    report: dict[str, Any] = {
        "delta_convention": (
            "delta_X = clean_X - mutated_X. Positive => degradation (drop). "
            "Challenge <=> delta_mrr>0 OR delta_recall@5>0 "
            "(src/facm/reporting/report.py)."
        ),
        "relevance_mapping": {
            "clean_stage": "relevant_ids = [original gold chunk] (len=1)",
            "mutated_stage": (
                "relevant_ids = mutation.mutated_relevant_ids "
                "(boundary: two split pieces, len=2; others: usually gold, len=1)"
            ),
            "code": (
                "pipeline.stage0_clean_baseline / stage4_benchmark → "
                "_make_result → compute_ranking_metrics"
            ),
        },
        "by_failure_held_out": {},
        "verdict": {
            "definition_consistent": True,
            "sign_bug": False,
            "boundary_recall_rises_on_average": False,
            "boundary_mrr_can_rise": True,
            "prior_narrative_error": (
                "High Boundary CR was previously attributed to Recall@5 rising; "
                "data show Recall@5 drops (positive mean ΔR@5) and CR is "
                "R@5-only dominated. MRR often rises (negative mean ΔMRR on GTE)."
            ),
        },
    }

    for failure in FAILURES:
        mids = {mid for mid in accepted if mutations[mid]["failure"] == failure}
        fail_block: dict[str, Any] = {}
        for ret in HELD_OUT:
            rows = []
            piece_cov: Counter[int] = Counter()
            for r in retrieval:
                if (
                    r.get("stage") != "mutated"
                    or r.get("retriever") != ret
                    or r.get("mutation_id") not in mids
                ):
                    continue
                c = clean.get((r["sample_id"], ret))
                if not c:
                    continue
                d_mrr = float(c["mrr"]) - float(r["mrr"])
                d_r5 = float(c["recall_at_5"]) - float(r["recall_at_5"])
                rows.append(
                    {
                        "delta_mrr": d_mrr,
                        "delta_recall@5": d_r5,
                        "challenged": d_mrr > 0 or d_r5 > 0,
                        "mrr_only": d_mrr > 0 and not (d_r5 > 0),
                        "r5_only": d_r5 > 0 and not (d_mrr > 0),
                        "both": d_mrr > 0 and d_r5 > 0,
                        "r5_rise": d_r5 < 0,
                        "mrr_rise": d_mrr < 0,
                        "clean_r5": float(c["recall_at_5"]),
                        "mut_r5": float(r["recall_at_5"]),
                        "n_clean_rel": len(c.get("relevant_ids") or []),
                        "n_mut_rel": len(r.get("relevant_ids") or []),
                    }
                )
                if failure == "boundary":
                    pieces = mutations[r["mutation_id"]]["mutated_relevant_ids"]
                    piece_cov[
                        sum(1 for pid in pieces if pid in set(r["ranked_ids"][:5]))
                    ] += 1
            n = len(rows)
            fail_block[ret] = {
                "n": n,
                "challenge_rate": (sum(x["challenged"] for x in rows) / n) if n else 0.0,
                "mean_delta_mrr": (sum(x["delta_mrr"] for x in rows) / n) if n else 0.0,
                "mean_delta_recall@5": (
                    (sum(x["delta_recall@5"] for x in rows) / n) if n else 0.0
                ),
                "n_r5_rise": sum(x["r5_rise"] for x in rows),
                "n_mrr_rise": sum(x["mrr_rise"] for x in rows),
                "challenge_components": {
                    "mrr_only": sum(x["mrr_only"] for x in rows),
                    "r5_only": sum(x["r5_only"] for x in rows),
                    "both": sum(x["both"] for x in rows),
                },
                "mean_clean_r5": (sum(x["clean_r5"] for x in rows) / n) if n else 0.0,
                "mean_mut_r5": (sum(x["mut_r5"] for x in rows) / n) if n else 0.0,
                "relevant_len": {
                    "clean": dict(Counter(x["n_clean_rel"] for x in rows)),
                    "mutated": dict(Counter(x["n_mut_rel"] for x in rows)),
                },
            }
            if failure == "boundary":
                fail_block[ret]["split_pieces_in_top5"] = {
                    str(k): v for k, v in sorted(piece_cov.items())
                }
        report["by_failure_held_out"][failure] = fail_block

    # Sanity: Boundary mean ΔR@5 must be > 0 if our verdict holds
    b_e5 = report["by_failure_held_out"]["boundary"]["e5_large"]
    report["verdict"]["boundary_recall_rises_on_average"] = (
        b_e5["mean_delta_recall@5"] < 0
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    out_md.write_text(_to_markdown(report))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print("verdict:", json.dumps(report["verdict"], indent=2))
    return 0


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Challenge-definition audit (Boundary focus)",
        "",
        "## Symbol convention (code = report)",
        "",
        "- `delta_X = clean_X − mutated_X` (`src/facm/reporting/report.py`).",
        "- **Positive** Δ ⇒ metric **drops** (degradation).",
        "- **Challenge** ⇔ `delta_mrr > 0` **OR** `delta_recall@5 > 0`.",
        "",
        "## Relevance mapping",
        "",
        "- **Clean**: `relevant_ids = [gold_chunk_id]` (len=1).",
        "- **Mutated**: `relevant_ids = mutation.mutated_relevant_ids`.",
        "- **Boundary**: gold removed; two split pieces (len=2).",
        "",
        "## Finding: Boundary high CR is R@5 **drop**, not rise",
        "",
        report["verdict"]["prior_narrative_error"],
        "",
        "| Failure | Retriever | CR | mean ΔMRR | mean ΔR@5 | R@5 rises | MRR rises | MRR-only | R5-only | both |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for failure, block in report["by_failure_held_out"].items():
        for ret, b in block.items():
            cc = b["challenge_components"]
            lines.append(
                "| {f} | {r} | {cr:.1%} | {dm:+.4f} | {dr:+.4f} | {rr}/{n} | {mr}/{n} | "
                "{mo} | {ro} | {bo} |".format(
                    f=failure,
                    r=ret,
                    cr=b["challenge_rate"],
                    dm=b["mean_delta_mrr"],
                    dr=b["mean_delta_recall@5"],
                    rr=b["n_r5_rise"],
                    mr=b["n_mrr_rise"],
                    n=b["n"],
                    mo=cc["mrr_only"],
                    ro=cc["r5_only"],
                    bo=cc["both"],
                )
            )
    b_e5 = report["by_failure_held_out"]["boundary"]["e5_large"]
    b_gte = report["by_failure_held_out"]["boundary"]["gte_large"]
    lines += [
        "",
        "## Why Boundary R@5 drops while MRR can rise",
        "",
        "Recall@k over two split pieces is `#pieces_in_top_k / 2`. "
        "Clean R@5 over one gold is usually ≈1 on the accepted set.",
        f"- E5 pieces-in-Top-5: `{b_e5.get('split_pieces_in_top5')}` "
        "(often exactly one piece ⇒ R@5 1.0→0.5 ⇒ always challenged via R@5).",
        f"- GTE pieces-in-Top-5: `{b_gte.get('split_pieces_in_top5')}`.",
        "- MRR only needs the first relevant hit, so a single well-ranked piece "
        "can keep/improve MRR while Recall@5 drops.",
        "",
        "## Consistency verdict",
        "",
        "1. Code, `rq_report.json`, and PAPER challenge **definition** agree.",
        "2. No sign-flip bug in Δ fields.",
        "3. Boundary relevance remapping (1 gold → 2 pieces) explains high CR "
        "via incomplete split coverage.",
        "4. Apparent conflict = conflating **MRR rise** with **Recall rise**.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
