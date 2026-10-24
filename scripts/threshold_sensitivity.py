#!/usr/bin/env python3
"""Offline ΔMRR / majority threshold sensitivity (no LLM, no re-retrieval).

Recomputes BER and held-out challenge rate from saved rankings under:
  Lenient   ΔMRR≥0.05, majority≥2/3
  Original  ΔMRR≥0.10, majority≥2/3  (frozen main experiment)
  Strict    ΔMRR≥0.15, majority≥2/3
  Consensus ΔMRR≥0.10, majority=3/3

Does not overwrite data/runs/main/4_behavior/ or reports/main/rq_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facm.io import load_models, read_jsonl  # noqa: E402
from facm.metrics.stats import bootstrap_mean_ci  # noqa: E402
from facm.reporting.report import is_random_hn  # noqa: E402
from facm.schemas.models import Mutation  # noqa: E402
from facm.validation.behavioral import (  # noqa: E402
    aggregate_acceptance,
    validate_behavior_for_retriever,
)

CONFIGS = [
    {"name": "Lenient", "delta_mrr_threshold": 0.05, "majority": 2},
    {"name": "Original", "delta_mrr_threshold": 0.10, "majority": 2},
    {"name": "Strict", "delta_mrr_threshold": 0.15, "majority": 2},
    {"name": "Consensus", "delta_mrr_threshold": 0.10, "majority": 3},
]

CONSTRUCTION = ["bm25", "bge_base", "hybrid"]
HELD_OUT = ["e5_large", "gte_large"]
BER_FAILURES = ["hard_negative", "conflict", "boundary", "noise"]


def _load_rank_indexes(
    retrieval: list[dict],
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    clean: dict[tuple[str, str], dict] = {}
    mut: dict[tuple[str, str], dict] = {}
    for r in retrieval:
        if r.get("stage") == "clean":
            clean[(r["sample_id"], r["retriever"])] = r
        elif r.get("stage") == "mutated" and r.get("mutation_id"):
            mut[(r["mutation_id"], r["retriever"])] = r
    return clean, mut


def _accept_mutations(
    mutations: list[Mutation],
    clean_idx: dict[tuple[str, str], dict],
    mut_idx: dict[tuple[str, str], dict],
    *,
    delta_mrr_threshold: float,
    majority: int,
    top_k: int,
) -> dict[str, dict[str, Any]]:
    """Return mid -> {accepted, reason, failure, semantic_passed, primary}."""
    out: dict[str, dict[str, Any]] = {}
    for mutation in mutations:
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
                    top_k=top_k,
                    delta_mrr_threshold=delta_mrr_threshold,
                )
            )
        if not per:
            continue
        ok, _, reason = aggregate_acceptance(per, majority=majority)
        mdict = mutation.model_dump(mode="json")
        primary = not is_random_hn(mdict)
        sem = bool(mutation.semantic and mutation.semantic.passed)
        out[mutation.mutation_id] = {
            "accepted": bool(ok and sem and primary),
            "construction_majority": bool(ok),
            "reason": reason,
            "failure": mutation.failure,
            "semantic_passed": sem,
            "primary": primary,
            "sample_id": mutation.sample_id,
        }
    return out


def _challenged(clean_row: dict, mut_row: dict) -> bool:
    return (
        float(clean_row["mrr"]) - float(mut_row["mrr"]) > 0.0
        or float(mut_row["recall_at_5"]) < float(clean_row["recall_at_5"])
    )


def _metrics_for_config(
    decisions: dict[str, dict[str, Any]],
    clean_idx: dict[tuple[str, str], dict],
    mut_idx: dict[tuple[str, str], dict],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    # BER denominators: primary + semantic_passed + failure != missing
    by_fail_sem: dict[str, list[str]] = defaultdict(list)
    by_fail_acc: dict[str, list[str]] = defaultdict(list)
    for mid, d in decisions.items():
        if not d["primary"] or d["failure"] == "missing":
            continue
        if d["semantic_passed"]:
            by_fail_sem[d["failure"]].append(mid)
            if d["accepted"]:
                by_fail_acc[d["failure"]].append(mid)

    ber: dict[str, Any] = {}
    for failure in BER_FAILURES:
        denom = by_fail_sem.get(failure, [])
        eff = by_fail_acc.get(failure, [])
        ber[failure] = {
            "semantically_valid": len(denom),
            "behaviorally_effective": len(eff),
            "behavioral_effectiveness_rate": (len(eff) / len(denom)) if denom else 0.0,
        }

    pooled_sem = [mid for f in BER_FAILURES for mid in by_fail_sem.get(f, [])]
    pooled_acc = [mid for f in BER_FAILURES for mid in by_fail_acc.get(f, [])]
    ber["pooled"] = {
        "semantically_valid": len(pooled_sem),
        "behaviorally_effective": len(pooled_acc),
        "behavioral_effectiveness_rate": (
            (len(pooled_acc) / len(pooled_sem)) if pooled_sem else 0.0
        ),
    }

    # Held-out challenge on accepted set
    held_out: dict[str, Any] = {}
    for failure in BER_FAILURES + ["pooled"]:
        mids = set(pooled_acc if failure == "pooled" else by_fail_acc.get(failure, []))
        ret_block: dict[str, Any] = {}
        flags_all: list[float] = []
        for ret in HELD_OUT:
            flags: list[float] = []
            for mid in mids:
                d = decisions[mid]
                c = clean_idx.get((d["sample_id"], ret))
                m = mut_idx.get((mid, ret))
                if c is None or m is None:
                    continue
                flags.append(1.0 if _challenged(c, m) else 0.0)
            flags_all.extend(flags)
            ret_block[ret] = {
                "n": len(flags),
                "challenge_rate": bootstrap_mean_ci(
                    flags, n_boot=n_boot, seed=seed + hash(failure + ret) % 10_000
                ),
            }
        ret_block["mean_e5_gte"] = {
            "n": len(flags_all),
            "challenge_rate": bootstrap_mean_ci(
                flags_all, n_boot=n_boot, seed=seed + hash(failure) % 10_000
            ),
        }
        held_out[failure] = ret_block

    return {
        "n_accepted": len(pooled_acc),
        "ber_by_failure": ber,
        "held_out_challenge": held_out,
    }


def _check_original(
    metrics: dict[str, Any], frozen_path: Path
) -> dict[str, Any]:
    if not frozen_path.exists():
        return {"ok": False, "error": f"missing {frozen_path}"}
    frozen = json.loads(frozen_path.read_text())
    ber_f = frozen["rq2_behavioral_effectiveness"]["ber_by_failure"]
    rq3 = frozen["rq3_transferability"]["held_out_by_failure"]
    mismatches: list[str] = []
    for failure in BER_FAILURES:
        got = metrics["ber_by_failure"][failure]
        exp = ber_f[failure]
        if got["behaviorally_effective"] != exp["behaviorally_effective"]:
            mismatches.append(
                f"BER n_eff {failure}: got {got['behaviorally_effective']} "
                f"expected {exp['behaviorally_effective']}"
            )
        if abs(
            got["behavioral_effectiveness_rate"]
            - exp["behavioral_effectiveness_rate"]
        ) > 1e-9:
            mismatches.append(
                f"BER rate {failure}: got {got['behavioral_effectiveness_rate']} "
                f"expected {exp['behavioral_effectiveness_rate']}"
            )
        for ret in HELD_OUT:
            if failure not in rq3 or ret not in rq3[failure]:
                continue
            g = metrics["held_out_challenge"][failure][ret]
            e = rq3[failure][ret]
            if g["n"] != e["n"]:
                mismatches.append(
                    f"challenge n {failure}/{ret}: got {g['n']} expected {e['n']}"
                )
            if abs(g["challenge_rate"]["mean"] - e["challenge_rate"]["mean"]) > 1e-9:
                mismatches.append(
                    f"challenge mean {failure}/{ret}: got "
                    f"{g['challenge_rate']['mean']} expected "
                    f"{e['challenge_rate']['mean']}"
                )
    return {"ok": not mismatches, "mismatches": mismatches}


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _fmt_ci(ci: dict[str, Any]) -> str:
    return (
        f"{_fmt_pct(ci['mean'])} "
        f"[{_fmt_pct(ci['ci_low'])}, {_fmt_pct(ci['ci_high'])}]"
    )


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Threshold sensitivity (offline)",
        "",
        "Recomputed from saved `5_benchmark/retrieval_result.jsonl` rankings. "
        "ΔMRR threshold applies to hard_negative and conflict only; "
        "noise/boundary criteria are fixed. Missing excluded from BER.",
        "",
        "## Configuration",
        "",
        "| Configuration | ΔMRR threshold | Retriever agreement |",
        "|---|---|---|",
    ]
    for c in report["configurations"]:
        maj = f"≥{c['majority']}/3" if c["majority"] < 3 else "3/3"
        lines.append(
            f"| {c['name']} | {c['delta_mrr_threshold']:.2f} | {maj} |"
        )
    lines += ["", "## BER by failure", "", "| Config | HN | Conflict | Boundary | Noise | Pooled | n_accepted |"]
    lines.append("|---|---|---|---|---|---|---|")
    for c in report["configurations"]:
        ber = c["metrics"]["ber_by_failure"]
        lines.append(
            "| {name} | {hn} | {cf} | {bd} | {nz} | {pl} | {n} |".format(
                name=c["name"],
                hn=_fmt_pct(ber["hard_negative"]["behavioral_effectiveness_rate"]),
                cf=_fmt_pct(ber["conflict"]["behavioral_effectiveness_rate"]),
                bd=_fmt_pct(ber["boundary"]["behavioral_effectiveness_rate"]),
                nz=_fmt_pct(ber["noise"]["behavioral_effectiveness_rate"]),
                pl=_fmt_pct(ber["pooled"]["behavioral_effectiveness_rate"]),
                n=c["metrics"]["n_accepted"],
            )
        )
    lines += [
        "",
        "## Held-out challenge rate (accepted set)",
        "",
        "| Config | Failure | E5 | GTE | Mean (E5+GTE) | n |",
        "|---|---|---|---|---|---|",
    ]
    for c in report["configurations"]:
        ho = c["metrics"]["held_out_challenge"]
        for failure in BER_FAILURES + ["pooled"]:
            block = ho[failure]
            n = block["e5_large"]["n"]
            lines.append(
                "| {cfg} | {f} | {e5} | {gte} | {mean} | {n} |".format(
                    cfg=c["name"],
                    f=failure,
                    e5=_fmt_ci(block["e5_large"]["challenge_rate"]),
                    gte=_fmt_ci(block["gte_large"]["challenge_rate"]),
                    mean=_fmt_ci(block["mean_e5_gte"]["challenge_rate"]),
                    n=n,
                )
            )
    sanity = report.get("original_matches_frozen")
    lines += ["", "## Sanity check (Original vs frozen rq_report.json)", ""]
    if sanity and sanity.get("ok"):
        lines.append("Original configuration **matches** frozen main RQ2/RQ3 numbers.")
    else:
        lines.append("Original configuration **mismatch**:")
        for m in (sanity or {}).get("mismatches") or ["unknown"]:
            lines.append(f"- {m}")
    lines += [
        "",
        "## Note",
        "",
        report.get(
            "note",
            "Challenge = ΔMRR>0 or Recall@5 drops vs clean on held-out retrievers.",
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="data/runs/main")
    p.add_argument("--out-json", default="reports/main/threshold_sensitivity.json")
    p.add_argument("--out-md", default="reports/main/threshold_sensitivity.md")
    p.add_argument("--frozen-report", default="reports/main/rq_report.json")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    mut_path = run_dir / "3_semantic" / "mutation.jsonl"
    if not mut_path.exists():
        mut_path = run_dir / "2_mutations" / "mutation.jsonl"
    ret_path = run_dir / "5_benchmark" / "retrieval_result.jsonl"
    mutations = load_models(mut_path, Mutation)
    retrieval = list(read_jsonl(ret_path))
    clean_idx, mut_idx = _load_rank_indexes(retrieval)

    configs_out: list[dict[str, Any]] = []
    original_metrics: dict[str, Any] | None = None
    for cfg in CONFIGS:
        print(
            f"[sensitivity] {cfg['name']}: "
            f"ΔMRR≥{cfg['delta_mrr_threshold']}, majority={cfg['majority']}"
        )
        decisions = _accept_mutations(
            mutations,
            clean_idx,
            mut_idx,
            delta_mrr_threshold=cfg["delta_mrr_threshold"],
            majority=cfg["majority"],
            top_k=args.top_k,
        )
        metrics = _metrics_for_config(
            decisions,
            clean_idx,
            mut_idx,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        row = {**cfg, "metrics": metrics}
        configs_out.append(row)
        if cfg["name"] == "Original":
            original_metrics = metrics

    sanity = (
        _check_original(original_metrics, Path(args.frozen_report))
        if original_metrics
        else {"ok": False, "error": "no Original"}
    )
    print(
        "[sensitivity] Original vs frozen:",
        "OK" if sanity.get("ok") else f"MISMATCH {sanity}",
    )

    report = {
        "source_mutations": str(mut_path),
        "source_retrieval": str(ret_path),
        "configurations": configs_out,
        "original_matches_frozen": sanity,
        "note": (
            "ΔMRR threshold gates hard_negative and conflict only. "
            "Noise/boundary rules fixed. Missing excluded from BER. "
            "Random HN excluded from primary metrics. "
            "Challenge = ΔMRR>0 or Recall@5 drops vs clean."
        ),
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    out_md.write_text(_to_markdown(report))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0 if sanity.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
