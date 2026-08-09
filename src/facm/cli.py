from __future__ import annotations

import argparse
from pathlib import Path

from facm.config import load_config
from facm.pipeline import (
    rebuild_report,
    run_pipeline,
    stage0_clean_baseline,
    stage1_mutate,
    stage2_semantic,
    stage3_behavioral,
    stage4_benchmark,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="facm",
        description=(
            "Behaviorally Validated Corpus Mutation for Retrieval Robustness Evaluation "
            "(Generate → Validate → Transfer)"
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/smoke.yaml",
        help="Path to YAML config",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("run", help="Run Stage 0–4 (Generate → Validate → Transfer)")
    p_all.add_argument("--skip-llm", action="store_true")

    sub.add_parser("stage0", help="Clean baseline retrieval")
    p1 = sub.add_parser("mutate", help="Stage 1 mutation generation")
    p1.add_argument("--skip-llm", action="store_true")
    p2 = sub.add_parser("validate-semantic", help="Stage 2 semantic validation")
    p2.add_argument("--skip-llm", action="store_true")
    sub.add_parser("validate-behavior", help="Stage 3 behavioral validation")
    sub.add_parser("benchmark", help="Stage 4 held-out transfer + RQ report")
    sub.add_parser(
        "rebuild-report",
        help=(
            "Recompute behavioral acceptance + RQ report from saved "
            "clean/mutated rankings (no LLM / no re-retrieval)"
        ),
    )

    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))

    if args.cmd == "run":
        run_pipeline(cfg, skip_llm=args.skip_llm)
    elif args.cmd == "stage0":
        stage0_clean_baseline(cfg)
    elif args.cmd == "mutate":
        stage1_mutate(cfg, require_llm=not args.skip_llm)
    elif args.cmd == "validate-semantic":
        stage2_semantic(cfg, use_llm=not args.skip_llm)
    elif args.cmd == "validate-behavior":
        stage3_behavioral(cfg)
    elif args.cmd == "benchmark":
        stage4_benchmark(cfg)
    elif args.cmd == "rebuild-report":
        rebuild_report(cfg)
    else:
        parser.error(f"unknown command {args.cmd}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
