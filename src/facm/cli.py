from __future__ import annotations

import argparse
from pathlib import Path

from facm.config import load_config
from facm.pipeline import (
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
        description="Failure-aware Corpus Mutation for Retrieval and Reranking Evaluation",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/smoke.yaml",
        help="Path to YAML config",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("run", help="Run Stage 0–4 end-to-end")
    p_all.add_argument("--skip-llm", action="store_true")
    p_all.add_argument("--skip-rerank", action="store_true")

    sub.add_parser("stage0", help="Clean baseline retrieval")
    p1 = sub.add_parser("mutate", help="Stage 1 mutation generation")
    p1.add_argument("--skip-llm", action="store_true")
    p2 = sub.add_parser("validate-semantic", help="Stage 2 semantic validation")
    p2.add_argument("--skip-llm", action="store_true")
    sub.add_parser("validate-behavior", help="Stage 3 behavioral validation")
    p4 = sub.add_parser("benchmark", help="Stage 4 held-out + rerank")
    p4.add_argument("--skip-rerank", action="store_true")

    args = parser.parse_args(argv)
    cfg = load_config(Path(args.config))

    if args.cmd == "run":
        run_pipeline(cfg, skip_llm=args.skip_llm, skip_rerank=args.skip_rerank)
    elif args.cmd == "stage0":
        stage0_clean_baseline(cfg)
    elif args.cmd == "mutate":
        stage1_mutate(cfg, require_llm=not args.skip_llm)
    elif args.cmd == "validate-semantic":
        stage2_semantic(cfg, use_llm=not args.skip_llm)
    elif args.cmd == "validate-behavior":
        stage3_behavioral(cfg)
    elif args.cmd == "benchmark":
        stage4_benchmark(cfg, run_rerank=not args.skip_rerank)
    else:
        parser.error(f"unknown command {args.cmd}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
