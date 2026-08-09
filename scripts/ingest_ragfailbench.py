#!/usr/bin/env python3
"""Ingest RAGFailBench chunks + clean seeds into a FACM run's 0_base/."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _seed_key(seed: dict) -> tuple[str, str, str]:
    source = seed.get("source") or {}
    return (
        str(seed.get("question") or "").strip(),
        str(source.get("chunk_id") or "").strip(),
        str(seed.get("gold_answer") or "").strip(),
    )


def _load_seeds_from_runs(
    ragfailbench_root: Path,
    seeds_runs: list[str],
    n_seeds: int | None,
) -> tuple[list[dict], list[str]]:
    """Load and dedupe seeds across runs; remap sample_id to contiguous seed_XXXXXX."""
    seen: dict[tuple[str, str, str], dict] = {}
    source_paths: list[str] = []
    for run in seeds_runs:
        path = ragfailbench_root / f"data/runs/{run}/6_final/clean_seeds.jsonl"
        if not path.exists():
            raise SystemExit(f"missing seeds: {path}")
        source_paths.append(str(path))
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                seed = json.loads(line)
                key = _seed_key(seed)
                if key in seen:
                    continue
                seen[key] = seed
                if n_seeds is not None and len(seen) >= n_seeds:
                    break
        if n_seeds is not None and len(seen) >= n_seeds:
            break

    seeds: list[dict] = []
    for i, seed in enumerate(seen.values()):
        out = dict(seed)
        out["sample_id"] = f"seed_{i:06d}"
        seeds.append(out)
    return seeds, source_paths


def ingest(
    *,
    ragfailbench_root: Path,
    out: Path,
    n_seeds: int | None,
    full_corpus: bool,
    extra_chunks: int,
    seeds_runs: list[str],
    corpus_run: str,
) -> None:
    if not seeds_runs:
        raise SystemExit("at least one seeds run is required")

    chunks_path = (
        ragfailbench_root / f"data/runs/{corpus_run}/3_processed/chunks.jsonl"
    )
    if not chunks_path.exists():
        raise SystemExit(f"missing chunks: {chunks_path}")

    out.mkdir(parents=True, exist_ok=True)
    seeds, source_seed_paths = _load_seeds_from_runs(
        ragfailbench_root, seeds_runs, n_seeds
    )

    if full_corpus:
        dest = out / "chunks.jsonl"
        shutil.copyfile(chunks_path, dest)
        n_chunks = sum(1 for _ in dest.open())
    else:
        gold_ids = {s["source"]["chunk_id"] for s in seeds}
        page_ids = {s["source"]["page_id"] for s in seeds}
        selected: dict[str, dict] = {}
        with chunks_path.open() as f:
            for i, line in enumerate(f):
                c = json.loads(line)
                if (
                    c["chunk_id"] in gold_ids
                    or c["page_id"] in page_ids
                    or i < extra_chunks
                ):
                    selected[c["chunk_id"]] = c
        with (out / "chunks.jsonl").open("w") as f:
            for c in selected.values():
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        n_chunks = len(selected)

    with (out / "clean_seeds.jsonl").open("w") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    meta = {
        "source_seeds": source_seed_paths,
        "source_chunks": str(chunks_path),
        "seeds_runs": seeds_runs,
        "n_seeds": len(seeds),
        "n_chunks": n_chunks,
        "full_corpus": full_corpus,
        "dedupe_key": "question+gold_chunk_id+gold_answer",
    }
    (out / "ingest_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(seeds)} seeds and {n_chunks} chunks → {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ragfailbench-root", required=True, type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-seeds", type=int, default=None, help="Cap seeds (default: all)")
    p.add_argument(
        "--full-corpus",
        action="store_true",
        help="Copy the full chunks.jsonl (recommended)",
    )
    p.add_argument("--extra-chunks", type=int, default=200)
    p.add_argument(
        "--seeds-run",
        default=None,
        help="Single RAGFailBench seeds run (legacy)",
    )
    p.add_argument(
        "--seeds-runs",
        default=None,
        help="Comma-separated seeds runs to merge/dedupe "
        "(e.g. pilot_stability_s42,pilot_stability_s123,pilot_stability_s2026)",
    )
    p.add_argument("--corpus-run", default="pilot_stability_corpus")
    args = p.parse_args()

    if args.seeds_runs:
        seeds_runs = [r.strip() for r in args.seeds_runs.split(",") if r.strip()]
    elif args.seeds_run:
        seeds_runs = [args.seeds_run]
    else:
        seeds_runs = ["pilot_stability_s42"]

    ingest(
        ragfailbench_root=args.ragfailbench_root,
        out=args.out,
        n_seeds=args.n_seeds,
        full_corpus=args.full_corpus,
        extra_chunks=args.extra_chunks,
        seeds_runs=seeds_runs,
        corpus_run=args.corpus_run,
    )


if __name__ == "__main__":
    main()
