#!/usr/bin/env python3
"""Ingest RAGFailBench chunks + clean seeds into a FACM run's 0_base/."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def ingest(
    *,
    ragfailbench_root: Path,
    out: Path,
    n_seeds: int | None,
    full_corpus: bool,
    extra_chunks: int,
    seeds_run: str,
    corpus_run: str,
) -> None:
    seeds_path = (
        ragfailbench_root / f"data/runs/{seeds_run}/6_final/clean_seeds.jsonl"
    )
    chunks_path = (
        ragfailbench_root / f"data/runs/{corpus_run}/3_processed/chunks.jsonl"
    )
    if not seeds_path.exists():
        raise SystemExit(f"missing seeds: {seeds_path}")
    if not chunks_path.exists():
        raise SystemExit(f"missing chunks: {chunks_path}")

    out.mkdir(parents=True, exist_ok=True)

    seeds: list[dict] = []
    with seeds_path.open() as f:
        for i, line in enumerate(f):
            if n_seeds is not None and i >= n_seeds:
                break
            if line.strip():
                seeds.append(json.loads(line))

    if full_corpus:
        # Fast path: copy entire corpus file
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
        "source_seeds": str(seeds_path),
        "source_chunks": str(chunks_path),
        "n_seeds": len(seeds),
        "n_chunks": n_chunks,
        "full_corpus": full_corpus,
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
        help="Copy the full chunks.jsonl (recommended for pilot)",
    )
    p.add_argument("--extra-chunks", type=int, default=200)
    p.add_argument("--seeds-run", default="pilot_stability_s42")
    p.add_argument("--corpus-run", default="pilot_stability_corpus")
    args = p.parse_args()
    ingest(
        ragfailbench_root=args.ragfailbench_root,
        out=args.out,
        n_seeds=args.n_seeds,
        full_corpus=args.full_corpus,
        extra_chunks=args.extra_chunks,
        seeds_run=args.seeds_run,
        corpus_run=args.corpus_run,
    )


if __name__ == "__main__":
    main()
