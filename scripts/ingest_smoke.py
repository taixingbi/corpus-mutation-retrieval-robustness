#!/usr/bin/env python3
"""Copy a smoke subset from a local RAGFailBench checkout into data/runs/smoke/0_base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ragfailbench-root", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("data/runs/smoke/0_base"))
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--extra-chunks", type=int, default=200)
    args = p.parse_args()

    seeds_path = (
        args.ragfailbench_root
        / "data/runs/pilot_stability_s42/6_final/clean_seeds.jsonl"
    )
    chunks_path = (
        args.ragfailbench_root
        / "data/runs/pilot_stability_corpus/3_processed/chunks.jsonl"
    )
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = []
    with seeds_path.open() as f:
        for i, line in enumerate(f):
            if i >= args.n_seeds:
                break
            seeds.append(json.loads(line))
    gold_ids = {s["source"]["chunk_id"] for s in seeds}
    page_ids = {s["source"]["page_id"] for s in seeds}

    selected: dict[str, dict] = {}
    with chunks_path.open() as f:
        for i, line in enumerate(f):
            c = json.loads(line)
            if c["chunk_id"] in gold_ids or c["page_id"] in page_ids or i < args.extra_chunks:
                selected[c["chunk_id"]] = c

    with (args.out / "clean_seeds.jsonl").open("w") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with (args.out / "chunks.jsonl").open("w") as f:
        for c in selected.values():
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(seeds)} seeds and {len(selected)} chunks to {args.out}")


if __name__ == "__main__":
    main()
