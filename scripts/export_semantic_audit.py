#!/usr/bin/env python3
"""Sample mutations for human semantic audit (default 50 / failure)."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mutations",
        default="data/runs/main/3_semantic/mutation.jsonl",
        help="Semantic-validated (or raw) mutation.jsonl",
    )
    p.add_argument("--out", default="reports/audit/semantic_audit_sample.jsonl")
    p.add_argument("--per-failure", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    by_fail: dict[str, list[dict]] = defaultdict(list)
    path = Path(args.mutations)
    for line in path.open():
        if not line.strip():
            continue
        m = json.loads(line)
        by_fail[m["failure"]].append(m)

    rng = random.Random(args.seed)
    out_rows: list[dict] = []
    for failure, rows in sorted(by_fail.items()):
        rng.shuffle(rows)
        take = rows[: args.per_failure]
        for m in take:
            out_rows.append(
                {
                    "mutation_id": m["mutation_id"],
                    "sample_id": m["sample_id"],
                    "failure": m["failure"],
                    "question": m.get("question", ""),
                    "gold_answer": m.get("gold_answer", ""),
                    "semantic_passed": (m.get("semantic") or {}).get("passed"),
                    "added_texts": [
                        (m.get("chunk_payloads") or {})
                        .get(cid, {})
                        .get("text", "")
                        if isinstance((m.get("chunk_payloads") or {}).get(cid), dict)
                        else str((m.get("chunk_payloads") or {}).get(cid, ""))
                        for cid in (m.get("add") or [])
                    ],
                    "auditor_label": "",  # fill: ok | topical_fail | answer_leak | unnatural | other
                    "auditor_notes": "",
                }
            )
        print(f"{failure}: sampled {len(take)} / {len(rows)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} rows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
