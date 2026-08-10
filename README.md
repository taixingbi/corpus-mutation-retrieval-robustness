# FACM — Behaviorally Validated Corpus Mutation for Retrieval Robustness Evaluation

**Story:** Generate → Validate → Transfer  

**Core question:** Can controlled corpus mutations create valid, behaviorally effective, and transferable retrieval failures?

**Finding:** Behavioral validation—not semantic validation alone—is the key step for constructing robust retrieval evaluation benchmarks.

See [PAPER.md](PAPER.md) for the locked design.

## Pipeline

| Stage | Output | RQ |
|-------|--------|-----|
| 0 Clean | `1_clean/retrieval_result.jsonl` | — |
| 1 Mutate | `2_mutations/mutation.jsonl` | one `(seed, failure)` (+ optional Random HN) |
| 2 Semantic | `3_semantic/mutation_passed.jsonl` | **RQ1** SVR |
| 3 Behavioral | `4_behavior/mutation_accepted.jsonl` | **RQ2** BER (BM25 / BGE / Hybrid) |
| 4 Held-out | `5_benchmark/*` + `reports/<run>/rq_report.json` | **RQ3** E5 / GTE + ablation |

Held-out: **E5-large**, **GTE-large** (never used for acceptance).

## Research questions

1. **RQ1** — Semantic Validity Rate (SVR) + human audit  
2. **RQ2** — Behavioral Effectiveness Rate (BER); paired Clean→Mutated (MRR, Recall@k, Top-k crossing)  
3. **RQ3** — Held-out transfer; Raw → +Semantic → +Behavioral; Random vs LLM HN  

Appendix (secondary): nDCG@5/10/20 — not used for acceptance.  
Missing is a **deterministic negative control** (excluded from BER).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # FACM_INFERENCE_* — see example_inference.md, model.md
```

## Data

```bash
python scripts/ingest_ragfailbench.py \
  --ragfailbench-root /path/to/RAGFailBench \
  --out data/runs/main/0_base \
  --full-corpus \
  --seeds-runs pilot_stability_s42,pilot_stability_s123,pilot_stability_s2026,pilot_v1
```

~531 unique seeds available by merging RAGFailBench runs. 1000 seeds needs more Paper-2 seed generation.

## Run

Canonical path is a **full main run** (after operator / validator changes):

```bash
facm -c configs/main.yaml run
```

After rankings exist, only re-aggregate metrics (no LLM / no re-retrieval):

```bash
facm -c configs/main.yaml rebuild-report
```

Human semantic audit sample (after main mutate+semantic):

```bash
python scripts/export_semantic_audit.py --per-failure 50
```

See [reports/audit/README.md](reports/audit/README.md).

Pilot: `facm -c configs/pilot.yaml run`  
Smoke: `facm -c configs/smoke.yaml run --skip-llm`

## Mutations

| Failure | Operation | Generation |
|---------|-----------|------------|
| Missing | remove gold | Rule (negative control) |
| Hard Negative | same entity/relation, wrong plausible fact | LLM |
| Hard Negative (baseline) | related corpus chunk | Random BM25 |
| Conflict | contradiction | LLM |
| Boundary | split gold through answer span → A/B | Rule |
| Noise | distractors | Corpus/LLM + BM25 mine |

Behavioral criteria are **failure-specific** (HN outrank / Top-5 cross / ΔMRR≥0.10; Conflict in Top-5 + effect; Noise Top-5 + Δrank≥3 or Top-5 cross; Boundary uses split relevance).

## Config

```yaml
llm_concurrency: 8
embed_concurrency: 1
embed_batch_size: 128
include_random_hn_baseline: true
mutation_oversample: { missing: 1, hard_negative: 1, conflict: 1, boundary: 1, noise: 1 }
```

Embeddings cache: `{output_dir}/_embed_cache/` (reused across runs).
