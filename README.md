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
2. **RQ2** — Behavioral Effectiveness Rate (BER); paired Clean→Mutated metrics  
3. **RQ3** — Held-out transfer; Raw → +Semantic → +Behavioral; Random vs LLM HN  

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

```bash
# Full main experiment
facm -c configs/main.yaml run

# Recompute behavior + RQ report from saved rankings (no LLM / no re-retrieval)
facm -c configs/main.yaml rebuild-report

# Human semantic audit sample
python scripts/export_semantic_audit.py --per-failure 50
```

Pilot archive: `facm -c configs/pilot.yaml run`  
Smoke: `facm -c configs/smoke.yaml run --skip-llm`

## Mutations

| Failure | Operation | Generation |
|---------|-----------|------------|
| Missing | remove gold | Rule (negative control) |
| Hard Negative | same entity/relation, wrong plausible fact | LLM |
| Hard Negative (baseline) | related corpus chunk | Random BM25 |
| Conflict | contradiction | LLM |
| Boundary | split gold → pieces A/B | Rule |
| Noise | distractors | Corpus/LLM + BM25 mine |

Behavioral criteria are **failure-specific** (e.g. HN outrank / Top-5 cross / ΔMRR≥0.10; Noise needs Top-5 entry + meaningful gold drop).

## Config

```yaml
llm_concurrency: 8
embed_concurrency: 1
embed_batch_size: 128
include_random_hn_baseline: true
mutation_oversample: { missing: 1, hard_negative: 1, conflict: 1, boundary: 1, noise: 1 }
```

Embeddings cache: `{output_dir}/_embed_cache/`.
