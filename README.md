# FACM — Behaviorally Validated Corpus Mutation for Retrieval Robustness Evaluation

**Story:** Generate → Validate → Transfer  

**Core question:** Can controlled corpus mutations create valid, behaviorally effective, and transferable retrieval failures?

See [PAPER.md](PAPER.md) for the locked design.

## Pipeline

| Stage | Output | RQ |
|-------|--------|-----|
| 0 Clean baseline | `1_clean/retrieval_result.jsonl` | — |
| 1 Mutate | `2_mutations/mutation.jsonl` | one `(seed, failure)` at a time |
| 2 Semantic | `3_semantic/mutation_passed.jsonl` | **RQ1** SVR |
| 3 Behavioral | `4_behavior/mutation_accepted.jsonl` | **RQ2** BER (BM25 / BGE / Hybrid only) |
| 4 Held-out transfer | `5_benchmark/*` + `reports/<run>/rq_report.json` | **RQ3** E5 / GTE + ablation |

E5-large and GTE-large are **held-out**: never used for mutation acceptance.

## Research questions

1. **RQ1 Validity** — Semantic Validity Rate (SVR) + human audit (30–50 / failure)  
2. **RQ2 Effectiveness** — Behavioral Effectiveness Rate (BER); paired Clean→Mutated Recall@k / MRR / Gold Rank Δ / Top-k crossing  
3. **RQ3 Transferability** — Held-out ΔMRR / ΔRecall@5 / Challenge Rate; ablation Raw → +Semantic → +Behavioral  

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# FACM_INFERENCE_URL for LLM mutate/judge (see example_inference.md)
# Model inventory: model.md
```

## Data (RAGFailBench)

```bash
python scripts/ingest_ragfailbench.py \
  --ragfailbench-root /path/to/RAGFailBench \
  --out data/runs/pilot/0_base \
  --full-corpus
```

## Run

```bash
facm -c configs/pilot.yaml run

# Rebuild RQ1–3 from existing JSONL
facm -c configs/pilot.yaml rebuild-report

# Human semantic audit sample
python scripts/export_semantic_audit.py \
  --mutations data/runs/pilot/3_semantic/mutation.jsonl \
  --out reports/audit/semantic_audit_sample.jsonl \
  --per-failure 50
```

Smoke (local MiniLM stand-ins): `facm -c configs/smoke.yaml run --skip-llm`

## Mutations (one failure per corpus state)

| Failure | Operation | Generation |
|---------|-----------|------------|
| Missing | remove gold | Rule |
| Hard Negative | add relevant-but-wrong | LLM + BM25 mine |
| Conflict | add contradiction | LLM |
| Boundary | split gold evidence | Rule |
| Noise | add distractors | Corpus/LLM + BM25 mine |

## Schemas

Reused from [RAGFailBench](https://github.com/taixingbi/RAGFailBench): `Chunk.json`, `CleanSeed.json`.  

New: `Mutation.json`, `BehaviorValidation.json`, `RetrievalResult.json`.

## Config knobs

```yaml
llm_concurrency: 8
embed_concurrency: 1
embed_batch_size: 128
mutation_oversample:
  hard_negative: 4
  noise: 6
```

Embeddings cache: `{output_dir}/_embed_cache/`.
