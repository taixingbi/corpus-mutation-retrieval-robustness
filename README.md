# FACM — Failure-aware Corpus Mutation for Retrieval and Reranking Evaluation

Generate → Validate → Benchmark. Corpus mutations are accepted only when they pass **semantic** and **behavioral** checks; held-out retrievers measure generalization.

## Pipeline

| Stage | Output | Notes |
|-------|--------|------|
| 0 Clean baseline | `1_clean/retrieval_result.jsonl` | BM25 / BGE / Hybrid + held-out E5 / GTE |
| 1 Mutate | `2_mutations/mutation.jsonl` | One failure per `(seed, failure)` |
| 2 Semantic | `3_semantic/mutation_passed.jsonl` | Rules + Bedrock LLM judge |
| 3 Behavioral | `4_behavior/mutation_accepted.jsonl` | Construction retrievers only |
| 4 Benchmark | `5_benchmark/*` + `reports/<run>/rq_report.json` | Held-out eval + ECS rerank + ablation |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Set FACM_INFERENCE_URL / FACM_RERANK_URL (see example_inference.md, example_reranker.md)
```

## Data ingest (from local RAGFailBench)

```bash
# Smoke subset
python scripts/ingest_ragfailbench.py \
  --ragfailbench-root /path/to/RAGFailBench \
  --out data/runs/smoke/0_base \
  --n-seeds 5

# Pilot: full chunk corpus + all seeds (or cap with --n-seeds)
python scripts/ingest_ragfailbench.py \
  --ragfailbench-root /path/to/RAGFailBench \
  --out data/runs/pilot/0_base \
  --full-corpus
```

## Run

```bash
# Smoke (5 seeds, MiniLM stand-in encoders)
facm -c configs/smoke.yaml run

# 50-seed pilot with real BGE / E5 / GTE (shares pilot/0_base)
facm -c configs/pilot_small.yaml run

# Full 200-seed pilot (1 attempt / failure)
facm -c configs/pilot.yaml run

# Scaled generation for RQ4 (~150 recoverable cases / failure)
# Oversamples noise×10 and hard_negative×6; shared Hybrid Top-20 pool
facm -c configs/pilot_scale.yaml run
```

## Schemas

Reused from [RAGFailBench](https://github.com/taixingbi/RAGFailBench/tree/main/schemas): `Chunk.json`, `CleanSeed.json`.

New: `Mutation.json`, `BehaviorValidation.json`, `RetrievalResult.json`, `RerankerResult.json`.

## Research questions

- **RQ1** Semantic Success Rate  
- **RQ2** Behavioral Success Rate (failure-specific criteria)  
- **RQ3** Held-out challenge rate (E5-large, GTE-large)  
- **RQ4** Rerank recovery on a **fixed Hybrid Top-K pool** (Hit@1, Recall@5, MRR@10, nDCG@10 + bootstrap CI / paired Wilcoxon).  
  `missing` is an **unrecoverable negative control** (gold removed); other failures are recoverable only if gold stays in the pool.

Ablation: Raw LLM → +Semantic → +Semantic+Behavioral on held-out challenge rate.

## Concurrency

Sync ThreadPool (not asyncio). Configure in YAML:

```yaml
llm_concurrency: 8       # mutate / semantic / stage0 / behavior / benchmark
embed_concurrency: 1     # dense encode parallelism (1 = one model at a time)
embed_batch_size: 128    # SentenceTransformer.encode batch size
embed_device: null       # null = auto (cuda > mps > cpu)
rerank_concurrency: 1    # ECS /v1/rerank in-flight limit (1 = serialized)
http_retries: 5          # chat + rerank retry on 502/503/timeout
rerank_pool_size: 20     # shared Hybrid Top-K for every reranker
mutation_oversample:     # attempts per (seed, failure); noise/HN need more
  noise: 10
  hard_negative: 6
```

`llm_concurrency: 1` forces fully sequential runs (debug).

Base-corpus embeddings are written to `{output_dir}/_embed_cache/` and reused across stage0/3/4 and re-runs (invalidate by deleting that folder if the corpus changes).
