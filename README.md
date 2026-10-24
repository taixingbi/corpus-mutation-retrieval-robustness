# Behaviorally Validated Corpus Mutation for Retrieval Robustness Evaluation

**IEEE BigData 2026 camera-ready** — reviewer check entry point.

| | |
|--|--|
| **Paper PDF** | [IEEE BigData2026 - Behaviorally Validated Corpus Mutation for Retrieval Robustness Evaluation.pdf](IEEE%20BigData2026%20-%20Behaviorally%20Validated%20Corpus%20Mutation%20for%20Retrieval%20Robustness%20Evaluation.pdf) |
| **Frozen metrics** | [`reports/main/rq_report.json`](reports/main/rq_report.json) |
| **Design / paste notes** | [`PAPER.md`](PAPER.md) |
| **Models** | [`model.md`](model.md) |

**Story:** Generate → Semantic Validate → Behaviorally Validate → Transfer  
**Claim:** Semantic validity ≠ behavioral effectiveness ≠ cross-retriever transfer; all three must be measured separately.

Primary RQ1–RQ3 numbers are **frozen**. Offline analyses below do **not** re-run LLM, embedding, or retrieval.

---

## What to verify (map)

| Paper claim | Where to look |
|-------------|---------------|
| RQ1 SVR + human audit 99.6% | `rq_report.json` → `rq1_semantic_validity`; [`reports/audit/`](reports/audit/) |
| RQ2 BER (HN 75.9%, Conflict 70.1%, Boundary 65.7%, Noise 0.9%) | `rq_report.json` → `rq2_behavioral_effectiveness` |
| RQ3 held-out CR (HN/Conflict ~24%, Boundary 62–70%) | `rq_report.json` → `rq3_transferability` |
| Δ = clean − mutated (positive = degradation) | `src/facm/reporting/report.py`; CR = ΔMRR>0 ∨ ΔR@5>0 |
| Threshold sensitivity (Lenient=Original=Strict; Consensus shrinks set) | [`reports/main/threshold_sensitivity.md`](reports/main/threshold_sensitivity.md) |
| Failure taxonomy (non-transfer ≈ gold-stable in Top-20) | [`reports/main/failure_taxonomy.md`](reports/main/failure_taxonomy.md) |
| Boundary CR = Recall@5-only (E5 95/6/15; GTE 86/4/12) | [`reports/main/challenge_definition_audit.md`](reports/main/challenge_definition_audit.md) |
| LLM judge fail-**closed**; main-run triggers = 0 | `src/facm/validation/semantic.py`; audit note in `PAPER.md` |
| Human audit codebook | [`reports/audit/README.md`](reports/audit/README.md) |

---

## Frozen main results (quick)

**Scale:** 531 clean seeds → one mutation per failure (+ Random HN baseline).  
**Construction (acceptance):** BM25, BGE-base, Hybrid (≥2/3).  
**Held-out (never for acceptance):** E5-large, GTE-large.

| Failure | SVR (auto) | BER | E5 CR | GTE CR | n accepted |
|---------|------------|-----|-------|--------|------------|
| Hard negative | 94.3% | **75.9%** | 23.7% | 23.4% | 350 |
| Conflict | 98.3% | **70.1%** | 25.5% | 24.1% | 365 |
| Boundary † | 100% | **65.7%** | 70.3% | 61.8% | 165 |
| Noise | 100% | **0.9%** | — (n=5) | — | 5 |
| Missing | 100% | excluded (control) | — | — | — |

† Boundary CR is mostly **Recall@5 coverage** of two answer-span fragments, not MRR degradation — see challenge audit.

**Human audit:** 249/250 ok (99.6%); sole miss = 1 Random HN `topical_fail`.

**Filter ablation (pooled held-out CR):** Raw 35.3% → +Semantic 35.2% → +Semantic+Behavioral 32.3%.

---

## Artifact layout (main run)

```
data/runs/main/
  0_base/                 # corpus + clean seeds
  1_clean/                # clean rankings
  2_mutations/            # raw mutations
  3_semantic/             # RQ1  mutation.jsonl / mutation_passed.jsonl
  4_behavior/             # RQ2  behavior_* / mutation_accepted.jsonl
  5_benchmark/            # clean+mutated rankings (all retrievers)
reports/main/
  rq_report.json          # frozen RQ1–RQ3
  threshold_sensitivity.* # offline ΔMRR / majority sweep
  failure_taxonomy.*      # offline non-transfer buckets
  challenge_definition_audit.*  # Boundary CR components + Δ signs
reports/audit/            # human semantic audit sample + summary
```

---

## Offline checks (no re-run)

After cloning with `data/runs/main` + `reports/main` present:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Re-aggregate frozen rankings → must match rq_report.json (Original thresholds)
facm -c configs/main.yaml rebuild-report

# Camera-ready offline tables
python scripts/threshold_sensitivity.py   # Original row must match frozen BER/CR
python scripts/failure_taxonomy.py
python scripts/challenge_definition_audit.py
```

**Threshold sensitivity protocol:** rescore acceptance on **frozen construction rankings**, then score CR on **frozen held-out rankings**.

---

## Pipeline (for orientation)

| Stage | Output | RQ |
|-------|--------|-----|
| Mutate | `2_mutations/mutation.jsonl` | one `(seed, failure)` |
| Semantic | `3_semantic/` | **RQ1** SVR (+ LLM judge for HN/Conflict) |
| Behavioral | `4_behavior/` | **RQ2** BER on BM25 / BGE / Hybrid |
| Held-out | `5_benchmark/` + `rq_report.json` | **RQ3** E5 / GTE |

Operators: Missing (rule control) · Hard Negative (LLM) · Random HN (BM25 baseline) · Conflict (LLM) · Boundary (rule split) · Noise (distractors).

Behavioral criteria (failure-specific): HN outrank / Top-5 cross / ΔMRR≥0.10; Conflict in Top-5 + effect; Noise Top-5 + Δrank≥3 or cross; Boundary split relevance. Missing excluded from BER.

LLM judge: `openai.gpt-oss-120b-1:0`, temp 0; **fail-closed** on truncated/unparsable/unavailable. Prompts: `src/facm/validation/semantic.py`, `src/facm/mutations/operators.py`.

---

## Full reproduction (optional; not required for review)

```bash
cp .env.example .env   # FACM_INFERENCE_* — see example_inference.md
python scripts/ingest_ragfailbench.py --ragfailbench-root /path/to/RAGFailBench \
  --out data/runs/main/0_base --full-corpus \
  --seeds-runs pilot_stability_s42,pilot_stability_s123,pilot_stability_s2026,pilot_v1
facm -c configs/main.yaml run
```

Pilot / smoke: `configs/pilot.yaml`, `configs/smoke.yaml`.  
Embeddings cache: `{output_dir}/_embed_cache/`.

---

## Out of scope

Reranker recovery · generator EM/F1 / faithfulness · cost / latency · ColBERT / 2nd dataset / e2e RAG (Limitations / future work).
