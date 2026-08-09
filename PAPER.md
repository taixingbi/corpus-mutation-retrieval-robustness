# Paper 4 — Locked Scope

**Title:** Behaviorally Validated Corpus Mutation for Retrieval Robustness Evaluation

**Core question:** Can controlled corpus mutations create valid, behaviorally effective, and transferable retrieval failures?

**Story:** Generate → Validate → Transfer

**Finding:** Behavioral validation—not semantic validation alone—is the key step for constructing robust retrieval evaluation benchmarks.

Reranker recovery, generator EM/F1, cost, and latency are **out of scope**.

## Research questions

| RQ | Question | Main metrics |
|----|----------|--------------|
| **RQ1 Validity** | Semantically valid mutations? | SVR + human audit 30–50/failure |
| **RQ2 Effectiveness** | Do they induce retrieval failures? | BER, Recall@5/20, MRR, Gold Rank Δ, Top-k crossing |
| **RQ3 Transferability** | Transfer to unseen retrievers? | Held-out ΔMRR / ΔRecall@5 / Challenge Rate |

**Ablation (RQ3):** Raw → +Semantic → +Semantic+Behavioral on held-out challenge rate (bootstrap 95% CI).

**Behavioral validation (locked):** failure-specific meaningful effects (HN outrank / Top-5 cross / ΔMRR≥0.10; Conflict in Top-5 + effect; Noise Top-5 + Δrank≥3 or Top-5 cross; Boundary uses split relevance). Missing is a **deterministic negative control** (excluded from BER).

**Baseline (RQ3):** Random HN (BM25-related corpus chunk, no LLM) vs LLM HN under the same filters.

## Dataset scale

| Run | Clean seeds | Notes |
|-----|-------------|-------|
| `pilot` | 200 | Pipeline / signal check (done) |
| `main` | ~531 | Merge unique RAGFailBench stability seeds (current max) |
| Future | 1000 | Requires generating more clean seeds in RAGFailBench (Paper 2) |

Each seed yields **one mutation per failure** (never inject five failures into one corpus):

```
(seed_i, failure_f) → one mutated corpus state
```

Failures: `missing` | `hard_negative` | `conflict` | `boundary` | `noise`

## Pipeline

```
Base Corpus + Clean Seeds
        ↓
 Corpus Mutation (one failure at a time; + Random HN baseline)
        ↓
 Semantic Validation          → RQ1
        ↓
 Behavioral Validation        → RQ2  (construction: BM25, BGE, Hybrid only)
        ↓
 Accepted Mutations
        ↓
 Held-out Retrievers (E5, GTE) → RQ3  (never used for acceptance)
        ↓
 Transferability + ablation + Random vs LLM HN
```

## Operators (locked)

| Failure | Change? | Note |
|---------|---------|------|
| Missing | No | Rule; strong control |
| Hard Negative | Prompt only | Same entity/relation, wrong plausible fact, lexical overlap |
| Conflict | No | Transfer OK |
| Boundary | No | Transfer OK |
| Noise | No | Semantically easy, behaviorally hard — **paper finding** |

## Contributions

1. **Controlled corpus mutation** — five retrieval-level failure operators  
2. **Behavioral validation** — semantic validity ≠ retrieval effectiveness  
3. **Cross-retriever transfer** — transferable vs retriever-specific failures  

## Explicitly out of scope

- Reranker comparison / recovery  
- LLM generation EM/F1, faithfulness, citation accuracy  
- Cost / serving latency  
