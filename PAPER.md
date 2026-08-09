# Paper 4 — Locked Scope

**Title:** Behaviorally Validated Corpus Mutation for Retrieval Robustness Evaluation

**Core question:** Can controlled corpus mutations create valid, behaviorally effective, and transferable retrieval failures?

**Story:** Generate → Validate → Transfer

Reranker recovery, generator EM/F1, cost, and latency are **out of scope**.

## Dataset

Reuse RAGFailBench assets:

- `chunks.jsonl` — base corpus  
- `clean_seeds.jsonl` — query + gold evidence  

Each seed yields **one mutation per failure** (never inject five failures into one corpus):

```
(seed_i, failure_f) → one mutated corpus state
```

Failures: `missing` | `hard_negative` | `conflict` | `boundary` | `noise`  
(`evidence_position` is not used.)

## Pipeline

```
Base Corpus + Clean Seeds
        ↓
 Corpus Mutation (one failure at a time)
        ↓
 Semantic Validation          → RQ1
        ↓
 Behavioral Validation        → RQ2  (construction: BM25, BGE, Hybrid only)
        ↓
 Accepted Mutations
        ↓
 Held-out Retrievers (E5, GTE) → RQ3  (never used for acceptance)
        ↓
 Transferability + ablation
```

## Research questions

| RQ | Question | Main metrics |
|----|----------|--------------|
| **RQ1 Validity** | Semantically valid mutations? | SVR (Semantic Validity Rate) + human audit 30–50/failure |
| **RQ2 Effectiveness** | Do they induce retrieval failures? | BER, Recall@5/20, MRR, Gold Rank Δ, Top-5/20 crossing |
| **RQ3 Transferability** | Transfer to unseen retrievers? | Held-out ΔMRR / ΔRecall@5 / Challenge Rate |

**Ablation (under RQ3):** Raw → +Semantic → +Semantic+Behavioral on held-out challenge rate.

## Contributions

1. **Controlled corpus mutation** — five retrieval-level failure operators  
2. **Behavioral validation** — semantic validity ≠ retrieval effectiveness  
3. **Cross-retriever transfer** — transferable vs retriever-specific failures  

## Explicitly out of scope

- Reranker comparison / recovery  
- LLM generation EM/F1, faithfulness, citation accuracy  
- Cost / serving latency  
