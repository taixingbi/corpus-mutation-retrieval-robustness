# Paper 4 — Locked Scope + Camera-ready Analysis Notes

**Title:** Behaviorally Validated Corpus Mutation for Retrieval Robustness Evaluation

**Core question:** Can controlled corpus mutations create valid, behaviorally effective, and transferable retrieval failures?

**Story:** Generate → Validate → Transfer

**Finding:** Behavioral validation—not semantic validation alone—is the key step for constructing robust retrieval evaluation benchmarks.

Reranker recovery, generator EM/F1, cost, and latency are **out of scope**.

Primary RQ1–RQ3 numbers are **frozen** from `reports/main/rq_report.json`. Camera-ready additions below are offline recomputations / documentation only (no re-run of LLM, embedding, or retrieval).

## Research questions

| RQ | Question | Main metrics |
|----|----------|--------------|
| **RQ1 Validity** | Semantically valid mutations? | SVR + human audit 30–50/failure |
| **RQ2 Effectiveness** | Do they induce retrieval failures? | BER, Recall@5/20, MRR, Gold Rank Δ, Top-k crossing |
| **RQ3 Transferability** | Transfer to unseen retrievers? | Held-out ΔMRR / ΔRecall@5 / Challenge Rate |

**Secondary / appendix:** nDCG@5/10/20 (clean + paired Δ + held-out Δ). Not used for acceptance or primary claims.

**Ablation (RQ3):** Raw → +Semantic → +Semantic+Behavioral on held-out challenge rate (bootstrap 95% CI).

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
| Boundary | Yes | Answer-span split: neither piece alone contains full gold |
| Noise | No | Semantically easy, behaviorally hard — **paper finding** |

## Contributions

1. **Controlled corpus mutation** — five retrieval-level failure operators  
2. **Behavioral validation** — semantic validity ≠ retrieval effectiveness  
3. **Cross-retriever transfer** — transferable vs retriever-specific failures  

## Explicitly out of scope

- Reranker comparison / recovery  
- LLM generation EM/F1, faithfulness, citation accuracy  
- Cost / serving latency  

---

## Validation criteria (Methods)

Acceptance requires **semantic pass** and **construction majority** (≥2 of BM25, BGE-base, Hybrid). Held-out retrievers (E5-large, GTE-large) are never used for acceptance. Missing is a deterministic negative control and is **excluded from BER**.

| Failure | Behavioral success (per construction retriever) |
|---------|--------------------------------------------------|
| Hard negative | HN outranks gold **OR** gold Top-5→out-of-Top-5 **OR** ΔMRR ≥ 0.10 |
| Conflict | Conflict chunk in Top-5 **AND** (outranks gold **OR** ΔMRR ≥ 0.10) |
| Noise | Noise in Top-5 **AND** (gold Δrank ≥ 3 **OR** Top-5 crossing) |
| Boundary | Clean gold in Top-5 **AND** (split R@5 &lt; 1 **OR** first relevant &gt; 5 **OR** R@5 drops) |
| Missing | Never behaviorally accepted (operation success tracked separately) |

**Δ sign convention (locked):** `ΔX = clean_X − mutated_X`. Positive Δ = degradation (drop). Reported in `rq_report.json` as `delta_mrr` / `delta_recall@5`.

Held-out **challenge** (RQ3): `ΔMRR > 0` **or** `ΔRecall@5 > 0` (same predicate in `src/facm/reporting/report.py`).

**Boundary relevance mapping:** clean metrics use the single gold chunk; mutated metrics use the **two** answer-span split pieces as `relevant_ids`. So mutated Recall@5 ∈ {0, 0.5, 1.0}. This is intentional (measures incomplete split coverage), not a sign bug.

---

## Semantic validation + human audit (Methods)

**Rules** always apply (leak / structural checks). An **LLM judge** (`openai.gpt-oss-120b-1:0`, temperature 0) is required only for Hard Negative and Conflict (`natural` ∧ `failure_intent_ok`). Missing / Boundary / Noise are rule-only. If the judge response is truncated or unparsable after retry, the gate **fails closed** (mutation rejected). Offline audit of the frozen main `3_semantic/mutation.jsonl`: judge pass 982 / reject 37 / fail-closed-trigger 0 on primary HN+Conflict (n=1019), so changing open→closed does not alter frozen RQ1–RQ3. See Appendix B for prompts.

**Human audit (RQ1, semantic only — not BER):**

| Item | Value |
|------|--------|
| Sample | 50 / failure, seed=0 (`scripts/export_semantic_audit.py`) |
| Source | `data/runs/main/3_semantic/mutation.jsonl` |
| Labels | `ok` \| `topical_fail` \| `answer_leak` \| `unnatural` \| `other` |
| Protocol | Single-rater; codebook in `reports/audit/README.md` |
| Result | 249/250 ok (99.6%); sole miss = 1 Random HN `topical_fail` |
| HN split | LLM HN 23/23 ok; Random HN 26/27 ok |

Do **not** reject audit rows because ranks dropped — that is RQ2.

---

## Analysis: Threshold sensitivity (camera-ready)

Offline recomputation from frozen rankings (`scripts/threshold_sensitivity.py`). ΔMRR threshold applies only to HN/Conflict; noise/boundary rules fixed. Full tables: `reports/main/threshold_sensitivity.md`.

| Configuration | ΔMRR | Agreement | HN BER | Conflict BER | Boundary BER | Noise BER | Pooled BER | n_acc | Held-out challenge (pooled mean) |
|---|---|---|---|---|---|---|---|---|---|
| Lenient | 0.05 | ≥2/3 | 75.9% | 70.1% | 65.7% | 0.9% | 50.2% | 885 | 32.3% |
| Original | 0.10 | ≥2/3 | 75.9% | 70.1% | 65.7% | 0.9% | 50.2% | 885 | 32.3% |
| Strict | 0.15 | ≥2/3 | 75.9% | 70.1% | 65.7% | 0.9% | 50.2% | 885 | 32.3% |
| Consensus | 0.10 | 3/3 | 69.8% | 61.0% | 37.5% | 0.6% | 41.8% | 737 | 30.2% |

**Takeaway:** Lenient/Original/Strict are identical — accepted HN/Conflict cases already fire outrank or Top-5 crossing, so ΔMRR∈{0.05,0.10,0.15} is not the binding gate. Consensus (3/3) shrinks the set and lowers BER, but held-out challenge stays in a similar band. Thresholds are therefore not fragile artifacts of the 0.10 choice. Original matches frozen `rq_report.json`.

---

## Analysis: Failure taxonomy (camera-ready)

Offline from the Original accepted set (`scripts/failure_taxonomy.py`). Ranks are top-20 only (gold outside top-20 is unknown). Full tables: `reports/main/failure_taxonomy.md`.

Among **885** primary accepted mutations, **566 (64.0%)** challenge neither E5 nor GTE (construction-only). For HN/Conflict (~24% per-retriever challenge), non-transfer is dominated by **mutation in Top-20 but gold rank stable** (~75–76%), not by mutations missing from Top-20 (&lt;1%). Boundary transfers much more often (~62–70% challenged).

**Boundary CR decomposition (audit, not a wording patch):** High Boundary CR is **not** from Recall@5 rising. On accepted held-out Boundary (n=165): mean `ΔRecall@5` is **positive** (E5 +0.34, GTE +0.29) = Recall **drops**; mean `ΔMRR` can be **negative** on GTE (−0.07) = MRR **rises**. Challenged rows are mostly **R@5-only** (E5: 95 R5-only / 6 MRR-only / 15 both). Mechanically, ~60–65% of cases retrieve exactly one split piece in Top-5 ⇒ Recall goes 1.0→0.5 and fires the R@5 branch of composite CR, while MRR (first relevant only) can stay flat or improve. Full audit: `reports/main/challenge_definition_audit.md` (`scripts/challenge_definition_audit.py`).

| Failure | Retriever | Challenged | Never Top-20 | Top-20, gold stable | n |
|---|---|---|---|---|---|
| hard_negative | E5 | 23.7% | 0.6% | 75.7% | 350 |
| hard_negative | GTE | 23.4% | 0.3% | 76.3% | 350 |
| conflict | E5 | 25.5% | 0.0% | 74.5% | 365 |
| conflict | GTE | 24.1% | 0.0% | 75.9% | 365 |
| boundary | E5 | 70.3% | 0.0% | 0.0% | 165 |
| boundary | GTE | 61.8% | 1.2% | 0.6% | 165 |

**Construction rejects** (semantic pass, not accepted): Noise is almost always `noise_in_top5_but_gold_stable`; HN `no_meaningful_hn_effect`; Conflict `conflict_in_top5_but_weak` / `conflict_not_in_top5`. Clean gold already @1 is common on boundary held-out rows but rare as the sole explanation of HN/Conflict non-transfer.

**Takeaway:** The ~24% HN/Conflict transfer rate reflects **retriever-specific ranking effects** (construction-sensitive, held-out gold stable), not failed injection into the candidate set.

---

## Limitations

- Held-out transfer is evaluated on **two dense bi-encoders** (E5-large, GTE-large). Late-interaction (e.g. ColBERT), additional sparse/neural hybrids, and rerankers are left to future work.
- Gold ranks are observed only within **top-20**; deeper demotions are not measured without re-retrieval.
- Noise remains near-zero BER by design finding (semantic ease ≠ behavioral effect).
- No second dataset, end-to-end RAG generation metrics, or multi-LLM generator sweep in this paper.

---

## Appendix A — Generation prompts (verbatim)

Model: `openai.gpt-oss-120b-1:0`, temperature 0 (see `configs/main.yaml`). Missing / Boundary / Random HN are rule-based (no LLM text generation).

### Hard negative (system)

```
You write cross-retriever hard-negatives for IR evaluation: same entity, same relation, wrong plausible fact, high lexical overlap. Return ONLY JSON: {"text": "..."}.
```

### Hard negative (user template)

```
Question: {question}
Correct gold answer (FORBIDDEN — never state or paraphrase): {gold_answer}
Gold passage style:
{style}

{optional mined distractor block}
Write ONE hard-negative Wikipedia-style chunk (~80-150 words) that:
1. Mentions the SAME primary entity/entities as the question
2. Addresses the SAME relation / question intent ...
3. States a WRONG but PLAUSIBLE fact for that relation ...
4. Is lexically similar to the question ...
5. Never includes the gold answer value or a paraphrase of it
6. Sounds like natural encyclopedic prose (no meta commentary)
```

### Conflict (system)

```
You write Wikipedia-style passages that contradict a known fact. Return ONLY JSON: {"text": "..."}.
```

### Conflict (user template)

```
Question: {question}
Correct gold answer (you must CONTRADICT this): {gold_answer}
Style reference:
{style}

Write ONE conflicting chunk (~80-150 words):
- Same entity/topic as the question
- Explicitly state a different incorrect answer to the question
- Sound like a confident Wikipedia paragraph
- Do not acknowledge that the claim is wrong
```

Source: `src/facm/mutations/operators.py`.

---

## Appendix B — LLM semantic judge (verbatim)

**Gate:** required only for `hard_negative` and `conflict` (`llm_natural` ∧ `llm_failure_intent_ok`). Fail-closed on truncated / unparsable / unavailable judge after one retry.

### System

```
You are an IR mutation validator. Decide three booleans. Be moderately permissive. Final answer must be a single compact JSON object with exactly three keys.
```

### User template

```
Failure type: {failure}
Question: {question}
Gold answer: {gold_answer}
Chunks:
{json payloads, up to 3 chunks, 400 chars each}

{FAILURE_JUDGE_INSTRUCTIONS[failure]}

Output ONE JSON object and nothing else.
Exact keys only: {"topical":true|false,"natural":true|false,"failure_intent_ok":true|false}
No notes. No markdown. No explanation.
```

### Per-failure judge instructions

**hard_negative:** A hard-negative chunk must be Wikipedia-style and topically related to the question, but must NOT contain or entail the gold answer. Set failure_intent_ok=true if it is a plausible near-miss distractor. Set natural=true if prose looks encyclopedic. Set topical=true if it shares topic with the question (even if it does not answer it).

**conflict:** A conflict chunk must be Wikipedia-style and assert an alternate/incorrect answer (or contradict the gold answer) for the question. Set failure_intent_ok=true if it clearly conflicts with the gold answer. Set natural=true if prose looks encyclopedic. Set topical=true if it addresses the same question/entity.

Source: `src/facm/validation/semantic.py`.

---

## Appendix C — Human audit codebook

See `reports/audit/README.md` and aggregate `reports/audit/semantic_audit_summary.json`.

---

## Reproduce camera-ready analyses

```bash
# Offline only — does not overwrite frozen rq_report.json / 4_behavior for Original claims
python scripts/threshold_sensitivity.py
python scripts/failure_taxonomy.py
python scripts/challenge_definition_audit.py
```

Artifacts: `reports/main/threshold_sensitivity.{json,md}`, `reports/main/failure_taxonomy.{json,md}`, `reports/main/challenge_definition_audit.{json,md}`.
