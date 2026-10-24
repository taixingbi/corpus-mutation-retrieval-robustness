# Challenge-definition audit (Boundary focus)

## Symbol convention (code = report)

- `delta_X = clean_X − mutated_X` (`src/facm/reporting/report.py`).
- **Positive** Δ ⇒ metric **drops** (degradation).
- **Challenge** ⇔ `delta_mrr > 0` **OR** `delta_recall@5 > 0`.

## Relevance mapping

- **Clean**: `relevant_ids = [gold_chunk_id]` (len=1).
- **Mutated**: `relevant_ids = mutation.mutated_relevant_ids`.
- **Boundary**: gold removed; two split pieces (len=2).

## Finding: Boundary high CR is R@5 **drop**, not rise

High Boundary CR was previously attributed to Recall@5 rising; data show Recall@5 drops (positive mean ΔR@5) and CR is R@5-only dominated. MRR often rises (negative mean ΔMRR on GTE).

| Failure | Retriever | CR | mean ΔMRR | mean ΔR@5 | R@5 rises | MRR rises | MRR-only | R5-only | both |
|---|---|---|---|---|---|---|---|---|---|
| boundary | e5_large | 70.3% | +0.0175 | +0.3394 | 0/165 | 13/165 | 6 | 95 | 15 |
| boundary | gte_large | 61.8% | -0.0659 | +0.2909 | 6/165 | 33/165 | 4 | 86 | 12 |
| hard_negative | e5_large | 23.7% | +0.0918 | +0.0057 | 0/350 | 0/350 | 81 | 0 | 2 |
| hard_negative | gte_large | 23.4% | +0.0815 | +0.0057 | 0/350 | 0/350 | 80 | 0 | 2 |
| conflict | e5_large | 25.5% | +0.0967 | +0.0055 | 0/365 | 0/365 | 91 | 0 | 2 |
| conflict | gte_large | 24.1% | +0.0750 | +0.0110 | 0/365 | 0/365 | 84 | 0 | 4 |
| noise | e5_large | 80.0% | +0.0733 | +0.0000 | 0/5 | 0/5 | 4 | 0 | 0 |
| noise | gte_large | 80.0% | +0.1638 | +0.0000 | 0/5 | 0/5 | 4 | 0 | 0 |

## Why Boundary R@5 drops while MRR can rise

Recall@k over two split pieces is `#pieces_in_top_k / 2`. Clean R@5 over one gold is usually ≈1 on the accepted set.
- E5 pieces-in-Top-5: `{'0': 3, '1': 108, '2': 54}` (often exactly one piece ⇒ R@5 1.0→0.5 ⇒ always challenged via R@5).
- GTE pieces-in-Top-5: `{'0': 9, '1': 100, '2': 56}`.
- MRR only needs the first relevant hit, so a single well-ranked piece can keep/improve MRR while Recall@5 drops.

## Consistency verdict

1. Code, `rq_report.json`, and PAPER challenge **definition** agree.
2. No sign-flip bug in Δ fields.
3. Boundary relevance remapping (1 gold → 2 pieces) explains high CR via incomplete split coverage.
4. Apparent conflict = conflating **MRR rise** with **Recall rise**.
