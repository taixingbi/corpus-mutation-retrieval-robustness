# Failure taxonomy (offline)

Original criteria (ΔMRR≥0.10, ≥2/3 construction). Ranks limited to top-20 (consistent with the experiment).

Accepted primary mutations: **885**. Construction-only (no held-out challenge on E5 or GTE): **566** (64.0%).

## Held-out non-transfer buckets (accepted set)

| Failure | Retriever | Challenged | Never Top-20 | Top-20 gold stable | Top-20 no challenge | n |
|---|---|---|---|---|---|---|
| hard_negative | e5_large | 23.7% | 0.6% | 75.7% | 0.0% | 350 |
| hard_negative | gte_large | 23.4% | 0.3% | 76.3% | 0.0% | 350 |
| conflict | e5_large | 25.5% | 0.0% | 74.5% | 0.0% | 365 |
| conflict | gte_large | 24.1% | 0.0% | 75.9% | 0.0% | 365 |
| boundary | e5_large | 70.3% | 0.0% | 0.0% | 29.7% | 165 |
| boundary | gte_large | 61.8% | 1.2% | 0.6% | 36.4% | 165 |
| noise | e5_large | 80.0% | 0.0% | 20.0% | 0.0% | 5 |
| noise | gte_large | 80.0% | 0.0% | 20.0% | 0.0% | 5 |

## Construction-only transfer gap (mutation-level)

| Failure | Construction-only | Transfers to ≥1 held-out |
|---|---|---|
| hard_negative | 262 | 88 |
| conflict | 266 | 99 |
| boundary | 37 | 128 |
| noise | 1 | 4 |

## Clean gold already @1 (attribute on accepted × held-out)

| Failure | Retriever | Clean gold@1 rate | Non-transfer ∧ gold@1 |
|---|---|---|---|
| hard_negative | e5_large | 19.1% | 2.6% |
| hard_negative | gte_large | 16.6% | 2.0% |
| conflict | e5_large | 19.5% | 2.2% |
| conflict | gte_large | 15.6% | 2.5% |
| boundary | e5_large | 83.6% | 26.1% |
| boundary | gte_large | 67.9% | 23.0% |
| noise | e5_large | 0.0% | 0.0% |
| noise | gte_large | 20.0% | 0.0% |

## Construction reject reasons (semantic pass, not accepted)

| Failure | n_rejected | Per-retriever fail reasons (top) |
|---|---|---|
| hard_negative | 111 | no_meaningful_hn_effect=257 |
| conflict | 156 | conflict_in_top5_but_weak=298, conflict_not_in_top5=73 |
| boundary | 86 | splits_still_cover_top5=183, clean_gold_not_in_top5=44 |
| noise | 526 | noise_in_top5_but_gold_stable=1483, noise_not_in_top5=67 |

## Bucket definitions

- **challenged**: ΔMRR>0 or Recall@5 drops vs clean
- **mutation_never_top20**: No failure chunk in held-out top-20
- **enters_top20_gold_stable**: Failure chunk in top-20 (or N/A for boundary splits) but gold best-rank unchanged within top-20; not challenged
- **enters_top20_no_challenge_signal**: Not challenged, mutation present / gold moved, but ΔMRR/R@5 challenge definition not met
- **construction_only_no_heldout_challenge**: Behaviorally accepted on construction (≥2/3) but neither E5 nor GTE challenged
- **clean_gold_at1**: Overlapping attribute: clean gold already rank-1 on that held-out retriever (hard to displace)
