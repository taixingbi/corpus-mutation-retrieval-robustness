# Threshold sensitivity (offline)

Recomputed from saved `5_benchmark/retrieval_result.jsonl` rankings. ΔMRR threshold applies to hard_negative and conflict only; noise/boundary criteria are fixed. Missing excluded from BER.

## Configuration

| Configuration | ΔMRR threshold | Retriever agreement |
|---|---|---|
| Lenient | 0.05 | ≥2/3 |
| Original | 0.10 | ≥2/3 |
| Strict | 0.15 | ≥2/3 |
| Consensus | 0.10 | 3/3 |

## BER by failure

| Config | HN | Conflict | Boundary | Noise | Pooled | n_accepted |
|---|---|---|---|---|---|---|
| Lenient | 75.9% | 70.1% | 65.7% | 0.9% | 50.2% | 885 |
| Original | 75.9% | 70.1% | 65.7% | 0.9% | 50.2% | 885 |
| Strict | 75.9% | 70.1% | 65.7% | 0.9% | 50.2% | 885 |
| Consensus | 69.8% | 61.0% | 37.5% | 0.6% | 41.8% | 737 |

## Held-out challenge rate (accepted set)

| Config | Failure | E5 | GTE | Mean (E5+GTE) | n |
|---|---|---|---|---|---|
| Lenient | hard_negative | 23.7% [19.4%, 28.6%] | 23.4% [19.1%, 28.3%] | 23.6% [20.6%, 26.9%] | 350 |
| Lenient | conflict | 25.5% [21.1%, 29.9%] | 24.1% [19.5%, 28.8%] | 24.8% [21.6%, 27.8%] | 365 |
| Lenient | boundary | 70.3% [63.0%, 77.0%] | 61.8% [53.3%, 69.1%] | 66.1% [60.9%, 70.9%] | 165 |
| Lenient | noise | 80.0% [40.0%, 100.0%] | 80.0% [40.0%, 100.0%] | 80.0% [50.0%, 100.0%] | 5 |
| Lenient | pooled | 33.4% [30.2%, 36.4%] | 31.2% [28.0%, 34.1%] | 32.3% [30.2%, 34.5%] | 885 |
| Original | hard_negative | 23.7% [19.4%, 28.6%] | 23.4% [19.1%, 28.3%] | 23.6% [20.6%, 26.9%] | 350 |
| Original | conflict | 25.5% [21.1%, 29.9%] | 24.1% [19.5%, 28.8%] | 24.8% [21.6%, 27.8%] | 365 |
| Original | boundary | 70.3% [63.0%, 77.0%] | 61.8% [53.3%, 69.1%] | 66.1% [60.9%, 70.9%] | 165 |
| Original | noise | 80.0% [40.0%, 100.0%] | 80.0% [40.0%, 100.0%] | 80.0% [50.0%, 100.0%] | 5 |
| Original | pooled | 33.4% [30.2%, 36.4%] | 31.2% [28.0%, 34.1%] | 32.3% [30.2%, 34.5%] | 885 |
| Strict | hard_negative | 23.7% [19.4%, 28.6%] | 23.4% [19.1%, 28.3%] | 23.6% [20.6%, 26.9%] | 350 |
| Strict | conflict | 25.5% [21.1%, 29.9%] | 24.1% [19.5%, 28.8%] | 24.8% [21.6%, 27.8%] | 365 |
| Strict | boundary | 70.3% [63.0%, 77.0%] | 61.8% [53.3%, 69.1%] | 66.1% [60.9%, 70.9%] | 165 |
| Strict | noise | 80.0% [40.0%, 100.0%] | 80.0% [40.0%, 100.0%] | 80.0% [50.0%, 100.0%] | 5 |
| Strict | pooled | 33.4% [30.2%, 36.4%] | 31.2% [28.0%, 34.1%] | 32.3% [30.2%, 34.5%] | 885 |
| Consensus | hard_negative | 22.4% [17.7%, 27.3%] | 21.7% [17.4%, 26.4%] | 22.0% [18.8%, 25.3%] | 322 |
| Consensus | conflict | 21.7% [17.6%, 26.4%] | 20.8% [16.0%, 25.2%] | 21.2% [18.2%, 24.5%] | 318 |
| Consensus | boundary | 88.3% [80.9%, 94.7%] | 84.0% [75.5%, 90.4%] | 86.2% [80.9%, 91.0%] | 94 |
| Consensus | noise | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 100.0% [100.0%, 100.0%] | 3 |
| Consensus | pooled | 30.8% [27.4%, 33.9%] | 29.6% [26.2%, 32.8%] | 30.2% [27.9%, 32.5%] | 737 |

## Sanity check (Original vs frozen rq_report.json)

Original configuration **matches** frozen main RQ2/RQ3 numbers.

## Finding

Lenient / Original / Strict yield **identical** BER and held-out challenge rates: accepted HN/Conflict mutations already satisfy outrank or Top-5-crossing criteria, so the ΔMRR cutoff in {0.05, 0.10, 0.15} is not the binding constraint. Requiring **3/3 consensus** shrinks the accepted set (885→737) and lowers pooled BER (50.2%→41.8%), while held-out challenge rates stay in a similar range (pooled mean 32.3%→30.2%).

## Note

ΔMRR threshold gates hard_negative and conflict only. Noise/boundary rules fixed. Missing excluded from BER. Random HN excluded from primary metrics. Challenge = ΔMRR>0 or Recall@5 drops vs clean.
