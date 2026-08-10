from __future__ import annotations

import math
from typing import Sequence


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return 1.0 if any(r in top for r in relevant_ids) else 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(ranked_ids[:k])
    return sum(1 for r in relevant_ids if r in top) / len(relevant_ids)


def mrr(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def mrr_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """MRR restricted to top-k (0 if first relevant is beyond k)."""
    return mrr(ranked_ids[:k], relevant_ids)


def _dcg(relevances: Sequence[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    gains = [1.0 if cid in relevant else 0.0 for cid in ranked_ids[:k]]
    ideal = sorted(gains, reverse=True)
    # Ideal should use up to min(k, |relevant|) ones
    ideal = [1.0] * min(k, len(relevant_ids)) + [0.0] * max(0, k - len(relevant_ids))
    ideal = ideal[:k]
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg(gains) / idcg


def rank_of(ranked_ids: Sequence[str], chunk_id: str) -> int | None:
    try:
        return ranked_ids.index(chunk_id) + 1
    except ValueError:
        return None


def compute_ranking_metrics(
    ranked_ids: Sequence[str], relevant_ids: Sequence[str]
) -> dict[str, float]:
    return {
        "hit_at_1": hit_at_k(ranked_ids, relevant_ids, 1),
        "hit_at_5": hit_at_k(ranked_ids, relevant_ids, 5),
        "recall_at_5": recall_at_k(ranked_ids, relevant_ids, 5),
        "recall_at_10": recall_at_k(ranked_ids, relevant_ids, 10),
        "recall_at_20": recall_at_k(ranked_ids, relevant_ids, 20),
        "mrr": mrr(ranked_ids, relevant_ids),
        "mrr_at_10": mrr_at_k(ranked_ids, relevant_ids, 10),
        "ndcg_at_5": ndcg_at_k(ranked_ids, relevant_ids, 5),
        "ndcg_at_10": ndcg_at_k(ranked_ids, relevant_ids, 10),
        "ndcg_at_20": ndcg_at_k(ranked_ids, relevant_ids, 20),
    }
