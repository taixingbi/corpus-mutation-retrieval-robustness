from __future__ import annotations

from facm.clients.rerank import RerankClient
from facm.metrics.ranking import (
    compute_ranking_metrics,
    recoverability_class,
    recovery_rate,
)
from facm.schemas.models import Mutation, RerankerResult, RetrievalResult


def run_rerankers(
    *,
    mutation: Mutation,
    query: str,
    hybrid_top: list[tuple[str, str]],
    # list of (chunk_id, text) — shared candidate pool for ALL rerankers
    reranker_models: list[str],
    client: RerankClient | None,
    top_n: int,
    clean_metrics: dict[str, float] | None,
    candidate_pool: str = "hybrid_top_20",
) -> list[RerankerResult]:
    """Rerank a fixed Hybrid Top-K pool with None baseline + remote models."""
    pre_ids = [cid for cid, _ in hybrid_top]
    docs = [text for _, text in hybrid_top]
    id_by_doc = {text: cid for cid, text in hybrid_top}
    relevant = mutation.mutated_relevant_ids
    pre = compute_ranking_metrics(pre_ids, relevant)
    rclass = recoverability_class(mutation.failure, pre_ids, relevant)

    results: list[RerankerResult] = []

    # None baseline = keep hybrid order, truncate to top_n for "post"
    post_none = pre_ids[:top_n]
    none_metrics = compute_ranking_metrics(post_none, relevant)
    results.append(
        _pack(
            mutation,
            "none",
            pre_ids,
            post_none,
            relevant,
            pre,
            none_metrics,
            clean_metrics,
            recoverability=rclass,
            candidate_pool=candidate_pool,
        )
    )

    if client is None:
        return results

    for model in reranker_models:
        try:
            # Always send the same full pool; top_n only truncates the returned ranking
            raw = client.rerank(
                model=model,
                query=query,
                documents=docs,
                top_n=min(top_n, len(docs)),
                return_documents=True,
            )
        except Exception as exc:
            print(f"[rerank] {model} failed: {exc}")
            continue
        post_ids: list[str] = []
        for item in raw:
            doc = item.get("document")
            if isinstance(doc, dict):
                text = doc.get("text") or doc.get("content") or ""
            else:
                text = doc if isinstance(doc, str) else ""
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(pre_ids):
                post_ids.append(pre_ids[idx])
            elif text in id_by_doc:
                post_ids.append(id_by_doc[text])
        # Keep a full permutation of the shared pool (fair compare across models)
        for cid in pre_ids:
            if cid not in post_ids:
                post_ids.append(cid)
        post_ids = post_ids[: len(pre_ids)]
        post = compute_ranking_metrics(post_ids, relevant)
        results.append(
            _pack(
                mutation,
                model,
                pre_ids,
                post_ids,
                relevant,
                pre,
                post,
                clean_metrics,
                recoverability=rclass,
                candidate_pool=candidate_pool,
            )
        )
    return results


def _pack(
    mutation: Mutation,
    reranker: str,
    pre_ids: list[str],
    post_ids: list[str],
    relevant: list[str],
    pre: dict[str, float],
    post: dict[str, float],
    clean: dict[str, float] | None,
    *,
    recoverability: str,
    candidate_pool: str,
) -> RerankerResult:
    clean = clean or {}

    def _rec(key: str) -> float | None:
        return recovery_rate(clean.get(key, 0.0), pre[key], post[key])

    return RerankerResult(
        mutation_id=mutation.mutation_id,
        sample_id=mutation.sample_id,
        failure=mutation.failure,
        reranker=reranker,
        pre_ranked_ids=pre_ids,
        post_ranked_ids=post_ids,
        relevant_ids=relevant,
        hit_at_1_pre=pre["hit_at_1"],
        hit_at_1_post=post["hit_at_1"],
        recall_at_5_pre=pre["recall_at_5"],
        recall_at_5_post=post["recall_at_5"],
        mrr_pre=pre["mrr"],
        mrr_post=post["mrr"],
        mrr_at_10_pre=pre["mrr_at_10"],
        mrr_at_10_post=post["mrr_at_10"],
        ndcg_at_5_pre=pre["ndcg_at_5"],
        ndcg_at_5_post=post["ndcg_at_5"],
        ndcg_at_10_pre=pre["ndcg_at_10"],
        ndcg_at_10_post=post["ndcg_at_10"],
        recovery_hit_at_1=_rec("hit_at_1"),
        recovery_recall_at_5=_rec("recall_at_5"),
        recovery_mrr=_rec("mrr"),
        recovery_mrr_at_10=_rec("mrr_at_10"),
        recovery_ndcg_at_5=_rec("ndcg_at_5"),
        recovery_ndcg_at_10=_rec("ndcg_at_10"),
        clean_hit_at_1=clean.get("hit_at_1"),
        clean_recall_at_5=clean.get("recall_at_5"),
        clean_mrr=clean.get("mrr"),
        clean_mrr_at_10=clean.get("mrr_at_10"),
        clean_ndcg_at_5=clean.get("ndcg_at_5"),
        clean_ndcg_at_10=clean.get("ndcg_at_10"),
        recoverability=recoverability,
        candidate_pool=candidate_pool,
    )


def lookup_clean_hybrid_metrics(
    clean_results: list[RetrievalResult], sample_id: str
) -> dict[str, float] | None:
    for r in clean_results:
        if r.sample_id == sample_id and r.retriever == "hybrid":
            return {
                "hit_at_1": r.hit_at_1,
                "recall_at_5": r.recall_at_5,
                "mrr": r.mrr,
                "mrr_at_10": r.mrr_at_10,
                "ndcg_at_5": r.ndcg_at_5,
                "ndcg_at_10": r.ndcg_at_10,
            }
    return None
