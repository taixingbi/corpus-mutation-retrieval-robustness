from __future__ import annotations

from pathlib import Path

from facm.clients.rerank import RerankClient
from facm.metrics.ranking import compute_ranking_metrics, recovery_rate
from facm.schemas.models import Mutation, RerankerResult, RetrievalResult


def run_rerankers(
    *,
    mutation: Mutation,
    query: str,
    hybrid_top: list[tuple[str, str]],
    # list of (chunk_id, text)
    reranker_models: list[str],
    client: RerankClient | None,
    top_n: int,
    clean_metrics: dict[str, float] | None,
) -> list[RerankerResult]:
    """Rerank Hybrid Top-20 with None baseline + remote models."""
    pre_ids = [cid for cid, _ in hybrid_top]
    docs = [text for _, text in hybrid_top]
    id_by_doc = {text: cid for cid, text in hybrid_top}
    relevant = mutation.mutated_relevant_ids
    pre = compute_ranking_metrics(pre_ids, relevant)

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
        )
    )

    if client is None:
        return results

    for model in reranker_models:
        try:
            raw = client.rerank(
                model=model,
                query=query,
                documents=docs,
                top_n=top_n,
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
            # Prefer index mapping when present
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(pre_ids):
                post_ids.append(pre_ids[idx])
            elif text in id_by_doc:
                post_ids.append(id_by_doc[text])
        # fill remaining from pre order if API returned fewer
        for cid in pre_ids:
            if cid not in post_ids:
                post_ids.append(cid)
            if len(post_ids) >= top_n:
                break
        post_ids = post_ids[:top_n]
        post = compute_ranking_metrics(post_ids, relevant)
        results.append(
            _pack(mutation, model, pre_ids, post_ids, relevant, pre, post, clean_metrics)
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
) -> RerankerResult:
    clean = clean or {}
    return RerankerResult(
        mutation_id=mutation.mutation_id,
        sample_id=mutation.sample_id,
        failure=mutation.failure,
        reranker=reranker,
        pre_ranked_ids=pre_ids,
        post_ranked_ids=post_ids,
        relevant_ids=relevant,
        recall_at_5_pre=pre["recall_at_5"],
        recall_at_5_post=post["recall_at_5"],
        mrr_pre=pre["mrr"],
        mrr_post=post["mrr"],
        ndcg_at_5_pre=pre["ndcg_at_5"],
        ndcg_at_5_post=post["ndcg_at_5"],
        recovery_recall_at_5=recovery_rate(
            clean.get("recall_at_5", 0.0), pre["recall_at_5"], post["recall_at_5"]
        ),
        recovery_mrr=recovery_rate(clean.get("mrr", 0.0), pre["mrr"], post["mrr"]),
        recovery_ndcg_at_5=recovery_rate(
            clean.get("ndcg_at_5", 0.0), pre["ndcg_at_5"], post["ndcg_at_5"]
        ),
        clean_recall_at_5=clean.get("recall_at_5"),
        clean_mrr=clean.get("mrr"),
        clean_ndcg_at_5=clean.get("ndcg_at_5"),
    )


def lookup_clean_hybrid_metrics(
    clean_results: list[RetrievalResult], sample_id: str
) -> dict[str, float] | None:
    for r in clean_results:
        if r.sample_id == sample_id and r.retriever == "hybrid":
            return {
                "recall_at_5": r.recall_at_5,
                "mrr": r.mrr,
                "ndcg_at_5": r.ndcg_at_5,
            }
    return None
