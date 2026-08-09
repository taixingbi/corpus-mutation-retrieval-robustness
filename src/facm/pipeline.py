from __future__ import annotations

from pathlib import Path

from facm.clients.chat import ChatClient
from facm.concurrent import map_concurrent
from facm.config import AppConfig
from facm.io import load_models, read_jsonl, write_jsonl
from facm.metrics.ranking import compute_ranking_metrics
from facm.mutations.mining import DistractorMiner
from facm.mutations.operators import generate_one_mutation, generate_random_hard_negative
from facm.reporting.report import build_rq_report, write_report
from facm.retrieval.corpus import Corpus, load_seeds
from facm.retrieval.retrievers import Retriever, apply_mutation_retrievers, build_retrievers
from facm.schemas.models import (
    BehaviorValidation,
    CleanSeed,
    Mutation,
    RetrievalResult,
)
from facm.validation.behavioral import aggregate_acceptance, validate_behavior_for_retriever
from facm.validation.semantic import validate_semantic


def _role(name: str, cfg: AppConfig) -> str:
    if name in cfg.held_out_retrievers:
        return "held_out"
    return "construction"


def _search_ids(retriever: Retriever, query: str, top_k: int) -> tuple[list[str], list[float]]:
    hits = retriever.search(query, top_k=top_k)
    return [c for c, _ in hits], [s for _, s in hits]


def _make_result(
    *,
    sample_id: str,
    retriever: str,
    role: str,
    ranked_ids: list[str],
    scores: list[float],
    relevant_ids: list[str],
    stage: str,
    mutation_id: str | None = None,
    failure: str | None = None,
) -> RetrievalResult:
    metrics = compute_ranking_metrics(ranked_ids, relevant_ids)
    return RetrievalResult(
        mutation_id=mutation_id,
        sample_id=sample_id,
        failure=failure,
        stage=stage,
        retriever=retriever,
        retriever_role=role,  # type: ignore[arg-type]
        ranked_ids=ranked_ids,
        scores=scores,
        relevant_ids=relevant_ids,
        **metrics,
    )


def embed_cache_dir(cfg: AppConfig) -> Path:
    return cfg.output_path / "_embed_cache"


def build_base_retrievers(
    cfg: AppConfig,
    corpus: Corpus | None = None,
) -> tuple[Corpus, dict[str, Retriever]]:
    corpus = corpus or Corpus.from_jsonl(cfg.base_path / "chunks.jsonl")
    retrievers = build_retrievers(
        corpus.all(),
        cfg.dense_models,
        cfg.rrf_k,
        embed_concurrency=cfg.embed_concurrency,
        embed_batch_size=cfg.embed_batch_size,
        embed_device=cfg.embed_device,
        cache_dir=embed_cache_dir(cfg),
    )
    return corpus, retrievers


def stage0_clean_baseline(
    cfg: AppConfig,
    *,
    corpus: Corpus | None = None,
    retrievers: dict[str, Retriever] | None = None,
) -> list[RetrievalResult]:
    if retrievers is None:
        print("[clean] loading/encoding base retrievers…")
        _, retrievers = build_base_retrievers(cfg, corpus)
    seeds = load_seeds(cfg.base_path / "clean_seeds.jsonl", cfg.seed_limit)
    names = list(dict.fromkeys(cfg.construction_retrievers + cfg.held_out_retrievers))

    def _one(seed: CleanSeed) -> list[RetrievalResult]:
        relevant = [seed.gold_chunk_id]
        rows: list[RetrievalResult] = []
        for name in names:
            ids, scores = _search_ids(retrievers[name], seed.question, cfg.top_k)
            rows.append(
                _make_result(
                    sample_id=seed.sample_id,
                    retriever=name,
                    role=_role(name, cfg),
                    ranked_ids=ids,
                    scores=scores,
                    relevant_ids=relevant,
                    stage="clean",
                )
            )
        return rows

    nested = map_concurrent(
        seeds, _one, max_concurrency=cfg.llm_concurrency, desc="stage0-clean"
    )
    out = [r for rows in nested for r in rows]
    write_jsonl(cfg.output_path / "1_clean" / "retrieval_result.jsonl", out)
    return out


def stage1_mutate(cfg: AppConfig, *, require_llm: bool = True) -> list[Mutation]:
    corpus = Corpus.from_jsonl(cfg.base_path / "chunks.jsonl")
    seeds = load_seeds(cfg.base_path / "clean_seeds.jsonl", cfg.seed_limit)
    chat = None
    if require_llm:
        try:
            chat = ChatClient(model=cfg.chat.model, retries=cfg.http_retries)
        except RuntimeError as exc:
            print(f"[mutate] LLM unavailable ({exc}); rule/corpus operators only")

    print("[mutate] building BM25 distractor miner for HN/noise…")
    miner = DistractorMiner.build(corpus)

    jobs: list[tuple[CleanSeed, str, int]] = []
    for seed in seeds:
        for failure in cfg.failures:
            n_try = max(1, int(cfg.mutation_oversample.get(failure, 1)))
            for attempt in range(1, n_try + 1):
                jobs.append((seed, failure, attempt))
    print(f"[mutate] {len(seeds)} seeds × oversample → {len(jobs)} generation jobs")

    def _job(item: tuple[CleanSeed, str, int]) -> Mutation | None:
        seed, failure, attempt = item
        return generate_one_mutation(
            seed,
            corpus,
            failure,
            chat=chat,
            model=cfg.chat.model,
            temperature=cfg.chat.temperature,
            max_tokens=cfg.chat.max_tokens,
            noise_add_count=cfg.noise_add_count,
            attempt=attempt,
            miner=miner,
        )

    results = map_concurrent(
        jobs, _job, max_concurrency=cfg.llm_concurrency, desc="stage1-mutate"
    )
    mutations = [m for m in results if m is not None]

    if cfg.include_random_hn_baseline:
        print(f"[mutate] random HN baseline for {len(seeds)} seeds…")

        def _rand(seed: CleanSeed) -> Mutation | None:
            return generate_random_hard_negative(seed, corpus, miner=miner, attempt=1)

        rand_rows = map_concurrent(
            seeds, _rand, max_concurrency=cfg.llm_concurrency, desc="stage1-random-hn"
        )
        mutations.extend(m for m in rand_rows if m is not None)

    write_jsonl(cfg.output_path / "2_mutations" / "mutation.jsonl", mutations)
    return mutations


def stage2_semantic(cfg: AppConfig, *, use_llm: bool = True) -> list[Mutation]:
    path = cfg.output_path / "2_mutations" / "mutation.jsonl"
    mutations = load_models(path, Mutation)
    chat = None
    if use_llm:
        try:
            chat = ChatClient(model=cfg.chat.model, retries=cfg.http_retries)
        except RuntimeError as exc:
            print(f"[semantic] LLM unavailable ({exc}); rules only")
            use_llm = False

    def _one(m: Mutation) -> Mutation:
        return validate_semantic(
            m,
            chat=chat,
            model=cfg.chat.model,
            use_llm=use_llm,
            judge_max_tokens=cfg.chat.judge_max_tokens,
        )

    out = map_concurrent(
        mutations, _one, max_concurrency=cfg.llm_concurrency, desc="stage2-semantic"
    )
    write_jsonl(cfg.output_path / "3_semantic" / "mutation.jsonl", out)
    passed = [m for m in out if m.semantic.passed]
    write_jsonl(cfg.output_path / "3_semantic" / "mutation_passed.jsonl", passed)
    return out


def stage3_behavioral(
    cfg: AppConfig,
    *,
    corpus: Corpus | None = None,
    retrievers: dict[str, Retriever] | None = None,
) -> list[BehaviorValidation]:
    if retrievers is None:
        print("[behavior] loading/encoding base retrievers…")
        corpus, clean_retrievers = build_base_retrievers(cfg, corpus)
    else:
        corpus = corpus or Corpus.from_jsonl(cfg.base_path / "chunks.jsonl")
        clean_retrievers = retrievers
    seeds = {s.sample_id: s for s in load_seeds(cfg.base_path / "clean_seeds.jsonl")}
    mutations = load_models(cfg.output_path / "3_semantic" / "mutation_passed.jsonl", Mutation)

    def _one(mutation: Mutation) -> tuple[list[BehaviorValidation], Mutation | None, dict]:
        seed = seeds.get(mutation.sample_id)
        if seed is None:
            return [], None, {}
        _, mut_retrievers = apply_mutation_retrievers(
            clean_retrievers, corpus, mutation, rrf_k=cfg.rrf_k
        )
        per_ret: list[BehaviorValidation] = []
        for name in cfg.construction_retrievers:
            clean_ids, _ = _search_ids(clean_retrievers[name], seed.question, cfg.top_k)
            mut_ids, _ = _search_ids(mut_retrievers[name], seed.question, cfg.top_k)
            per_ret.append(
                validate_behavior_for_retriever(
                    mutation,
                    retriever_name=name,
                    clean_ranked=clean_ids,
                    mutated_ranked=mut_ids,
                    top_k=cfg.top_k,
                )
            )
        ok, _, reason = aggregate_acceptance(per_ret, majority=cfg.behavioral_majority)
        pass_map = {r.retriever: bool(r.accepted_for_retriever) for r in per_ret}
        summary = {
            "sample_id": mutation.sample_id,
            "mutation_id": mutation.mutation_id,
            "failure": mutation.failure,
            "bm25_pass": pass_map.get("bm25", False),
            "dense_pass": pass_map.get("bge_base", False),
            "hybrid_pass": pass_map.get("hybrid", False),
            "accepted": ok,
            "reason": reason,
            "baseline": (mutation.metadata or {}).get("baseline"),
        }
        return per_ret, mutation if ok else None, summary

    results = map_concurrent(
        mutations, _one, max_concurrency=cfg.llm_concurrency, desc="stage3-behavior"
    )
    rows: list[BehaviorValidation] = []
    accepted_mutations: list[Mutation] = []
    summaries: list[dict] = []
    for per_ret, accepted, summary in results:
        rows.extend(per_ret)
        if summary:
            summaries.append(summary)
        if accepted is not None:
            accepted_mutations.append(accepted)

    out_dir = cfg.output_path / "4_behavior"
    write_jsonl(out_dir / "behavior_validation.jsonl", rows)
    write_jsonl(out_dir / "behavior_summary.jsonl", summaries)
    write_jsonl(out_dir / "mutation_accepted.jsonl", accepted_mutations)
    return rows


def stage4_benchmark(
    cfg: AppConfig,
    *,
    corpus: Corpus | None = None,
    retrievers: dict[str, Retriever] | None = None,
) -> None:
    if retrievers is None:
        print("[benchmark] loading/encoding base retrievers…")
        corpus, base_retrievers = build_base_retrievers(cfg, corpus)
    else:
        corpus = corpus or Corpus.from_jsonl(cfg.base_path / "chunks.jsonl")
        base_retrievers = retrievers
    seeds = {s.sample_id: s for s in load_seeds(cfg.base_path / "clean_seeds.jsonl")}
    clean_results = load_models(
        cfg.output_path / "1_clean" / "retrieval_result.jsonl", RetrievalResult
    )
    raw = load_models(cfg.output_path / "2_mutations" / "mutation.jsonl", Mutation)
    sem = load_models(cfg.output_path / "3_semantic" / "mutation_passed.jsonl", Mutation)
    beh_path = cfg.output_path / "4_behavior" / "mutation_accepted.jsonl"
    beh = load_models(beh_path, Mutation) if beh_path.exists() else []

    by_id = {m.mutation_id: m for m in raw}
    for m in sem + beh:
        by_id[m.mutation_id] = m

    names = list(dict.fromkeys(cfg.construction_retrievers + cfg.held_out_retrievers))
    retrieval_out: list[RetrievalResult] = list(clean_results)
    mutations = list(by_id.values())

    def _one(mutation: Mutation) -> list[RetrievalResult]:
        seed = seeds.get(mutation.sample_id)
        if seed is None:
            return []
        _, mut_retrievers = apply_mutation_retrievers(
            base_retrievers, corpus, mutation, rrf_k=cfg.rrf_k
        )
        rows: list[RetrievalResult] = []
        for name in names:
            ids, scores = _search_ids(mut_retrievers[name], seed.question, cfg.top_k)
            rows.append(
                _make_result(
                    sample_id=seed.sample_id,
                    retriever=name,
                    role=_role(name, cfg),
                    ranked_ids=ids,
                    scores=scores,
                    relevant_ids=mutation.mutated_relevant_ids,
                    stage="mutated",
                    mutation_id=mutation.mutation_id,
                    failure=mutation.failure,
                )
            )
        return rows

    nested = map_concurrent(
        mutations, _one, max_concurrency=cfg.llm_concurrency, desc="stage4-benchmark"
    )
    for rows in nested:
        retrieval_out.extend(rows)

    write_jsonl(cfg.output_path / "5_benchmark" / "retrieval_result.jsonl", retrieval_out)
    rebuild_report(cfg)


def rebuild_report(cfg: AppConfig) -> Path:
    """Recompute behavioral acceptance + RQ report from saved rankings (no LLM/retrieval)."""
    mut_path = cfg.output_path / "3_semantic" / "mutation.jsonl"
    if not mut_path.exists():
        mut_path = cfg.output_path / "2_mutations" / "mutation.jsonl"
    ret_path = cfg.output_path / "5_benchmark" / "retrieval_result.jsonl"
    if not mut_path.exists():
        raise FileNotFoundError(mut_path)
    if not ret_path.exists():
        raise FileNotFoundError(ret_path)

    mutations = load_models(mut_path, Mutation)
    retrieval = list(read_jsonl(ret_path))

    clean_idx: dict[tuple[str, str], list[str]] = {}
    mut_idx: dict[tuple[str, str], list[str]] = {}
    for r in retrieval:
        if r.get("stage") == "clean":
            clean_idx[(r["sample_id"], r["retriever"])] = list(r.get("ranked_ids") or [])
        elif r.get("stage") == "mutated" and r.get("mutation_id"):
            mut_idx[(r["mutation_id"], r["retriever"])] = list(r.get("ranked_ids") or [])

    rows: list[BehaviorValidation] = []
    summaries: list[dict] = []
    accepted_mutations: list[Mutation] = []

    for mutation in mutations:
        per: list[BehaviorValidation] = []
        for name in cfg.construction_retrievers:
            clean_ranked = clean_idx.get((mutation.sample_id, name))
            mut_ranked = mut_idx.get((mutation.mutation_id, name))
            if clean_ranked is None or mut_ranked is None:
                continue
            per.append(
                validate_behavior_for_retriever(
                    mutation,
                    retriever_name=name,
                    clean_ranked=clean_ranked,
                    mutated_ranked=mut_ranked,
                    top_k=cfg.top_k,
                )
            )
        if not per:
            continue
        ok, _, reason = aggregate_acceptance(per, majority=cfg.behavioral_majority)
        rows.extend(per)
        pass_map = {r.retriever: bool(r.accepted_for_retriever) for r in per}
        summaries.append(
            {
                "sample_id": mutation.sample_id,
                "mutation_id": mutation.mutation_id,
                "failure": mutation.failure,
                "bm25_pass": pass_map.get("bm25", False),
                "dense_pass": pass_map.get("bge_base", False),
                "hybrid_pass": pass_map.get("hybrid", False),
                "accepted": ok,
                "reason": reason,
                "baseline": (mutation.metadata or {}).get("baseline"),
            }
        )
        if ok:
            accepted_mutations.append(mutation)

    out_dir = cfg.output_path / "4_behavior"
    write_jsonl(out_dir / "behavior_validation.jsonl", rows)
    write_jsonl(out_dir / "behavior_summary.jsonl", summaries)
    write_jsonl(out_dir / "mutation_accepted.jsonl", accepted_mutations)

    n_acc = sum(1 for s in summaries if s["accepted"])
    print(f"[rebuild] {len(summaries)} mutations scored; {n_acc} accepted")

    report = build_rq_report(
        mutations=[m.model_dump(mode="json") for m in mutations],
        behavior_summary=summaries,
        retrieval=retrieval,
        n_boot=1000,
        bootstrap_seed=0,
    )
    report_path = Path("reports") / cfg.run_id / "rq_report.json"
    write_report(report_path, report)
    write_report(cfg.output_path / "5_benchmark" / "rq_report.json", report)
    print(f"[rebuild] wrote {report_path}")
    return report_path


def run_pipeline(cfg: AppConfig, *, skip_llm: bool = False) -> None:
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    print("[run] building base retrievers once (disk-cached)…")
    corpus, base_retrievers = build_base_retrievers(cfg)
    stage0_clean_baseline(cfg, corpus=corpus, retrievers=base_retrievers)
    stage1_mutate(cfg, require_llm=not skip_llm)
    stage2_semantic(cfg, use_llm=not skip_llm)
    stage3_behavioral(cfg, corpus=corpus, retrievers=base_retrievers)
    stage4_benchmark(cfg, corpus=corpus, retrievers=base_retrievers)
