from __future__ import annotations

from facm.metrics.difficulty import cutoff_difficulty, missing_false_support_difficulty
from facm.metrics.ranking import rank_of
from facm.schemas.models import BehaviorValidation, Difficulty, Mutation


def _best_rank(ranks: dict[str, int | None]) -> int | None:
    vals = [r for r in ranks.values() if r is not None]
    return min(vals) if vals else None


def _second_best_rank(ranks: dict[str, int | None]) -> int | None:
    vals = sorted(r for r in ranks.values() if r is not None)
    if len(vals) < 2:
        return None
    return vals[1]


def evaluate_failure_criteria(
    mutation: Mutation,
    *,
    clean_ranked: list[str],
    mutated_ranked: list[str],
    top_k: int = 20,
) -> tuple[dict[str, bool], dict[str, int | None], dict[str, int | None], dict[str, int | None]]:
    clean_rel = {rid: rank_of(clean_ranked, rid) for rid in mutation.original_gold_ids}
    mut_rel = {rid: rank_of(mutated_ranked, rid) for rid in mutation.mutated_relevant_ids}
    fail_ranks = {cid: rank_of(mutated_ranked, cid) for cid in mutation.add}

    hits: dict[str, bool] = {}
    clean_best = _best_rank(clean_rel)
    mut_best = _best_rank(mut_rel)

    if mutation.failure == "hard_negative":
        hn_in_topk = any(r is not None and r <= top_k for r in fail_ranks.values())
        gold_worse = clean_best is not None and (mut_best is None or mut_best > clean_best)
        hits["hn_enters_topk"] = hn_in_topk
        hits["gold_rank_worsens"] = gold_worse
        hits["success"] = hn_in_topk or gold_worse

    elif mutation.failure == "conflict":
        conflict_in = any(r is not None and r <= top_k for r in fail_ranks.values())
        hits["conflict_enters_topk"] = conflict_in
        hits["success"] = conflict_in

    elif mutation.failure == "boundary":
        # Boundary success: complete evidence is no longer available in one clean gold hit.
        gold_was_in = clean_best is not None and clean_best <= top_k
        n_rel = len(mutation.mutated_relevant_ids)
        in_top5 = sum(1 for r in mut_rel.values() if r is not None and r <= 5)
        in_top1 = sum(1 for r in mut_rel.values() if r is not None and r <= 1)
        second = _second_best_rank(mut_rel)
        first = mut_best
        answer_ids = mutation.metadata.get("answer_piece_ids") or []
        if isinstance(answer_ids, list) and answer_ids:
            answer_ranks = {aid: rank_of(mutated_ranked, str(aid)) for aid in answer_ids}
            answer_best = _best_rank(answer_ranks)
        else:
            answer_best = mut_best

        incomplete_top1 = n_rel >= 2 and in_top1 < n_rel
        incomplete_top5 = n_rel >= 2 and in_top5 < n_rel
        dispersed = (
            first is not None
            and second is not None
            and (second - first) >= 2
        )
        answer_degraded = (
            clean_best is not None
            and (answer_best is None or answer_best > clean_best)
        )
        answer_out = answer_best is None or answer_best > 5

        hits["incomplete_top1"] = incomplete_top1
        hits["incomplete_top5"] = incomplete_top5
        hits["piece_dispersion"] = dispersed
        hits["answer_piece_degraded"] = answer_degraded
        hits["answer_outside_top5"] = bool(gold_was_in and answer_out)
        # Accept if clean had full evidence in one chunk, and mutation forces multi-chunk
        # retrieval with at least one degradation/dispersion/coverage signal.
        hits["success"] = bool(
            gold_was_in
            and n_rel >= 2
            and incomplete_top1
            and (incomplete_top5 or dispersed or answer_degraded or answer_out)
        )

    elif mutation.failure == "noise":
        noise_in = any(r is not None and r <= 5 for r in fail_ranks.values())
        gold_worse = clean_best is not None and (mut_best is None or mut_best > clean_best)
        hits["noise_consumes_topk"] = noise_in
        hits["gold_rank_worsens"] = gold_worse
        hits["success"] = noise_in or gold_worse

    elif mutation.failure == "missing":
        gold_absent = all(cid not in mutated_ranked for cid in mutation.original_gold_ids)
        top5 = mutated_ranked[:5]
        hits["gold_absent"] = gold_absent
        hits["topk_filled"] = len(top5) > 0
        hits["success"] = gold_absent and len(top5) > 0
    else:
        hits["success"] = False

    return hits, clean_rel, mut_rel, fail_ranks


def assign_difficulty(
    mutation: Mutation,
    *,
    clean_rel: dict[str, int | None],
    mut_rel: dict[str, int | None],
    fail_ranks: dict[str, int | None],
    mutated_ranked: list[str],
) -> Difficulty:
    if mutation.failure == "missing":
        top5 = mutated_ranked[:5]
        ratio = 1.0 if top5 else 0.0
        strong = any(r is not None and r <= 3 for r in fail_ranks.values()) if fail_ranks else False
        return missing_false_support_difficulty(
            top5_nonrelevant_ratio=ratio,
            top20_has_strong_distractor=strong or len(mutated_ranked) >= 20,
        )

    if mutation.failure == "boundary":
        answer_ids = mutation.metadata.get("answer_piece_ids") or mutation.mutated_relevant_ids
        answer_best = _best_rank(
            {str(aid): mut_rel.get(str(aid)) for aid in answer_ids}
        ) if mut_rel else _best_rank(mut_rel)
        # prefer answer-piece rank for cutoff difficulty
        clean_best = _best_rank(clean_rel)
        return cutoff_difficulty(
            clean_best_rank=clean_best,
            mutated_best_rank=answer_best,
            failure_outranks=False,
        )

    clean_best = _best_rank(clean_rel)
    mut_best = _best_rank(mut_rel)
    fail_best = _best_rank(fail_ranks)
    outranks = (
        fail_best is not None
        and (mut_best is None or fail_best < mut_best)
        and fail_best <= 3
    )
    return cutoff_difficulty(
        clean_best_rank=clean_best,
        mutated_best_rank=mut_best,
        failure_outranks=outranks,
    )


def validate_behavior_for_retriever(
    mutation: Mutation,
    *,
    retriever_name: str,
    clean_ranked: list[str],
    mutated_ranked: list[str],
    top_k: int = 20,
) -> BehaviorValidation:
    hits, clean_rel, mut_rel, fail_ranks = evaluate_failure_criteria(
        mutation,
        clean_ranked=clean_ranked,
        mutated_ranked=mutated_ranked,
        top_k=top_k,
    )
    difficulty = assign_difficulty(
        mutation,
        clean_rel=clean_rel,
        mut_rel=mut_rel,
        fail_ranks=fail_ranks,
        mutated_ranked=mutated_ranked,
    )
    return BehaviorValidation(
        mutation_id=mutation.mutation_id,
        sample_id=mutation.sample_id,
        failure=mutation.failure,
        retriever=retriever_name,
        retriever_role="construction",
        clean_relevant_ranks=clean_rel,
        mutated_relevant_ranks=mut_rel,
        failure_chunk_ranks=fail_ranks,
        criterion_hits=hits,
        accepted_for_retriever=bool(hits.get("success")),
        difficulty=difficulty,
    )


def aggregate_acceptance(
    rows: list[BehaviorValidation],
    majority: int = 2,
) -> tuple[bool, Difficulty | None]:
    if not rows:
        return False, None
    successes = sum(1 for r in rows if r.accepted_for_retriever)
    accepted = successes >= majority
    pool = [r for r in rows if r.accepted_for_retriever and r.difficulty] or [
        r for r in rows if r.difficulty
    ]
    order = {"easy": 0, "medium": 1, "hard": 2}
    difficulty = None
    if pool:
        difficulty = max((r.difficulty for r in pool if r.difficulty), key=lambda d: order[d])
    for r in rows:
        r.accepted = accepted
        r.difficulty = difficulty
    return accepted, difficulty
