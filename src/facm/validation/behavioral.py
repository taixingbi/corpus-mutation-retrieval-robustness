"""Failure-specific behavioral validation (meaningful retrieval effects)."""

from __future__ import annotations

from facm.metrics.difficulty import cutoff_difficulty, missing_false_support_difficulty
from facm.metrics.ranking import mrr, rank_of, recall_at_k
from facm.schemas.models import BehaviorValidation, Difficulty, Mutation


def _best_rank(ranks: dict[str, int | None]) -> int | None:
    vals = [r for r in ranks.values() if r is not None]
    return min(vals) if vals else None


def _mrr_delta(clean_ranked: list[str], mut_ranked: list[str], relevant: list[str]) -> float:
    return mrr(clean_ranked, relevant) - mrr(mut_ranked, relevant)


def evaluate_failure_criteria(
    mutation: Mutation,
    *,
    clean_ranked: list[str],
    mutated_ranked: list[str],
    top_k: int = 20,
) -> tuple[dict[str, bool], dict[str, int | None], dict[str, int | None], dict[str, int | None], str]:
    """Return hits, clean_rel ranks, mut_rel ranks, fail ranks, reason string."""
    gold_ids = list(mutation.original_gold_ids)
    if mutation.failure == "boundary":
        clean_rel_ids = gold_ids
        mut_rel_ids = list(mutation.mutated_relevant_ids)
    else:
        clean_rel_ids = gold_ids
        mut_rel_ids = list(mutation.mutated_relevant_ids) or gold_ids

    clean_rel = {rid: rank_of(clean_ranked, rid) for rid in clean_rel_ids}
    mut_rel = {rid: rank_of(mutated_ranked, rid) for rid in mut_rel_ids}
    fail_ranks = {cid: rank_of(mutated_ranked, cid) for cid in mutation.add}

    hits: dict[str, bool] = {}
    reason = "none"
    clean_best = _best_rank(clean_rel)
    mut_best = _best_rank(mut_rel)
    fail_best = _best_rank(fail_ranks)

    if mutation.failure == "hard_negative":
        gold_mut = {gid: rank_of(mutated_ranked, gid) for gid in gold_ids}
        gold_mut_best = _best_rank(gold_mut)
        hn_outranks = (
            fail_best is not None
            and (gold_mut_best is None or fail_best < gold_mut_best)
        )
        gold_clean_top5 = clean_best is not None and clean_best <= 5
        gold_mut_out_top5 = gold_mut_best is None or gold_mut_best > 5
        top5_cross = gold_clean_top5 and gold_mut_out_top5
        delta_mrr = _mrr_delta(clean_ranked, mutated_ranked, gold_ids)
        mrr_drop = delta_mrr >= 0.10
        hits["hn_outranks_gold"] = hn_outranks
        hits["gold_top5_crossing"] = top5_cross
        hits["delta_mrr_ge_0_10"] = mrr_drop
        hits["success"] = hn_outranks or top5_cross or mrr_drop
        if hn_outranks:
            reason = "failure_chunk_outranks_gold"
        elif top5_cross:
            reason = "gold_top5_crossing"
        elif mrr_drop:
            reason = "delta_mrr_ge_0_10"
        else:
            reason = "no_meaningful_hn_effect"

    elif mutation.failure == "conflict":
        conflict_top5 = fail_best is not None and fail_best <= 5
        gold_mut = {gid: rank_of(mutated_ranked, gid) for gid in gold_ids}
        gold_mut_best = _best_rank(gold_mut)
        conflict_outranks = (
            fail_best is not None
            and (gold_mut_best is None or fail_best < gold_mut_best)
        )
        delta_mrr = _mrr_delta(clean_ranked, mutated_ranked, gold_ids)
        mrr_drop = delta_mrr >= 0.10
        hits["conflict_enters_top5"] = conflict_top5
        hits["conflict_outranks_gold"] = conflict_outranks
        hits["delta_mrr_ge_0_10"] = mrr_drop
        hits["success"] = bool(conflict_top5 and (conflict_outranks or mrr_drop))
        if hits["success"]:
            reason = (
                "conflict_top5_outranks_gold"
                if conflict_outranks
                else "conflict_top5_delta_mrr"
            )
        elif not conflict_top5:
            reason = "conflict_not_in_top5"
        else:
            reason = "conflict_in_top5_but_weak"

    elif mutation.failure == "noise":
        noise_top5 = any(r is not None and r <= 5 for r in fail_ranks.values())
        gold_mut = {gid: rank_of(mutated_ranked, gid) for gid in gold_ids}
        gold_mut_best = _best_rank(gold_mut)
        delta_rank = None
        if clean_best is not None:
            mut_r = gold_mut_best if gold_mut_best is not None else 10_000
            delta_rank = mut_r - clean_best
        rank_drop = delta_rank is not None and delta_rank >= 3
        gold_clean_top5 = clean_best is not None and clean_best <= 5
        gold_mut_out_top5 = gold_mut_best is None or gold_mut_best > 5
        top5_cross = gold_clean_top5 and gold_mut_out_top5
        hits["noise_enters_top5"] = noise_top5
        hits["gold_rank_drop_ge_3"] = bool(rank_drop)
        hits["gold_top5_crossing"] = top5_cross
        hits["success"] = bool(noise_top5 and (rank_drop or top5_cross))
        if hits["success"]:
            reason = "noise_top5_and_gold_drop" if rank_drop else "noise_top5_and_crossing"
        elif not noise_top5:
            reason = "noise_not_in_top5"
        else:
            reason = "noise_in_top5_but_gold_stable"

    elif mutation.failure == "boundary":
        gold_clean_top5 = clean_best is not None and clean_best <= 5
        clean_r5 = recall_at_k(clean_ranked, clean_rel_ids, 5)
        mut_r5 = recall_at_k(mutated_ranked, mut_rel_ids, 5)
        mut_r20 = recall_at_k(mutated_ranked, mut_rel_ids, 20)
        clean_r20 = recall_at_k(clean_ranked, clean_rel_ids, 20)
        coverage_drop = mut_r5 < 1.0
        first_rel_out = mut_best is None or mut_best > 5
        hits["clean_gold_in_top5"] = gold_clean_top5
        hits["split_recall_at_5_lt_1"] = coverage_drop
        hits["first_relevant_outside_top5"] = first_rel_out
        hits["recall_at_20_drops"] = mut_r20 + 1e-12 < clean_r20
        hits["success"] = bool(
            gold_clean_top5
            and (coverage_drop or first_rel_out or (mut_r5 + 1e-12 < clean_r5))
        )
        if hits["success"]:
            if coverage_drop:
                reason = "split_incomplete_top5"
            elif first_rel_out:
                reason = "first_split_outside_top5"
            else:
                reason = "split_recall_drop"
        elif not gold_clean_top5:
            reason = "clean_gold_not_in_top5"
        else:
            reason = "splits_still_cover_top5"

    elif mutation.failure == "missing":
        gold_absent = all(cid not in mutated_ranked for cid in gold_ids)
        hits["gold_absent"] = gold_absent
        hits["operation_success"] = gold_absent
        hits["success"] = False
        hits["deterministic_control"] = True
        reason = "deterministic_negative_control" if gold_absent else "gold_still_present"

    else:
        hits["success"] = False
        reason = "unknown_failure"

    return hits, clean_rel, mut_rel, fail_ranks, reason


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
        strong = (
            any(r is not None and r <= 3 for r in fail_ranks.values()) if fail_ranks else False
        )
        return missing_false_support_difficulty(
            top5_nonrelevant_ratio=ratio,
            top20_has_strong_distractor=strong or len(mutated_ranked) >= 20,
        )

    if mutation.failure == "boundary":
        answer_ids = mutation.metadata.get("answer_piece_ids") or mutation.mutated_relevant_ids
        answer_best = (
            _best_rank({str(aid): mut_rel.get(str(aid)) for aid in answer_ids})
            if mut_rel
            else _best_rank(mut_rel)
        )
        return cutoff_difficulty(
            clean_best_rank=_best_rank(clean_rel),
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
    hits, clean_rel, mut_rel, fail_ranks, reason = evaluate_failure_criteria(
        mutation,
        clean_ranked=clean_ranked,
        mutated_ranked=mutated_ranked,
        top_k=top_k,
    )
    if mutation.failure == "missing":
        difficulty: Difficulty | None = None
    else:
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
        metadata={"reason": reason},
    )


def _majority_difficulty(
    rows: list[BehaviorValidation],
) -> Difficulty | None:
    pool = [r for r in rows if r.accepted_for_retriever and r.difficulty] or [
        r for r in rows if r.difficulty
    ]
    if not pool:
        return None
    order = {"easy": 0, "medium": 1, "hard": 2}
    return max((r.difficulty for r in pool if r.difficulty), key=lambda d: order[d])


def aggregate_acceptance(
    rows: list[BehaviorValidation],
    majority: int = 2,
) -> tuple[bool, Difficulty | None, str]:
    """Majority vote; missing never accepted for BER."""
    if not rows:
        return False, None, "no_rows"
    if rows[0].failure == "missing":
        for r in rows:
            r.accepted = False
        return False, None, "deterministic_negative_control"

    successes = sum(1 for r in rows if r.accepted_for_retriever)
    accepted = successes >= majority
    difficulty = _majority_difficulty(rows)
    reasons = [
        str((r.metadata or {}).get("reason") or "")
        for r in rows
        if r.accepted_for_retriever
    ]
    if accepted and reasons:
        reason = max(set(reasons), key=reasons.count)
    else:
        reason = "majority_fail"
    for r in rows:
        r.accepted = accepted
        r.difficulty = difficulty
        r.metadata = {**(r.metadata or {}), "aggregate_reason": reason}
    return accepted, difficulty, reason
