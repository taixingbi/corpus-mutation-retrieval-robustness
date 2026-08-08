from __future__ import annotations

from facm.schemas.models import Difficulty


def cutoff_difficulty(
    *,
    clean_best_rank: int | None,
    mutated_best_rank: int | None,
    failure_outranks: bool = False,
) -> Difficulty:
    """Cutoff-aware difficulty aligned with Recall@5/20.

    Easy: stays in same band (both in Top-5, or both in 6-20 without Top-5 exit).
    Medium: crosses Top-5 but remains Top-20.
    Hard: falls outside Top-20, or failure chunk strongly outranks.
    """
    if failure_outranks and (
        mutated_best_rank is None or (clean_best_rank is not None and mutated_best_rank > clean_best_rank)
    ):
        if mutated_best_rank is None or mutated_best_rank > 20:
            return "hard"

    if clean_best_rank is None:
        # Already missing on clean — treat mutated absence as hard if still missing
        return "hard" if mutated_best_rank is None else "easy"

    if mutated_best_rank is None or mutated_best_rank > 20:
        return "hard"

    if clean_best_rank <= 5:
        if mutated_best_rank <= 5:
            return "easy"
        if mutated_best_rank <= 20:
            return "medium"
        return "hard"

    # clean was 6-20
    if mutated_best_rank <= 20:
        return "easy"
    return "hard"


def missing_false_support_difficulty(
    *,
    top5_nonrelevant_ratio: float,
    top20_has_strong_distractor: bool,
) -> Difficulty:
    if top5_nonrelevant_ratio >= 0.8 or top20_has_strong_distractor:
        return "hard"
    if top5_nonrelevant_ratio >= 0.4:
        return "medium"
    return "easy"
