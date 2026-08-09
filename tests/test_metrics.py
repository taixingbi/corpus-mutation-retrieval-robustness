from facm.metrics.difficulty import cutoff_difficulty
from facm.metrics.ranking import (
    compute_ranking_metrics,
    mrr,
    pool_recovery_rate,
    recovery_rate,
)


def test_mrr_and_recall():
    ranked = ["a", "b", "gold", "c"]
    m = compute_ranking_metrics(ranked, ["gold"])
    assert m["mrr"] == 1 / 3
    assert m["mrr_at_10"] == 1 / 3
    assert m["hit_at_1"] == 0.0
    assert m["hit_at_5"] == 1.0
    assert m["recall_at_5"] == 1.0
    assert m["ndcg_at_10"] > 0.0


def test_recovery_rate():
    assert abs(recovery_rate(1.0, 0.4, 0.7) - 0.5) < 1e-9
    assert recovery_rate(0.5, 0.5, 0.6) is None


def test_pool_recovery_rate():
    # pre=0.0 → post=1.0 closes 100% of the gap
    assert abs(pool_recovery_rate(0.0, 1.0) - 1.0) < 1e-9
    # pre=0.5 → post=0.75 closes half of remaining gap
    assert abs(pool_recovery_rate(0.5, 0.75) - 0.5) < 1e-9
    # already at ceiling → undefined
    assert pool_recovery_rate(1.0, 1.0) is None


def test_cutoff_difficulty():
    assert cutoff_difficulty(clean_best_rank=2, mutated_best_rank=4) == "easy"
    assert cutoff_difficulty(clean_best_rank=2, mutated_best_rank=9) == "medium"
    assert cutoff_difficulty(clean_best_rank=2, mutated_best_rank=None) == "hard"


def test_missing_mutation_schema():
    from facm.mutations.operators import generate_missing
    from facm.schemas.models import CleanSeed, SourceRef

    seed = CleanSeed(
        sample_id="s1",
        question="When?",
        gold_answer="1990",
        answer_type="date",
        source=SourceRef(
            page_id=1,
            revision_id=1,
            page_title="X",
            section_title="Lead",
            chunk_id="c1",
        ),
    )
    m = generate_missing(seed)
    assert m.mutated_relevant_ids == []
    assert m.remove == ["c1"]
