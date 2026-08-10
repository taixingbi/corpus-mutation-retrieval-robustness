from facm.metrics.difficulty import cutoff_difficulty
from facm.metrics.ranking import compute_ranking_metrics


def test_mrr_and_recall():
    ranked = ["a", "b", "gold", "c"]
    m = compute_ranking_metrics(ranked, ["gold"])
    assert m["mrr"] == 1 / 3
    assert m["mrr_at_10"] == 1 / 3
    assert m["hit_at_1"] == 0.0
    assert m["hit_at_5"] == 1.0
    assert m["recall_at_5"] == 1.0
    assert m["ndcg_at_10"] > 0.0


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
