from facm.schemas.models import ChunkPayload, GeneratorInfo, Mutation
from facm.validation.behavioral import evaluate_failure_criteria


def _mut(**kwargs) -> Mutation:
    base = dict(
        mutation_id="m1",
        sample_id="s1",
        original_gold_ids=["g1"],
        mutated_relevant_ids=["g1"],
        question="q",
        gold_answer="1990",
        generator=GeneratorInfo(model="test"),
    )
    base.update(kwargs)
    return Mutation(**base)


def test_hn_outranks_gold():
    m = _mut(failure="hard_negative", add=["hn1"])
    hits, _, _, _, reason = evaluate_failure_criteria(
        m, clean_ranked=["g1", "x", "y"], mutated_ranked=["hn1", "g1", "x"]
    )
    assert hits["hn_outranks_gold"] is True
    assert hits["success"] is True
    assert reason == "failure_chunk_outranks_gold"


def test_hn_no_meaningful_effect():
    m = _mut(failure="hard_negative", add=["hn1"])
    hits, _, _, _, _ = evaluate_failure_criteria(
        m,
        clean_ranked=["a", "g1", "b", "c", "d"],
        mutated_ranked=["a", "g1", "b", "c", "d", "e", "f", "g", "h", "hn1"],
    )
    assert hits["success"] is False


def test_conflict_requires_top5_and_effect():
    m = _mut(failure="conflict", add=["c1"])
    hits, _, _, _, _ = evaluate_failure_criteria(
        m,
        clean_ranked=["g1", "x"],
        mutated_ranked=["g1", "x", "y", "z", "w", "c1"],
    )
    assert hits["success"] is False
    hits2, _, _, _, reason = evaluate_failure_criteria(
        m, clean_ranked=["g1", "x"], mutated_ranked=["c1", "g1", "x"]
    )
    assert hits2["success"] is True
    assert "conflict" in reason


def test_noise_requires_top5_and_meaningful_gold_drop():
    m = _mut(failure="noise", add=["n1", "n2"])
    hits, _, _, _, _ = evaluate_failure_criteria(
        m,
        clean_ranked=["g1", "a", "b", "c", "d"],
        mutated_ranked=["n1", "g1", "a", "b", "c"],
    )
    assert hits["noise_enters_top5"] is True
    assert hits["success"] is False
    hits2, _, _, _, _ = evaluate_failure_criteria(
        m,
        clean_ranked=["g1", "a", "b", "c", "d"],
        mutated_ranked=["n1", "a", "b", "c", "d", "g1"],
    )
    assert hits2["success"] is True


def test_boundary_uses_split_relevance():
    m = _mut(
        failure="boundary",
        mutated_relevant_ids=["g1_a", "g1_b"],
        add=["g1_a", "g1_b"],
        remove=["g1"],
        metadata={"answer_piece_ids": ["g1_b"]},
        chunk_payloads={
            "g1_a": ChunkPayload(text="ctx"),
            "g1_b": ChunkPayload(text="ans 1990"),
        },
    )
    hits, _, _, _, _ = evaluate_failure_criteria(
        m, clean_ranked=["g1", "x", "y"], mutated_ranked=["g1_a", "g1_b", "x"]
    )
    assert hits["success"] is False
    hits2, _, _, _, reason = evaluate_failure_criteria(
        m,
        clean_ranked=["g1", "x", "y"],
        mutated_ranked=["g1_a", "x", "y", "z", "w", "g1_b"],
    )
    assert hits2["success"] is True
    assert reason == "split_incomplete_top5"


def test_missing_never_behavioral_success():
    m = _mut(failure="missing", mutated_relevant_ids=[], add=[], remove=["g1"])
    hits, _, _, _, reason = evaluate_failure_criteria(
        m, clean_ranked=["g1", "x"], mutated_ranked=["x", "y", "z"]
    )
    assert hits["operation_success"] is True
    assert hits["success"] is False
    assert reason == "deterministic_negative_control"
