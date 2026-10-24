from facm.schemas.models import BehaviorValidation, ChunkPayload, GeneratorInfo, Mutation
from facm.validation.behavioral import (
    aggregate_acceptance,
    evaluate_failure_criteria,
    validate_behavior_for_retriever,
)


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


def test_delta_mrr_threshold_default_matches_frozen():
    """ΔMRR in [0.05, 0.10) fails default 0.10 but passes lenient 0.05."""
    m = _mut(failure="hard_negative", add=["hn1"])
    # clean MRR=1.0 (g1@1); mut MRR≈0.93 (g1@2? wait need ΔMRR≈0.07)
    # clean: g1@1 → MRR=1; mut: x@1, g1@2 → MRR=0.5 → Δ=0.5 too big
    # Need ΔMRR ≈ 0.07: clean g1@2 (MRR=0.5), mut g1@3 (MRR≈0.333) → Δ≈0.167
    # Better: clean g1@1 MRR=1; mut g1@1 still but with hn deep — ΔMRR=0
    # Use ranks where MRR drops by ~0.07: clean MRR=1 (rank1), mut MRR=1/1.07≈0.93
    # Practical: clean ["g1",...]; mut put g1 at rank such that 1 - 1/r ≈ 0.07 → 1/r=0.93 → r≈1.075 impossible
    # 1 - 1/2 = 0.5. For fractional: use clean g1@1 (1.0), mut g1 still@1 → 0.
    # Use conflict path? Simpler: HN with only ΔMRR criterion — hn not outranking, no top5 cross.
    # clean: g1 at rank 1; mut: filler keeps g1@1, hn at rank 10 → no effect, ΔMRR=0
    # For ΔMRR≈0.083: clean MRR=0.5 (g1@2), mut MRR≈0.417 (g1@3? 1/3≈0.333, Δ≈0.167)
    # Δ = 1/2 - 1/3 = 1/6 ≈ 0.1667 → passes both 0.10 and 0.05
    # Δ = 1/3 - 1/4 = 0.0833 → fails 0.10, passes 0.05
    clean = ["a", "b", "g1", "c", "d"]  # g1@3 MRR=1/3
    # mut: g1@4, hn far — no outrank (hn worse than gold), no top5 cross (g1 still in top5)
    mut = ["a", "b", "c", "g1", "d", "e", "f", "g", "h", "hn1"]  # g1@4 MRR=0.25; Δ≈0.083
    hits_default, _, _, _, _ = evaluate_failure_criteria(
        m, clean_ranked=clean, mutated_ranked=mut, delta_mrr_threshold=0.10
    )
    hits_lenient, _, _, _, _ = evaluate_failure_criteria(
        m, clean_ranked=clean, mutated_ranked=mut, delta_mrr_threshold=0.05
    )
    hits_strict, _, _, _, _ = evaluate_failure_criteria(
        m, clean_ranked=clean, mutated_ranked=mut, delta_mrr_threshold=0.15
    )
    assert hits_default["success"] is False
    assert hits_lenient["success"] is True
    assert hits_strict["success"] is False
    assert hits_default["delta_mrr_ge_0_10"] is False
    assert hits_lenient["delta_mrr_ge_threshold"] is True


def test_majority_consensus_stricter_than_two_of_three():
    m = _mut(failure="hard_negative", add=["hn1"])
    # One pass (outranks), two fail → majority 2 rejects; majority 3 rejects
    pass_row = validate_behavior_for_retriever(
        m,
        retriever_name="bm25",
        clean_ranked=["g1", "x", "y"],
        mutated_ranked=["hn1", "g1", "x"],
    )
    fail_a = validate_behavior_for_retriever(
        m,
        retriever_name="bge_base",
        clean_ranked=["g1", "x", "y"],
        mutated_ranked=["g1", "x", "y", "z", "w", "hn1"],
    )
    fail_b = validate_behavior_for_retriever(
        m,
        retriever_name="hybrid",
        clean_ranked=["g1", "x", "y"],
        mutated_ranked=["g1", "x", "y", "z", "w", "hn1"],
    )
    ok2, _, _ = aggregate_acceptance([pass_row, fail_a, fail_b], majority=2)
    ok3, _, _ = aggregate_acceptance([pass_row, fail_a, fail_b], majority=3)
    assert ok2 is False
    assert ok3 is False

    # Two pass, one fail → majority 2 accepts; majority 3 rejects
    pass_b = validate_behavior_for_retriever(
        m,
        retriever_name="bge_base",
        clean_ranked=["g1", "x", "y"],
        mutated_ranked=["hn1", "g1", "x"],
    )
    rows = [pass_row, pass_b, fail_b]
    # refresh accepted flags via new aggregate calls
    rows2 = [
        BehaviorValidation(**r.model_dump()) for r in rows
    ]
    rows3 = [
        BehaviorValidation(**r.model_dump()) for r in rows
    ]
    ok2b, _, _ = aggregate_acceptance(rows2, majority=2)
    ok3b, _, _ = aggregate_acceptance(rows3, majority=3)
    assert ok2b is True
    assert ok3b is False
