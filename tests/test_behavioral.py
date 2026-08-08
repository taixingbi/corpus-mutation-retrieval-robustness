from facm.schemas.models import ChunkPayload, GeneratorInfo, Mutation
from facm.validation.behavioral import evaluate_failure_criteria


def test_boundary_success_when_answer_piece_dispersed():
    m = Mutation(
        mutation_id="b1",
        sample_id="s1",
        failure="boundary",
        original_gold_ids=["g1"],
        mutated_relevant_ids=["g1_a", "g1_b"],
        add=["g1_a", "g1_b"],
        remove=["g1"],
        question="q",
        gold_answer="1990",
        generator=GeneratorInfo(model="rule"),
        chunk_payloads={
            "g1_a": ChunkPayload(text="context"),
            "g1_b": ChunkPayload(text="answer sentence 1990"),
        },
        metadata={"answer_piece_ids": ["g1_b"]},
    )
    clean = ["g1", "x", "y"]
    # answer piece at rank 6, context at 1 → incomplete top5 for answer + dispersion
    mutated = ["g1_a", "x", "y", "z", "w", "g1_b"]
    hits, _, _, _ = evaluate_failure_criteria(
        m, clean_ranked=clean, mutated_ranked=mutated, top_k=20
    )
    assert hits["incomplete_top1"] is True
    assert hits["success"] is True


def test_boundary_success_with_dispersion_both_in_top5():
    m = Mutation(
        mutation_id="b2",
        sample_id="s1",
        failure="boundary",
        original_gold_ids=["g1"],
        mutated_relevant_ids=["g1_a", "g1_b"],
        add=["g1_a", "g1_b"],
        remove=["g1"],
        question="q",
        gold_answer="1990",
        generator=GeneratorInfo(model="rule"),
        metadata={"answer_piece_ids": ["g1_b"]},
    )
    clean = ["g1"]
    mutated = ["g1_a", "x", "y", "g1_b"]  # ranks 1 and 4, gap=3
    hits, _, _, _ = evaluate_failure_criteria(
        m, clean_ranked=clean, mutated_ranked=mutated, top_k=20
    )
    assert hits["piece_dispersion"] is True
    assert hits["success"] is True
