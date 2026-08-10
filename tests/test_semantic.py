from facm.schemas.models import ChunkPayload, GeneratorInfo, Mutation
from facm.validation.semantic import rule_checks, validate_semantic


def _mut(failure: str, **kwargs) -> Mutation:
    base = dict(
        mutation_id=f"t_{failure}",
        sample_id="s1",
        failure=failure,
        original_gold_ids=["g1"],
        mutated_relevant_ids=["g1"] if failure != "missing" else [],
        question="When did X die?",
        gold_answer="6 January 1990",
        generator=GeneratorInfo(model="test"),
    )
    base.update(kwargs)
    return Mutation(**base)


def test_missing_rules_pass():
    m = _mut("missing", remove=["g1"], mutated_relevant_ids=[])
    checks = rule_checks(m)
    assert checks["gold_removed"] and checks["no_relevant_ids"]
    out = validate_semantic(m, chat=None, use_llm=False)
    assert out.semantic.passed is True


def test_boundary_rules_only_no_llm():
    m = _mut(
        "boundary",
        remove=["g1"],
        add=["g1_a", "g1_b"],
        mutated_relevant_ids=["g1_a", "g1_b"],
        chunk_payloads={
            "g1_a": ChunkPayload(text="Pavel died on 6 January"),
            "g1_b": ChunkPayload(text="1990 in Moscow."),
        },
        metadata={"split_pieces": ["Pavel died on 6 January", "1990 in Moscow."]},
    )
    out = validate_semantic(m, chat=None, use_llm=True)
    assert out.semantic.passed is True
    assert out.semantic.checks["rules_passed"] is True
    assert out.semantic.checks["answer_not_complete_in_either_piece"] is True


def test_boundary_rejects_full_answer_in_one_piece():
    m = _mut(
        "boundary",
        remove=["g1"],
        add=["g1_a", "g1_b"],
        mutated_relevant_ids=["g1_a", "g1_b"],
        chunk_payloads={
            "g1_a": ChunkPayload(text="Pavel died on"),
            "g1_b": ChunkPayload(text="6 January 1990 in Moscow."),
        },
        metadata={"split_pieces": ["Pavel died on", "6 January 1990 in Moscow."]},
    )
    out = validate_semantic(m, chat=None, use_llm=False)
    assert out.semantic.checks["answer_not_complete_in_either_piece"] is False
    assert out.semantic.passed is False


def test_noise_rules_pass_without_llm_relevant():
    m = _mut(
        "noise",
        add=["n1"],
        chunk_payloads={"n1": ChunkPayload(text="A related but non-answer paragraph.")},
    )
    out = validate_semantic(m, chat=None, use_llm=True)
    assert out.semantic.passed is True


def test_hn_requires_llm_when_available_but_rules_block_leak():
    m = _mut(
        "hard_negative",
        add=["hn1"],
        chunk_payloads={
            "hn1": ChunkPayload(text="He died on 6 January 1990 after a long career.")
        },
    )
    checks = rule_checks(m)
    assert checks["answer_not_in_hn"] is False
    out = validate_semantic(m, chat=None, use_llm=False)
    assert out.semantic.passed is False
