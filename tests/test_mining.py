from facm.mutations.mining import DistractorMiner
from facm.mutations.operators import generate_random_hard_negative
from facm.retrieval.corpus import Corpus
from facm.reporting.report import build_rq_report, is_random_hn
from facm.schemas.models import Chunk, CleanSeed, SourceRef


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        page_id=1,
        revision_id=1,
        page_title="T",
        section_title="Lead",
        text=text,
        token_count=len(text.split()),
        char_end=len(text),
    )


def _seed() -> CleanSeed:
    return CleanSeed(
        sample_id="s1",
        question="What is the capital of France?",
        gold_answer="Paris",
        source=SourceRef(
            page_id=1,
            revision_id=1,
            page_title="T",
            section_title="Lead",
            chunk_id="gold",
        ),
    )


def test_distractor_miner_excludes_gold_and_answer():
    chunks = [
        _chunk("gold", "Paris is the capital of France"),
        _chunk("d1", "France has many rivers and mountains"),
        _chunk("d2", "Berlin is a large European city"),
        _chunk("leak", "The capital is Paris according to guides"),
    ]
    corpus = Corpus(chunks)
    miner = DistractorMiner.build(corpus)
    out = miner.top_non_gold(_seed(), n=5)
    ids = {c.chunk_id for c in out}
    assert "gold" not in ids
    assert "leak" not in ids
    assert "d1" in ids or "d2" in ids


def test_generate_random_hard_negative():
    chunks = [
        _chunk("gold", "Paris is the capital of France"),
        _chunk("d1", "France has many rivers and mountains"),
    ]
    corpus = Corpus(chunks)
    miner = DistractorMiner.build(corpus)
    mut = generate_random_hard_negative(_seed(), corpus, miner=miner)
    assert mut is not None
    assert mut.failure == "hard_negative"
    assert mut.metadata.get("baseline") == "random"
    assert mut.generator.prompt_id == "hard_negative_random"
    assert is_random_hn(mut.model_dump(mode="json"))


def test_report_excludes_random_from_primary_rq1():
    mutations = [
        {
            "mutation_id": "s1_hard_negative_01",
            "failure": "hard_negative",
            "semantic": {"passed": True},
            "metadata": {"baseline": "llm"},
            "generator": {"prompt_id": "hard_negative"},
        },
        {
            "mutation_id": "s1_hard_negative_random_01",
            "failure": "hard_negative",
            "semantic": {"passed": True},
            "metadata": {"baseline": "random"},
            "generator": {"prompt_id": "hard_negative_random"},
        },
    ]
    report = build_rq_report(mutations=mutations, behavior=[], retrieval=[])
    assert report["rq1_semantic_validity"]["hard_negative"]["generated"] == 1
    assert "rq3_random_baseline" in report
    assert report["rq3_random_baseline"]["random_hard_negative"]["raw_mutations"][
        "n_mutations"
    ] == 1
