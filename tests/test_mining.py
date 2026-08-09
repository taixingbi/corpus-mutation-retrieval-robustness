from facm.mutations.mining import DistractorMiner
from facm.retrieval.corpus import Corpus
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


def test_distractor_miner_excludes_gold_and_answer():
    chunks = [
        _chunk("gold", "Paris is the capital of France"),
        _chunk("d1", "France has many rivers and mountains"),
        _chunk("d2", "Berlin is a large European city"),
        _chunk("leak", "The capital is Paris according to guides"),
    ]
    corpus = Corpus(chunks)
    miner = DistractorMiner.build(corpus)
    seed = CleanSeed(
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
    out = miner.top_non_gold(seed, n=5)
    ids = {c.chunk_id for c in out}
    assert "gold" not in ids
    assert "leak" not in ids
    assert "d1" in ids or "d2" in ids
