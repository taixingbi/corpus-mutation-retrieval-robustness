from facm.mutations.operators import _split_through_answer, generate_boundary
from facm.retrieval.corpus import Corpus
from facm.schemas.models import Chunk, CleanSeed, SourceRef


def test_split_masks_secondary_answer_copies():
    text = (
        "The largest of these is Jackson Lake, a natural glacial lake. "
        "Jackson Lake is popular for boating."
    )
    answer = "Jackson Lake"
    split = _split_through_answer(text, answer)
    assert split is not None
    a, b = split
    assert answer.lower() not in a.lower()
    assert answer.lower() not in b.lower()
    assert answer.lower() in (a + b).replace("…", "").lower() or answer.lower() in (
        a + b
    ).lower()


def test_generate_boundary_answer_span():
    chunk = Chunk(
        chunk_id="c1",
        text="Final Theory is a 2008 techno-thriller novel.",
        page_id=1,
        revision_id=1,
        page_title="Final Theory",
        section_title="",
    )
    corpus = Corpus([chunk])
    seed = CleanSeed(
        sample_id="s1",
        question="When was Final Theory published?",
        gold_answer="2008",
        supporting_sentence="Final Theory is a 2008 techno-thriller novel.",
        source=SourceRef(
            page_id=1,
            revision_id=1,
            page_title="Final Theory",
            section_title="",
            chunk_id="c1",
        ),
    )
    m = generate_boundary(seed, corpus)
    assert m is not None
    pieces = list(m.chunk_payloads.values())
    assert len(pieces) == 2
    assert all("2008" not in p.text for p in pieces)
    assert "2008" in "".join(p.text for p in pieces)
