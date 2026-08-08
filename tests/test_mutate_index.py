from facm.mutations.operators import generate_missing
from facm.retrieval.corpus import Corpus
from facm.retrieval.retrievers import BM25Retriever, DenseRetriever, apply_mutation_retrievers
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


def test_dense_mutate_reuses_embeddings_shape():
    chunks = [_chunk("c1", "alpha beta gamma"), _chunk("c2", "delta epsilon"), _chunk("c3", "zeta eta")]
    dense = DenseRetriever(
        chunks,
        "sentence-transformers/all-MiniLM-L6-v2",
        name="bge_base",
        role="construction",
    )
    dense._ensure()
    assert dense._embeddings is not None
    base_dim = dense._embeddings.shape[1]

    seed = CleanSeed(
        sample_id="s1",
        question="alpha?",
        gold_answer="gamma",
        answer_type="span",
        source=SourceRef(
            page_id=1, revision_id=1, page_title="T", section_title="Lead", chunk_id="c1"
        ),
    )
    mut = generate_missing(seed)
    corpus = Corpus(chunks)
    mut_dense = dense.mutate(mut, corpus.apply_mutation(mut))
    assert "c1" not in mut_dense.chunk_ids
    assert mut_dense._embeddings is not None
    assert mut_dense._embeddings.shape == (2, base_dim)


def test_apply_mutation_retrievers_bm25_missing():
    chunks = [_chunk("c1", "gold answer lives here"), _chunk("c2", "other text about topic")]
    corpus = Corpus(chunks)
    base = {
        "bm25": BM25Retriever(chunks),
        "bge_base": DenseRetriever(
            chunks,
            "sentence-transformers/all-MiniLM-L6-v2",
            name="bge_base",
            role="construction",
        ),
    }
    # attach hybrid after dense exists
    from facm.retrieval.retrievers import HybridRetriever

    base["hybrid"] = HybridRetriever(base["bm25"], base["bge_base"])  # type: ignore[arg-type]
    seed = CleanSeed(
        sample_id="s1",
        question="gold answer?",
        gold_answer="lives",
        answer_type="span",
        source=SourceRef(
            page_id=1, revision_id=1, page_title="T", section_title="Lead", chunk_id="c1"
        ),
    )
    mut = generate_missing(seed)
    mutated, rets = apply_mutation_retrievers(base, corpus, mut)
    assert mutated.get("c1") is None
    hits = rets["bm25"].search("gold answer", top_k=5)
    assert all(cid != "c1" for cid, _ in hits)
