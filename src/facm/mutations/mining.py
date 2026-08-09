from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from facm.retrieval.corpus import Corpus
from facm.schemas.models import Chunk, CleanSeed


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


@dataclass
class DistractorMiner:
    """Mine query-competitive non-gold chunks for HN/noise (cross-retriever transfer)."""

    corpus: Corpus
    _ids: list[str]
    _bm25: BM25Okapi

    @classmethod
    def build(cls, corpus: Corpus) -> DistractorMiner:
        chunks = corpus.all()
        ids = [c.chunk_id for c in chunks]
        tokenized = [_tokenize(c.text) for c in chunks]
        return cls(corpus=corpus, _ids=ids, _bm25=BM25Okapi(tokenized))

    def top_non_gold(
        self,
        seed: CleanSeed,
        *,
        n: int = 10,
        exclude: set[str] | None = None,
    ) -> list[Chunk]:
        """BM25-ranked corpus distractors (excludes gold / answer-bearing chunks)."""
        ban = set(exclude or ())
        ban.add(seed.gold_chunk_id)
        answer = (seed.gold_answer or "").lower().strip()
        scores = self._bm25.get_scores(_tokenize(seed.question))
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        out: list[Chunk] = []
        for i in order:
            cid = self._ids[i]
            if cid in ban:
                continue
            chunk = self.corpus.get(cid)
            if chunk is None:
                continue
            if answer and answer in chunk.text.lower():
                continue
            out.append(chunk)
            if len(out) >= n:
                break
        return out

    def format_examples(self, chunks: list[Chunk], *, max_chars: int = 280) -> str:
        lines: list[str] = []
        for i, c in enumerate(chunks, start=1):
            text = " ".join(c.text.split())
            if len(text) > max_chars:
                text = text[: max_chars - 3] + "..."
            lines.append(f"[{i}] {text}")
        return "\n".join(lines)
