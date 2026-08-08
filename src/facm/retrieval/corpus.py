from __future__ import annotations

from pathlib import Path

from facm.io import load_models
from facm.schemas.models import Chunk, CleanSeed, Mutation


class Corpus:
    """In-memory chunk store with mutation apply(add/remove)."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}

    @classmethod
    def from_jsonl(cls, path: Path) -> Corpus:
        return cls(load_models(path, Chunk))

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def all(self) -> list[Chunk]:
        return list(self._by_id.values())

    def ids(self) -> list[str]:
        return list(self._by_id.keys())

    def texts(self) -> list[str]:
        return [c.text for c in self.all()]

    def apply_mutation(self, mutation: Mutation) -> Corpus:
        by_id = dict(self._by_id)
        for rid in mutation.remove:
            by_id.pop(rid, None)
        for cid, payload in mutation.chunk_payloads.items():
            if cid not in mutation.add and cid not in mutation.replace.values():
                continue
            meta = dict(payload.metadata)
            by_id[cid] = Chunk(
                chunk_id=cid,
                page_id=int(meta.get("page_id", 0)),
                revision_id=int(meta.get("revision_id", 0)),
                page_title=str(meta.get("page_title", "mutated")),
                section_title=str(meta.get("section_title", "Mutation")),
                text=payload.text,
                token_count=len(payload.text.split()),
                char_end=len(payload.text),
                category_group=meta.get("category_group"),
                metadata={**meta, "mutated": True, "failure": mutation.failure},
            )
        # also add ids listed in add that have payloads
        for cid in mutation.add:
            if cid in by_id:
                continue
            payload = mutation.chunk_payloads.get(cid)
            if payload is None:
                continue
            meta = dict(payload.metadata)
            by_id[cid] = Chunk(
                chunk_id=cid,
                page_id=int(meta.get("page_id", 0)),
                revision_id=int(meta.get("revision_id", 0)),
                page_title=str(meta.get("page_title", "mutated")),
                section_title=str(meta.get("section_title", "Mutation")),
                text=payload.text,
                token_count=len(payload.text.split()),
                char_end=len(payload.text),
                category_group=meta.get("category_group"),
                metadata={**meta, "mutated": True, "failure": mutation.failure},
            )
        return Corpus(list(by_id.values()))


def load_seeds(path: Path, limit: int | None = None) -> list[CleanSeed]:
    seeds = load_models(path, CleanSeed)
    if limit is not None:
        return seeds[:limit]
    return seeds
