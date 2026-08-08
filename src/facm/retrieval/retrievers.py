from __future__ import annotations

import hashlib
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from facm.concurrent import map_concurrent
from facm.retrieval.corpus import Corpus
from facm.schemas.models import Chunk, Mutation


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


class Retriever(ABC):
    name: str
    role: str  # construction | held_out

    @abstractmethod
    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return (chunk_id, score) descending."""


class BM25Retriever(Retriever):
    name = "bm25"
    role = "construction"

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunk_ids = [c.chunk_id for c in chunks]
        tokenized = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.bm25 or not self.chunk_ids:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in idx]


class DenseRetriever(Retriever):
    _model_cache: dict[str, object] = {}
    _model_locks: dict[str, threading.Lock] = {}
    _cache_lock = threading.Lock()
    _default_batch_size: int = 128
    _default_device: str | None = None

    def __init__(
        self,
        chunks: Sequence[Chunk],
        model_name: str,
        name: str,
        role: str,
        query_prefix: str = "",
        doc_prefix: str = "",
        *,
        embeddings: np.ndarray | None = None,
        encode_now: bool = True,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.batch_size = batch_size or DenseRetriever._default_batch_size
        self.device = device if device is not None else DenseRetriever._default_device
        self.chunk_ids = [c.chunk_id for c in chunks]
        self._texts = [c.text for c in chunks]
        self._embeddings = embeddings
        if encode_now and self._embeddings is None:
            self._ensure()

    @classmethod
    def configure_defaults(
        cls, *, batch_size: int = 128, device: str | None = None
    ) -> str:
        """Set process-wide encode defaults. Returns resolved device string."""
        cls._default_batch_size = max(1, int(batch_size))
        resolved = device or _resolve_embed_device(None)
        cls._default_device = resolved
        return resolved

    @classmethod
    def _lock_for(cls, model_name: str) -> threading.Lock:
        with cls._cache_lock:
            if model_name not in cls._model_locks:
                cls._model_locks[model_name] = threading.Lock()
            return cls._model_locks[model_name]

    def _model(self):
        with DenseRetriever._cache_lock:
            if self.model_name not in DenseRetriever._model_cache:
                from sentence_transformers import SentenceTransformer

                dev = self.device or _resolve_embed_device(None)
                DenseRetriever._model_cache[self.model_name] = SentenceTransformer(
                    self.model_name, device=dev
                )
                print(f"[embed] loaded {self.model_name} on {dev}")
            return DenseRetriever._model_cache[self.model_name]

    def _encode(self, docs: list[str], *, show_progress: bool = False) -> np.ndarray:
        with DenseRetriever._lock_for(self.model_name):
            model = self._model()
            return np.asarray(
                model.encode(
                    docs,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=show_progress,
                )
            )

    def _ensure(self) -> None:
        if self._embeddings is not None:
            return
        docs = [self.doc_prefix + t for t in self._texts]
        self._embeddings = self._encode(docs, show_progress=len(docs) > 500)

    def mutate(self, mutation: Mutation, mutated_corpus: Corpus) -> DenseRetriever:
        """Reuse base embeddings; only encode newly added chunk texts."""
        self._ensure()
        assert self._embeddings is not None
        remove = set(mutation.remove)
        keep_ids: list[str] = []
        keep_emb: list[np.ndarray] = []
        keep_texts: list[str] = []
        id_to_row = {cid: i for i, cid in enumerate(self.chunk_ids)}
        for cid in self.chunk_ids:
            if cid in remove:
                continue
            i = id_to_row[cid]
            keep_ids.append(cid)
            keep_emb.append(self._embeddings[i])
            keep_texts.append(self._texts[i])

        new_chunks = [
            mutated_corpus.get(cid)
            for cid in mutation.add
            if mutated_corpus.get(cid) is not None and cid not in id_to_row
        ]
        new_chunks = [c for c in new_chunks if c is not None]
        if new_chunks:
            docs = [self.doc_prefix + c.text for c in new_chunks]
            new_emb = self._encode(docs, show_progress=False)
            for c, emb in zip(new_chunks, new_emb):
                keep_ids.append(c.chunk_id)
                keep_emb.append(emb)
                keep_texts.append(c.text)

        emb = np.vstack(keep_emb) if keep_emb else np.zeros((0, self._embeddings.shape[1]))
        out = DenseRetriever(
            [],
            self.model_name,
            name=self.name,
            role=self.role,
            query_prefix=self.query_prefix,
            doc_prefix=self.doc_prefix,
            embeddings=emb,
            encode_now=False,
            batch_size=self.batch_size,
            device=self.device,
        )
        out.chunk_ids = keep_ids
        out._texts = keep_texts
        return out

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if not self.chunk_ids:
            return []
        self._ensure()
        assert self._embeddings is not None
        q = self._encode([self.query_prefix + query], show_progress=False)[0]
        scores = self._embeddings @ np.asarray(q)
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in idx]


def _resolve_embed_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class HybridRetriever(Retriever):
    name = "hybrid"
    role = "construction"

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        pool = max(top_k * 5, 50)
        fused: dict[str, float] = {}
        for rank, (cid, _) in enumerate(self.bm25.search(query, pool), start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (cid, _) in enumerate(self.dense.search(query, pool), start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked


def _embed_cache_path(cache_dir: Path, model_name: str, doc_prefix: str) -> Path:
    slug = model_name.replace("/", "__")
    # doc_prefix affects vectors (e.g. e5 "passage: ")
    tag = "nopre" if not doc_prefix else hashlib.sha1(doc_prefix.encode()).hexdigest()[:8]
    return cache_dir / f"{slug}__{tag}.npz"


def _try_load_embed_cache(
    cache_dir: Path | None,
    model_name: str,
    doc_prefix: str,
    chunk_ids: Sequence[str],
) -> np.ndarray | None:
    if cache_dir is None:
        return None
    path = _embed_cache_path(cache_dir, model_name, doc_prefix)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        cached_ids = [str(x) for x in data["chunk_ids"].tolist()]
        if cached_ids != list(chunk_ids):
            print(f"[embed] cache miss (chunk_ids changed): {path.name}")
            return None
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        if emb.shape[0] != len(chunk_ids):
            return None
        print(f"[embed] cache hit {path.name} shape={emb.shape}")
        return emb
    except Exception as exc:
        print(f"[embed] cache load failed ({path.name}): {exc}")
        return None


def _save_embed_cache(
    cache_dir: Path | None,
    model_name: str,
    doc_prefix: str,
    chunk_ids: Sequence[str],
    embeddings: np.ndarray,
) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _embed_cache_path(cache_dir, model_name, doc_prefix)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        chunk_ids=np.asarray(list(chunk_ids), dtype=object),
    )
    tmp.replace(path)
    print(f"[embed] wrote cache {path.name}")


def build_retrievers(
    chunks: Sequence[Chunk],
    dense_models: dict[str, str],
    rrf_k: int = 60,
    *,
    embed_concurrency: int | None = 1,
    embed_batch_size: int = 128,
    embed_device: str | None = None,
    cache_dir: Path | str | None = None,
) -> dict[str, Retriever]:
    """Build BM25 + dense retrievers. Distinct dense models encode concurrently.

    When ``cache_dir`` is set, base-corpus embeddings are loaded/saved there so
    stage0 / stage3 / stage4 (and re-runs) do not re-encode.
    """
    device = DenseRetriever.configure_defaults(
        batch_size=embed_batch_size, device=embed_device
    )
    cache_path = Path(cache_dir) if cache_dir is not None else None
    bm25 = BM25Retriever(chunks)
    chunk_ids = [c.chunk_id for c in chunks]

    dense_specs = [
        ("bge_base", dense_models["bge_base"], "construction", "", ""),
        (
            "e5_large",
            dense_models["e5_large"],
            "held_out",
            "query: ",
            "passage: ",
        ),
        ("gte_large", dense_models["gte_large"], "held_out", "", ""),
    ]

    unique_models: dict[str, list[tuple]] = {}
    for spec in dense_specs:
        unique_models.setdefault(spec[1], []).append(spec)

    def _build_one(model_name: str) -> dict[str, DenseRetriever]:
        specs = unique_models[model_name]
        name, _, role, qpre, dpre = specs[0]
        cached = _try_load_embed_cache(cache_path, model_name, dpre, chunk_ids)
        primary = DenseRetriever(
            chunks,
            model_name,
            name=name,
            role=role,
            query_prefix=qpre,
            doc_prefix=dpre,
            embeddings=cached,
            encode_now=cached is None,
            batch_size=embed_batch_size,
            device=device,
        )
        if cached is None:
            primary._ensure()
            assert primary._embeddings is not None
            _save_embed_cache(
                cache_path, model_name, dpre, primary.chunk_ids, primary._embeddings
            )
        out: dict[str, DenseRetriever] = {name: primary}
        for name_i, _, role_i, qpre_i, dpre_i in specs[1:]:
            if dpre_i == dpre and qpre_i == qpre:
                clone = DenseRetriever(
                    [],
                    model_name,
                    name=name_i,
                    role=role_i,
                    query_prefix=qpre_i,
                    doc_prefix=dpre_i,
                    embeddings=primary._embeddings,
                    encode_now=False,
                    batch_size=embed_batch_size,
                    device=device,
                )
                clone.chunk_ids = list(primary.chunk_ids)
                clone._texts = list(primary._texts)
                out[name_i] = clone
            else:
                cached_i = _try_load_embed_cache(cache_path, model_name, dpre_i, chunk_ids)
                out[name_i] = DenseRetriever(
                    chunks,
                    model_name,
                    name=name_i,
                    role=role_i,
                    query_prefix=qpre_i,
                    doc_prefix=dpre_i,
                    embeddings=cached_i,
                    encode_now=cached_i is None,
                    batch_size=embed_batch_size,
                    device=device,
                )
                if cached_i is None:
                    out[name_i]._ensure()
                    assert out[name_i]._embeddings is not None
                    _save_embed_cache(
                        cache_path,
                        model_name,
                        dpre_i,
                        out[name_i].chunk_ids,
                        out[name_i]._embeddings,
                    )
        return out

    model_names = list(unique_models.keys())
    # None = all models in parallel; int (default 1) caps concurrent encodes
    workers = len(model_names) if embed_concurrency is None else embed_concurrency
    print(
        f"[embed] device={device} batch_size={embed_batch_size} "
        f"parallel_models={min(max(1, workers), len(model_names))}/{len(model_names)}"
        + (f" cache={cache_path}" if cache_path else "")
    )
    built_maps = map_concurrent(
        model_names,
        _build_one,
        max_concurrency=min(max(1, workers), len(model_names)),
        desc="embed-models",
    )
    dense_by_name: dict[str, DenseRetriever] = {}
    for m in built_maps:
        dense_by_name.update(m)

    bge = dense_by_name["bge_base"]
    hybrid = HybridRetriever(bm25, bge, rrf_k=rrf_k)
    return {
        "bm25": bm25,
        "bge_base": bge,
        "hybrid": hybrid,
        "e5_large": dense_by_name["e5_large"],
        "gte_large": dense_by_name["gte_large"],
    }


def apply_mutation_retrievers(
    base: dict[str, Retriever],
    base_corpus: Corpus,
    mutation: Mutation,
    rrf_k: int = 60,
) -> tuple[Corpus, dict[str, Retriever]]:
    """Cheap mutated indexes: BM25 rebuild + dense embedding reuse."""
    mutated_corpus = base_corpus.apply_mutation(mutation)
    chunks = mutated_corpus.all()
    out: dict[str, Retriever] = {}
    bge_mut: DenseRetriever | None = None
    for name, ret in base.items():
        if isinstance(ret, DenseRetriever):
            mut = ret.mutate(mutation, mutated_corpus)
            out[name] = mut
            if name == "bge_base":
                bge_mut = mut
        elif isinstance(ret, BM25Retriever):
            out[name] = BM25Retriever(chunks)
        elif isinstance(ret, HybridRetriever):
            continue
        else:
            out[name] = ret
    bm25 = out.get("bm25")
    if isinstance(bm25, BM25Retriever) and bge_mut is not None:
        out["hybrid"] = HybridRetriever(bm25, bge_mut, rrf_k=rrf_k)
    elif isinstance(base.get("hybrid"), HybridRetriever):
        out["hybrid"] = HybridRetriever(
            BM25Retriever(chunks),
            bge_mut
            or DenseRetriever(
                chunks,
                "sentence-transformers/all-MiniLM-L6-v2",
                "bge_base",
                "construction",
            ),
            rrf_k=rrf_k,
        )
    return mutated_corpus, out
