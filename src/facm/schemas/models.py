from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

FailureType = Literal[
    "missing",
    "hard_negative",
    "conflict",
    "boundary",
    "noise",
]

Difficulty = Literal["easy", "medium", "hard"]
RetrieverRole = Literal["construction", "held_out"]


class SourceRef(BaseModel):
    page_id: int
    revision_id: int
    page_title: str
    section_title: str
    chunk_id: str


class Chunk(BaseModel):
    schema_version: str = "1.0"
    chunk_id: str
    page_id: int
    revision_id: int
    page_title: str
    section_path: list[str] = Field(default_factory=list)
    section_title: str
    paragraph_index: int = 0
    chunk_index: int = 0
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0
    text: str
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    category_group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CleanSeed(BaseModel):
    schema_version: str = "1.0"
    sample_id: str
    question: str
    gold_answer: str
    supporting_sentence: str = ""
    clean_contexts: list[str] = Field(default_factory=list)
    answer_type: str = "unknown"
    difficulty: str = "easy"
    reasoning_type: str = "single_fact"
    is_time_sensitive: bool = False
    source: SourceRef
    category_group: str | None = None
    quality_score: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def gold_chunk_id(self) -> str:
        return self.source.chunk_id


class ChunkPayload(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratorInfo(BaseModel):
    model: str = ""
    prompt_id: str = ""
    temperature: float = 0.0


class SemanticInfo(BaseModel):
    passed: bool | None = None
    checks: dict[str, Any] = Field(default_factory=dict)


class Mutation(BaseModel):
    schema_version: str = "1.0"
    mutation_id: str
    sample_id: str
    failure: FailureType
    original_gold_ids: list[str]
    mutated_relevant_ids: list[str]
    add: list[str] = Field(default_factory=list)
    remove: list[str] = Field(default_factory=list)
    replace: dict[str, str] = Field(default_factory=dict)
    chunk_payloads: dict[str, ChunkPayload] = Field(default_factory=dict)
    generator: GeneratorInfo = Field(default_factory=GeneratorInfo)
    semantic: SemanticInfo = Field(default_factory=SemanticInfo)
    question: str = ""
    gold_answer: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class BehaviorValidation(BaseModel):
    schema_version: str = "1.0"
    mutation_id: str
    sample_id: str
    failure: FailureType
    retriever: str
    retriever_role: RetrieverRole = "construction"
    clean_relevant_ranks: dict[str, int | None] = Field(default_factory=dict)
    mutated_relevant_ranks: dict[str, int | None] = Field(default_factory=dict)
    failure_chunk_ranks: dict[str, int | None] = Field(default_factory=dict)
    criterion_hits: dict[str, bool] = Field(default_factory=dict)
    accepted_for_retriever: bool = False
    accepted: bool | None = None
    difficulty: Difficulty | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RankedHit(BaseModel):
    chunk_id: str
    score: float
    rank: int


class RetrievalResult(BaseModel):
    schema_version: str = "1.0"
    mutation_id: str | None = None
    sample_id: str
    failure: str | None = None
    stage: str = "clean"
    retriever: str
    retriever_role: RetrieverRole
    ranked_ids: list[str]
    scores: list[float] = Field(default_factory=list)
    relevant_ids: list[str] = Field(default_factory=list)
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    mrr: float = 0.0
    mrr_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

