from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ChatConfig(BaseModel):
    model: str = "openai.gpt-oss-120b-1:0"
    temperature: float = 0.0
    max_tokens: int = 1024
    judge_max_tokens: int = 1024


class AppConfig(BaseModel):
    run_id: str = "smoke"
    base_dir: str = "data/runs/smoke/0_base"
    output_dir: str = "data/runs/smoke"
    failures: list[str] = Field(
        default_factory=lambda: [
            "missing",
            "hard_negative",
            "conflict",
            "boundary",
            "noise",
        ]
    )
    construction_retrievers: list[str] = Field(
        default_factory=lambda: ["bm25", "bge_base", "hybrid"]
    )
    held_out_retrievers: list[str] = Field(
        default_factory=lambda: ["e5_large", "gte_large"]
    )
    dense_models: dict[str, str] = Field(
        default_factory=lambda: {
            "bge_base": "BAAI/bge-base-en-v1.5",
            "e5_large": "intfloat/e5-large-v2",
            "gte_large": "thenlper/gte-large",
        }
    )
    rrf_k: int = 60
    top_k: int = 20
    rerank_top_n: int = 5
    chat: ChatConfig = Field(default_factory=ChatConfig)
    rerankers: list[str] = Field(
        default_factory=lambda: [
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            "jinaai/jina-reranker-v2-base-multilingual",
            "mixedbread-ai/mxbai-rerank-large-v1",
            "BAAI/bge-reranker-v2-m3",
        ]
    )
    noise_add_count: int = 3
    behavioral_majority: int = 2
    seed_limit: int | None = None
    # Sync ThreadPool concurrency (1 = fully sequential)
    llm_concurrency: int = 8
    # Dense encode: 1 = one model at a time (safer on MPS / shared GPU)
    embed_concurrency: int = 1
    embed_batch_size: int = 128
    # null = auto (cuda if available else cpu)
    embed_device: str | None = None
    # Remote ECS rerank: 1 = no concurrent /v1/rerank calls
    rerank_concurrency: int = 1
    http_retries: int = 5
    # Shared Hybrid candidate pool size for ALL rerankers (fair compare)
    rerank_pool_size: int = 20
    # Per-failure generation attempts (oversample weak RQ4 funnels)
    mutation_oversample: dict[str, int] = Field(
        default_factory=lambda: {
            "missing": 1,
            "hard_negative": 1,
            "conflict": 1,
            "boundary": 1,
            "noise": 1,
        }
    )
    # Soft target for recoverable (or missing control) cases in RQ4 reporting
    rq4_target_recoverable: int = 150
    bootstrap_n: int = 1000
    bootstrap_seed: int = 0

    @property
    def base_path(self) -> Path:
        return Path(self.base_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def load_config(path: str | Path) -> AppConfig:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    return AppConfig.model_validate(data)
