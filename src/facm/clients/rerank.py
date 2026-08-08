from __future__ import annotations

import os
import threading
from typing import Any

import httpx
from dotenv import load_dotenv

from facm.clients.http_retry import request_with_retry


class RerankClient:
    """ECS reranker MVP client (POST /v1/rerank).

    By default only one in-flight request is allowed process-wide so concurrent
    stage4 workers do not stampede a flaky ALB.
    """

    _gate: threading.Semaphore | None = None
    _gate_lock = threading.Lock()
    _gate_slots: int | None = None

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
        *,
        concurrency: int = 1,
        retries: int = 5,
    ) -> None:
        load_dotenv()
        self.base_url = (base_url or os.getenv("FACM_RERANK_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("FACM_RERANK_API_KEY") or "1234"
        self.timeout = timeout
        self.concurrency = max(1, int(concurrency))
        self.retries = max(0, int(retries))
        if not self.base_url:
            raise RuntimeError(
                "FACM_RERANK_URL is not set. See example_reranker.md / .env.example"
            )
        self._ensure_gate(self.concurrency)

    @classmethod
    def _ensure_gate(cls, slots: int) -> None:
        with cls._gate_lock:
            if cls._gate is None or cls._gate_slots != slots:
                cls._gate = threading.Semaphore(slots)
                cls._gate_slots = slots

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _get(self, path: str) -> dict[str, Any]:
        def _once() -> dict[str, Any]:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.base_url}{path}", headers=self._headers())
                resp.raise_for_status()
                return resp.json()

        assert self._gate is not None
        with self._gate:
            return request_with_retry(
                _once, retries=self.retries, label=f"rerank{path}"
            )

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def models(self) -> dict[str, Any]:
        return self._get("/v1/models")

    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: list[str],
        top_n: int = 5,
        return_documents: bool = True,
    ) -> list[dict[str, Any]]:
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        }
        headers = {**self._headers(), "Content-Type": "application/json"}

        def _once() -> list[dict[str, Any]]:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/v1/rerank", headers=headers, json=payload
                )
                resp.raise_for_status()
                data = resp.json()
            if data.get("error"):
                raise RuntimeError(
                    f"rerank error: {data.get('error')} {data.get('detail')}"
                )
            return list(data.get("results") or [])

        assert self._gate is not None
        with self._gate:
            return request_with_retry(
                _once, retries=self.retries, label=f"rerank:{model}"
            )
