from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict | BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            if isinstance(row, BaseModel):
                payload = row.model_dump(mode="json")
            else:
                payload = row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_models(path: Path, model: type[T]) -> list[T]:
    return [model.model_validate(row) for row in read_jsonl(path)]
