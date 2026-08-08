from __future__ import annotations

import json
import re

from facm.clients.chat import ChatClient
from facm.retrieval.corpus import Corpus
from facm.schemas.models import (
    Chunk,
    ChunkPayload,
    CleanSeed,
    GeneratorInfo,
    Mutation,
)


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        return json.loads(m.group(0))


def _contains_answer(text: str, answer: str) -> bool:
    return answer.lower().strip() in text.lower()


def _split_sentence(sentence: str) -> tuple[str, str]:
    words = sentence.split()
    if len(words) < 4:
        mid = max(1, len(sentence) // 2)
        return sentence[:mid].strip(), sentence[mid:].strip()
    mid = len(words) // 2
    return " ".join(words[:mid]), " ".join(words[mid:])


def _llm_chunk(
    chat: ChatClient,
    *,
    prompt_id: str,
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    content = chat.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )
    data = _extract_json(content)
    text = str(data.get("text") or data.get("chunk") or "").strip()
    if not text:
        raise ValueError(f"{prompt_id}: empty chunk text from LLM")
    return text


def _mid(seed: CleanSeed, failure: str, attempt: int = 1) -> str:
    return f"{seed.sample_id}_{failure}_{attempt:02d}"


def generate_missing(
    seed: CleanSeed, model: str = "rule", *, attempt: int = 1
) -> Mutation:
    gold = seed.gold_chunk_id
    return Mutation(
        mutation_id=_mid(seed, "missing", attempt),
        sample_id=seed.sample_id,
        failure="missing",
        original_gold_ids=[gold],
        mutated_relevant_ids=[],
        remove=[gold],
        generator=GeneratorInfo(model=model, prompt_id="missing_remove_gold", temperature=0),
        question=seed.question,
        gold_answer=seed.gold_answer,
        metadata={"attempt": attempt},
    )


def generate_boundary(
    seed: CleanSeed, corpus: Corpus, *, attempt: int = 1
) -> Mutation | None:
    """Split gold so full evidence is no longer in one chunk.

    Piece A: surrounding context with the supporting sentence removed.
    Piece B: supporting sentence (answer-bearing piece).
    """
    gold = corpus.get(seed.gold_chunk_id)
    if gold is None:
        return None
    sentence = (seed.supporting_sentence or "").strip()
    if not sentence or sentence not in gold.text:
        # Fall back to mid-split of gold text; mark second half as answer piece.
        a, b = _split_sentence(gold.text)
        if not a or not b:
            return None
        suf = "" if attempt == 1 else f"_t{attempt}"
        id_a = f"{seed.gold_chunk_id}_a{suf}"
        id_b = f"{seed.gold_chunk_id}_b{suf}"
        return Mutation(
            mutation_id=_mid(seed, "boundary", attempt),
            sample_id=seed.sample_id,
            failure="boundary",
            original_gold_ids=[seed.gold_chunk_id],
            mutated_relevant_ids=[id_a, id_b],
            add=[id_a, id_b],
            remove=[seed.gold_chunk_id],
            chunk_payloads={
                id_a: ChunkPayload(
                    text=a,
                    metadata={
                        "page_id": gold.page_id,
                        "revision_id": gold.revision_id,
                        "page_title": gold.page_title,
                        "section_title": gold.section_title,
                        "split": "a",
                        "role": "context_piece",
                    },
                ),
                id_b: ChunkPayload(
                    text=b,
                    metadata={
                        "page_id": gold.page_id,
                        "revision_id": gold.revision_id,
                        "page_title": gold.page_title,
                        "section_title": gold.section_title,
                        "split": "b",
                        "role": "answer_piece",
                    },
                ),
            },
            generator=GeneratorInfo(model="rule", prompt_id="boundary_split", temperature=0),
            question=seed.question,
            gold_answer=seed.gold_answer,
            metadata={
                "split_pieces": [a, b],
                "answer_piece_ids": [id_b],
                "attempt": attempt,
            },
        )

    context = gold.text.replace(sentence, " ").strip()
    context = " ".join(context.split())
    if not context:
        # Ensure A is non-empty: use first half of sentence as weak context.
        a_fallback, b_fallback = _split_sentence(sentence)
        context, sentence = a_fallback, b_fallback
    suf = "" if attempt == 1 else f"_t{attempt}"
    id_a = f"{seed.gold_chunk_id}_a{suf}"
    id_b = f"{seed.gold_chunk_id}_b{suf}"
    return Mutation(
        mutation_id=_mid(seed, "boundary", attempt),
        sample_id=seed.sample_id,
        failure="boundary",
        original_gold_ids=[seed.gold_chunk_id],
        mutated_relevant_ids=[id_a, id_b],
        add=[id_a, id_b],
        remove=[seed.gold_chunk_id],
        chunk_payloads={
            id_a: ChunkPayload(
                text=context,
                metadata={
                    "page_id": gold.page_id,
                    "revision_id": gold.revision_id,
                    "page_title": gold.page_title,
                    "section_title": gold.section_title,
                    "split": "a",
                    "role": "context_piece",
                },
            ),
            id_b: ChunkPayload(
                text=sentence,
                metadata={
                    "page_id": gold.page_id,
                    "revision_id": gold.revision_id,
                    "page_title": gold.page_title,
                    "section_title": gold.section_title,
                    "split": "b",
                    "role": "answer_piece",
                },
            ),
        },
        generator=GeneratorInfo(model="rule", prompt_id="boundary_split", temperature=0),
        question=seed.question,
        gold_answer=seed.gold_answer,
        metadata={
            "split_pieces": [context, sentence],
            "answer_piece_ids": [id_b],
            "attempt": attempt,
        },
    )


def generate_hard_negative(
    seed: CleanSeed,
    corpus: Corpus,
    chat: ChatClient,
    model: str,
    temperature: float,
    max_tokens: int,
    *,
    attempt: int = 1,
) -> Mutation | None:
    gold = corpus.get(seed.gold_chunk_id)
    style = gold.text[:500] if gold else ""
    text = _llm_chunk(
        chat,
        prompt_id="hard_negative",
        system=(
            "You write Wikipedia-style passages for an IR hard-negative benchmark. "
            'Return ONLY JSON: {"text": "..."}.'
        ),
        user=(
            f"Question: {seed.question}\n"
            f"Gold answer (must NOT appear anywhere, including paraphrases): {seed.gold_answer}\n"
            f"Style reference:\n{style}\n\n"
            "Write ONE hard-negative chunk (~80-150 words):\n"
            "- Same entity/topic as the question (topical near-miss)\n"
            "- Natural encyclopedic Wikipedia prose\n"
            "- Discuss related facts that do NOT answer the question\n"
            "- Do not mention the gold answer value at all"
        ),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if _contains_answer(text, seed.gold_answer):
        return None
    cid = f"{seed.sample_id}_hn_{attempt:02d}"
    return Mutation(
        mutation_id=_mid(seed, "hard_negative", attempt),
        sample_id=seed.sample_id,
        failure="hard_negative",
        original_gold_ids=[seed.gold_chunk_id],
        mutated_relevant_ids=[seed.gold_chunk_id],
        add=[cid],
        chunk_payloads={
            cid: ChunkPayload(
                text=text,
                metadata={"role": "hard_negative", "page_title": "HardNegative"},
            )
        },
        generator=GeneratorInfo(model=model, prompt_id="hard_negative", temperature=temperature),
        question=seed.question,
        gold_answer=seed.gold_answer,
        metadata={"attempt": attempt},
    )


def generate_conflict(
    seed: CleanSeed,
    corpus: Corpus,
    chat: ChatClient,
    model: str,
    temperature: float,
    max_tokens: int,
    *,
    attempt: int = 1,
) -> Mutation | None:
    gold = corpus.get(seed.gold_chunk_id)
    style = gold.text[:500] if gold else ""
    text = _llm_chunk(
        chat,
        prompt_id="conflict",
        system=(
            "You write Wikipedia-style passages that contradict a known fact. "
            'Return ONLY JSON: {"text": "..."}.'
        ),
        user=(
            f"Question: {seed.question}\n"
            f"Correct gold answer (you must CONTRADICT this): {seed.gold_answer}\n"
            f"Style reference:\n{style}\n\n"
            "Write ONE conflicting chunk (~80-150 words):\n"
            "- Same entity/topic as the question\n"
            "- Explicitly state a different incorrect answer to the question\n"
            "- Sound like a confident Wikipedia paragraph\n"
            "- Do not acknowledge that the claim is wrong"
        ),
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    cid = f"{seed.sample_id}_conflict_{attempt:02d}"
    return Mutation(
        mutation_id=_mid(seed, "conflict", attempt),
        sample_id=seed.sample_id,
        failure="conflict",
        original_gold_ids=[seed.gold_chunk_id],
        mutated_relevant_ids=[seed.gold_chunk_id],
        add=[cid],
        chunk_payloads={
            cid: ChunkPayload(
                text=text,
                metadata={"role": "conflict", "page_title": "Conflict"},
            )
        },
        generator=GeneratorInfo(model=model, prompt_id="conflict", temperature=temperature),
        question=seed.question,
        gold_answer=seed.gold_answer,
        metadata={"attempt": attempt},
    )


def _select_noise_chunks(
    seed: CleanSeed, corpus: Corpus, n: int, *, offset: int = 0
) -> list[Chunk]:
    q_tokens = set(seed.question.lower().split())
    pool: list[Chunk] = []
    for c in corpus.all():
        if c.chunk_id == seed.gold_chunk_id:
            continue
        if _contains_answer(c.text, seed.gold_answer):
            continue
        if len(q_tokens & set(c.text.lower().split())) > 0:
            pool.append(c)
    pool.sort(key=lambda c: len(q_tokens & set(c.text.lower().split())), reverse=True)
    start = max(0, offset * n)
    chosen = pool[start : start + n]
    if len(chosen) < n and start > 0:
        # wrap / backfill from head for remaining slots
        for c in pool:
            if c in chosen:
                continue
            chosen.append(c)
            if len(chosen) >= n:
                break
    return chosen[:n]


def generate_noise(
    seed: CleanSeed,
    corpus: Corpus,
    n: int,
    chat: ChatClient | None = None,
    model: str = "corpus",
    temperature: float = 0.0,
    max_tokens: int = 512,
    *,
    attempt: int = 1,
) -> Mutation | None:
    chosen = _select_noise_chunks(seed, corpus, n, offset=attempt - 1)
    payloads: dict[str, ChunkPayload] = {}
    add_ids: list[str] = []
    for i, c in enumerate(chosen):
        cid = f"{seed.sample_id}_noise_t{attempt:02d}_{i:02d}"
        add_ids.append(cid)
        payloads[cid] = ChunkPayload(
            text=c.text,
            metadata={
                "role": "noise",
                "source_chunk_id": c.chunk_id,
                "page_id": c.page_id,
                "revision_id": c.revision_id,
                "page_title": c.page_title,
                "section_title": c.section_title,
            },
        )
    while chat is not None and len(add_ids) < n:
        i = len(add_ids)
        try:
            text = _llm_chunk(
                chat,
                prompt_id="noise",
                system='Return ONLY JSON: {"text": "..."} for a topical distractor.',
                user=(
                    f"Question: {seed.question}\n"
                    f"Do NOT include the answer: {seed.gold_answer}\n"
                    "Write a distracting but related Wikipedia-style paragraph."
                ),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            break
        if _contains_answer(text, seed.gold_answer):
            continue
        cid = f"{seed.sample_id}_noise_t{attempt:02d}_{i:02d}"
        add_ids.append(cid)
        payloads[cid] = ChunkPayload(
            text=text, metadata={"role": "noise", "page_title": "Noise", "source": "llm"}
        )
    if not add_ids:
        return None
    return Mutation(
        mutation_id=_mid(seed, "noise", attempt),
        sample_id=seed.sample_id,
        failure="noise",
        original_gold_ids=[seed.gold_chunk_id],
        mutated_relevant_ids=[seed.gold_chunk_id],
        add=add_ids,
        chunk_payloads=payloads,
        generator=GeneratorInfo(
            model=model if chat else "corpus",
            prompt_id="noise",
            temperature=temperature,
        ),
        question=seed.question,
        gold_answer=seed.gold_answer,
        metadata={"attempt": attempt},
    )


def generate_one_mutation(
    seed: CleanSeed,
    corpus: Corpus,
    failure: str,
    *,
    chat: ChatClient | None,
    model: str,
    temperature: float,
    max_tokens: int,
    noise_add_count: int,
    attempt: int = 1,
) -> Mutation | None:
    """Generate a single (seed, failure[, attempt]) mutation."""
    # Deterministic operators: only one useful attempt
    if failure in {"missing", "boundary"} and attempt > 1:
        return None
    temp = temperature
    if attempt > 1 and failure in {"hard_negative", "conflict", "noise"}:
        temp = min(0.9, temperature + 0.25 * (attempt - 1))
    try:
        if failure == "missing":
            return generate_missing(seed, attempt=attempt)
        if failure == "boundary":
            return generate_boundary(seed, corpus, attempt=attempt)
        if failure == "hard_negative":
            if chat is None:
                return None
            return generate_hard_negative(
                seed, corpus, chat, model, temp, max_tokens, attempt=attempt
            )
        if failure == "conflict":
            if chat is None:
                return None
            return generate_conflict(
                seed, corpus, chat, model, temp, max_tokens, attempt=attempt
            )
        if failure == "noise":
            return generate_noise(
                seed,
                corpus,
                noise_add_count,
                chat=chat,
                model=model,
                temperature=temp,
                max_tokens=max_tokens,
                attempt=attempt,
            )
    except Exception as exc:
        print(f"[mutate] {seed.sample_id}/{failure}/t{attempt} failed: {exc}")
        return None
    return None


def generate_mutations_for_seed(
    seed: CleanSeed,
    corpus: Corpus,
    failures: list[str],
    *,
    chat: ChatClient | None,
    model: str,
    temperature: float,
    max_tokens: int,
    noise_add_count: int,
) -> list[Mutation]:
    out: list[Mutation] = []
    for failure in failures:
        mut = generate_one_mutation(
            seed,
            corpus,
            failure,
            chat=chat,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            noise_add_count=noise_add_count,
            attempt=1,
        )
        if mut is not None:
            out.append(mut)
    return out
