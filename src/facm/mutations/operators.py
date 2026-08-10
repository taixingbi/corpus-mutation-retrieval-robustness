from __future__ import annotations

import json
import re

from facm.clients.chat import ChatClient
from facm.mutations.mining import DistractorMiner
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


def _find_answer_span(text: str, answer: str) -> tuple[int, int] | None:
    """Case-insensitive span of ``answer`` inside ``text``."""
    needle = answer.strip()
    if not needle or not text:
        return None
    i = text.lower().find(needle.lower())
    if i < 0:
        return None
    return i, i + len(needle)


def _contains_answer(text: str, answer: str) -> bool:
    a = answer.lower().strip()
    return bool(a) and a in text.lower()


def _mask_complete_answers(text: str, answer: str) -> str:
    """Replace full answer occurrences so a piece cannot leak the complete gold."""
    needle = answer.strip()
    if not needle or not text:
        return text
    out = text
    lower = out.lower()
    n = needle.lower()
    # Replace from the end so indices stay valid.
    starts = []
    start = 0
    while True:
        i = lower.find(n, start)
        if i < 0:
            break
        starts.append(i)
        start = i + max(1, len(n))
    for i in reversed(starts):
        out = out[:i] + "…" + out[i + len(needle) :]
        lower = out.lower()
    return out


def _split_through_answer(text: str, answer: str) -> tuple[str, str] | None:
    """Split ``text`` through the gold answer so neither piece alone has the full answer."""
    span = _find_answer_span(text, answer)
    if span is None:
        return None
    start, end = span
    if end - start < 2:
        # Single-char / empty-after-strip answers cannot be split safely.
        return None
    mid = start + (end - start) // 2
    if mid <= start:
        mid = start + 1
    if mid >= end:
        mid = end - 1
    left, right = text[:mid], text[mid:]
    # Mask any *complete* leftover answer copies (common in long wiki chunks).
    a = _mask_complete_answers(left, answer).strip()
    b = _mask_complete_answers(right, answer).strip()
    if not a or not b:
        return None
    if _contains_answer(a, answer) or _contains_answer(b, answer):
        return None
    # First-span union (pre-mask cut) must still cover the answer string.
    if not _contains_answer(left + right, answer):
        return None
    return a, b


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
    """Split gold through the answer span so neither piece alone contains gold.

    Cut the gold chunk at the midpoint of the ``gold_answer`` occurrence. Piece A
    keeps the prefix (incl. answer head); piece B keeps the suffix (incl. answer
    tail). Together they reconstruct the evidence; individually they fail the
    substring leak check used by semantic validation / human audit.
    """
    gold = corpus.get(seed.gold_chunk_id)
    if gold is None:
        return None

    # Prefer splitting the full gold chunk; fall back to supporting sentence text.
    candidates = [gold.text]
    sentence = (seed.supporting_sentence or "").strip()
    if sentence and sentence in gold.text:
        candidates.append(sentence)
    elif sentence:
        candidates.append(sentence)

    split: tuple[str, str] | None = None
    for text in candidates:
        split = _split_through_answer(text, seed.gold_answer)
        if split is not None:
            break
    if split is None:
        # Last resort: mid-split gold text only if it also satisfies no full-answer leak.
        a0, b0 = _split_sentence(gold.text)
        if (
            a0
            and b0
            and not _contains_answer(a0, seed.gold_answer)
            and not _contains_answer(b0, seed.gold_answer)
            and _contains_answer(a0 + b0, seed.gold_answer)
        ):
            split = (a0, b0)
    if split is None:
        return None

    a, b = split
    suf = "" if attempt == 1 else f"_t{attempt}"
    id_a = f"{seed.gold_chunk_id}_a{suf}"
    id_b = f"{seed.gold_chunk_id}_b{suf}"
    meta_base = {
        "page_id": gold.page_id,
        "revision_id": gold.revision_id,
        "page_title": gold.page_title,
        "section_title": gold.section_title,
    }
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
                metadata={**meta_base, "split": "a", "role": "answer_prefix"},
            ),
            id_b: ChunkPayload(
                text=b,
                metadata={**meta_base, "split": "b", "role": "answer_suffix"},
            ),
        },
        generator=GeneratorInfo(
            model="rule", prompt_id="boundary_answer_span_split", temperature=0
        ),
        question=seed.question,
        gold_answer=seed.gold_answer,
        metadata={
            "split_pieces": [a, b],
            "answer_piece_ids": [id_a, id_b],
            "attempt": attempt,
            "split_mode": "answer_span",
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
    miner: DistractorMiner | None = None,
) -> Mutation | None:
    """LLM hard-negative: same entity/relation, wrong plausible fact, lexically competitive."""
    gold = corpus.get(seed.gold_chunk_id)
    style = gold.text[:500] if gold else ""
    mined = miner.top_non_gold(seed, n=8) if miner is not None else []
    examples = ""
    if mined and miner is not None:
        examples = miner.format_examples(mined[:5])
        focus = mined[(attempt - 1) % len(mined)]
        distractor_block = (
            f"Lexically competitive non-answer distractors (style/vocab cues only):\n"
            f"{examples}\n\n"
            f"Optional focus phrase from distractor:\n{focus.text[:400]}\n\n"
        )
    else:
        distractor_block = ""

    prompt_id = "hard_negative"
    user = (
        f"Question: {seed.question}\n"
        f"Correct gold answer (FORBIDDEN — never state or paraphrase): {seed.gold_answer}\n"
        f"Gold passage style:\n{style}\n\n"
        f"{distractor_block}"
        "Write ONE hard-negative Wikipedia-style chunk (~80-150 words) that:\n"
        "1. Mentions the SAME primary entity/entities as the question\n"
        "2. Addresses the SAME relation / question intent "
        "(e.g. birth date, death date, location, role) — not a loosely related topic\n"
        "3. States a WRONG but PLAUSIBLE fact for that relation "
        "(a confident incorrect answer that a reader might believe)\n"
        "4. Is lexically similar to the question (shared names, dates, places, terms) "
        "so sparse and dense retrievers would rank it highly\n"
        "5. Never includes the gold answer value or a paraphrase of it\n"
        "6. Sounds like natural encyclopedic prose (no meta commentary)"
    )
    system = (
        "You write cross-retriever hard-negatives for IR evaluation: "
        "same entity, same relation, wrong plausible fact, high lexical overlap. "
        'Return ONLY JSON: {"text": "..."}.'
    )
    text = _llm_chunk(
        chat,
        prompt_id=prompt_id,
        system=system,
        user=user,
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
        generator=GeneratorInfo(model=model, prompt_id=prompt_id, temperature=temperature),
        question=seed.question,
        gold_answer=seed.gold_answer,
        metadata={
            "attempt": attempt,
            "baseline": "llm",
            "mined_distractor_ids": [c.chunk_id for c in mined[:5]],
        },
    )


def generate_random_hard_negative(
    seed: CleanSeed,
    corpus: Corpus,
    *,
    miner: DistractorMiner | None = None,
    attempt: int = 1,
    rank: int = 0,
) -> Mutation | None:
    """Baseline: add one BM25-related corpus chunk (no LLM)."""
    if miner is not None:
        pool = miner.top_non_gold(seed, n=max(20, rank + 5))
    else:
        q_tokens = set(seed.question.lower().split())
        pool = []
        for c in corpus.all():
            if c.chunk_id == seed.gold_chunk_id:
                continue
            if _contains_answer(c.text, seed.gold_answer):
                continue
            if q_tokens & set(c.text.lower().split()):
                pool.append(c)
        pool.sort(
            key=lambda c: len(q_tokens & set(c.text.lower().split())), reverse=True
        )
    if not pool:
        return None
    chosen = pool[min(rank, len(pool) - 1)]
    cid = f"{seed.sample_id}_hn_random_{attempt:02d}"
    return Mutation(
        mutation_id=f"{seed.sample_id}_hard_negative_random_{attempt:02d}",
        sample_id=seed.sample_id,
        failure="hard_negative",
        original_gold_ids=[seed.gold_chunk_id],
        mutated_relevant_ids=[seed.gold_chunk_id],
        add=[cid],
        chunk_payloads={
            cid: ChunkPayload(
                text=chosen.text,
                metadata={
                    "role": "hard_negative",
                    "page_title": chosen.page_title,
                    "source_chunk_id": chosen.chunk_id,
                    "page_id": chosen.page_id,
                    "revision_id": chosen.revision_id,
                    "section_title": chosen.section_title,
                },
            )
        },
        generator=GeneratorInfo(
            model="corpus",
            prompt_id="hard_negative_random",
            temperature=0.0,
        ),
        question=seed.question,
        gold_answer=seed.gold_answer,
        metadata={
            "attempt": attempt,
            "baseline": "random",
            "source_chunk_id": chosen.chunk_id,
            "bm25_rank": rank,
        },
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
    seed: CleanSeed,
    corpus: Corpus,
    n: int,
    *,
    offset: int = 0,
    miner: DistractorMiner | None = None,
) -> list[Chunk]:
    if miner is not None:
        pool = miner.top_non_gold(seed, n=max(n * 4, 20))
        start = max(0, offset * n)
        chosen = pool[start : start + n]
        if len(chosen) < n:
            for c in pool:
                if c not in chosen:
                    chosen.append(c)
                if len(chosen) >= n:
                    break
        return chosen[:n]

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
    miner: DistractorMiner | None = None,
) -> Mutation | None:
    chosen = _select_noise_chunks(
        seed, corpus, n, offset=attempt - 1, miner=miner
    )
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
    if chat is not None and miner is not None and len(add_ids) >= 1:
        try:
            examples = miner.format_examples(chosen[:3] or miner.top_non_gold(seed, n=3))
            text = _llm_chunk(
                chat,
                prompt_id="noise",
                system=(
                    "You write dense/sparse-competitive distractor paragraphs. "
                    'Return ONLY JSON: {"text": "..."}.'
                ),
                user=(
                    f"Question: {seed.question}\n"
                    f"Do NOT include the answer: {seed.gold_answer}\n"
                    f"Competing retrieved distractors:\n{examples}\n\n"
                    "Write one additional distractor (~80-120 words) that would "
                    "plausibly rank in the top results for both BM25 and dense "
                    "retrievers, without answering the question."
                ),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not _contains_answer(text, seed.gold_answer):
                cid = f"{seed.sample_id}_noise_t{attempt:02d}_llm"
                add_ids.append(cid)
                payloads[cid] = ChunkPayload(
                    text=text,
                    metadata={
                        "role": "noise",
                        "page_title": "Noise",
                        "source": "llm",
                    },
                )
        except Exception:
            pass
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
        metadata={
            "attempt": attempt,
            "source_chunk_ids": [
                (payloads[cid].metadata or {}).get("source_chunk_id")
                for cid in add_ids
                if (payloads[cid].metadata or {}).get("source_chunk_id")
            ],
        },
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
    miner: DistractorMiner | None = None,
) -> Mutation | None:
    """Generate a single (seed, failure[, attempt]) mutation."""
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
                seed,
                corpus,
                chat,
                model,
                temp,
                max_tokens,
                attempt=attempt,
                miner=miner,
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
                miner=miner,
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
