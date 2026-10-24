from __future__ import annotations

import json
import re

from facm.clients.chat import ChatClient
from facm.schemas.models import Mutation, SemanticInfo

# Structural / leak checks — always required.
RULE_KEYS = {
    "missing": ("gold_removed", "no_relevant_ids"),
    "boundary": (
        "gold_removed",
        "two_splits",
        "pieces_present",
        "pieces_nonempty",
        "answer_not_complete_in_either_piece",
        "answer_covered_by_union",
    ),
    "hard_negative": ("has_hn_chunk", "answer_not_in_hn", "gold_kept"),
    "conflict": ("has_conflict_chunk", "gold_kept"),
    "noise": ("has_noise", "answer_not_in_noise", "gold_kept"),
}

# LLM judge is advisory for generation quality; not used for rule-only ops.
LLM_REQUIRED_KEYS = {
    "hard_negative": ("llm_natural", "llm_failure_intent_ok"),
    "conflict": ("llm_natural", "llm_failure_intent_ok"),
    # noise / boundary: rules only (noise distractors need not be "relevant";
    # boundary splits are structural by design)
}

FAILURE_JUDGE_INSTRUCTIONS = {
    "hard_negative": (
        "A hard-negative chunk must be Wikipedia-style and topically related to the "
        "question, but must NOT contain or entail the gold answer. "
        "Set failure_intent_ok=true if it is a plausible near-miss distractor. "
        "Set natural=true if prose looks encyclopedic. "
        "Set topical=true if it shares topic with the question (even if it does not answer it)."
    ),
    "conflict": (
        "A conflict chunk must be Wikipedia-style and assert an alternate/incorrect "
        "answer (or contradict the gold answer) for the question. "
        "Set failure_intent_ok=true if it clearly conflicts with the gold answer. "
        "Set natural=true if prose looks encyclopedic. "
        "Set topical=true if it addresses the same question/entity."
    ),
    "noise": (
        "Noise chunks are distractors; they need not answer the question. "
        "Set failure_intent_ok=true if they look like plausible corpus distractors "
        "without leaking the gold answer. Set natural=true if readable. "
        "Set topical=true if loosely related OR false if generic distractors (both OK)."
    ),
    "boundary": (
        "Boundary mutations split the gold answer across chunks so neither piece "
        "alone contains the full answer. Set failure_intent_ok=true if the answer "
        "is incomplete in each piece but recoverable from both. Set natural=true "
        "if fragments still look like Wikipedia text. Set topical=true if fragments "
        "relate to the gold."
    ),
}


def _contains_answer(text: str, answer: str) -> bool:
    a = answer.lower().strip()
    if not a:
        return False
    return a in text.lower()


def _extract_json_obj(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def rule_checks(mutation: Mutation) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    texts = [p.text for p in mutation.chunk_payloads.values()]
    joined = "\n".join(texts)

    if mutation.failure == "missing":
        checks["gold_removed"] = bool(mutation.original_gold_ids) and (
            mutation.original_gold_ids[0] in mutation.remove
        )
        checks["no_relevant_ids"] = len(mutation.mutated_relevant_ids) == 0
    elif mutation.failure == "boundary":
        checks["gold_removed"] = all(
            g in mutation.remove for g in mutation.original_gold_ids
        )
        checks["two_splits"] = len(mutation.add) >= 2
        pieces = mutation.metadata.get("split_pieces") or []
        if isinstance(pieces, list) and len(pieces) >= 2:
            checks["pieces_present"] = all(
                any(str(p) in t for t in texts) for p in pieces
            )
            checks["pieces_nonempty"] = all(str(p).strip() for p in pieces)
        else:
            checks["pieces_present"] = False
            checks["pieces_nonempty"] = False
        piece_texts = [str(p) for p in pieces] if isinstance(pieces, list) and pieces else texts
        checks["answer_not_complete_in_either_piece"] = all(
            not _contains_answer(t, mutation.gold_answer) for t in piece_texts if str(t).strip()
        ) and bool(piece_texts)
        union = "".join(str(t) for t in piece_texts)
        checks["answer_covered_by_union"] = _contains_answer(
            union, mutation.gold_answer
        ) or _contains_answer(" ".join(str(t) for t in piece_texts), mutation.gold_answer)
    elif mutation.failure == "hard_negative":
        checks["has_hn_chunk"] = len(mutation.add) >= 1
        checks["answer_not_in_hn"] = not _contains_answer(joined, mutation.gold_answer)
        checks["gold_kept"] = mutation.original_gold_ids[0] not in mutation.remove
    elif mutation.failure == "conflict":
        checks["has_conflict_chunk"] = len(mutation.add) >= 1
        checks["gold_kept"] = mutation.original_gold_ids[0] not in mutation.remove
        # soft structural: conflict text should not simply copy the gold answer as fact
        checks["conflict_text_nonempty"] = bool(joined.strip())
    elif mutation.failure == "noise":
        checks["has_noise"] = len(mutation.add) >= 1
        checks["answer_not_in_noise"] = not _contains_answer(joined, mutation.gold_answer)
        checks["gold_kept"] = mutation.original_gold_ids[0] not in mutation.remove
    return checks


def llm_semantic_judge(
    mutation: Mutation,
    chat: ChatClient,
    model: str,
    *,
    max_tokens: int = 1024,
) -> dict:
    """LLM semantic gate — short JSON only (reasoning models burn tokens easily)."""
    payloads = {cid: p.text[:400] for cid, p in list(mutation.chunk_payloads.items())[:3]}
    instruction = FAILURE_JUDGE_INSTRUCTIONS.get(mutation.failure, "")
    prompt = (
        f"Failure type: {mutation.failure}\n"
        f"Question: {mutation.question}\n"
        f"Gold answer: {mutation.gold_answer}\n"
        f"Chunks:\n{json.dumps(payloads, ensure_ascii=False)}\n\n"
        f"{instruction}\n\n"
        "Output ONE JSON object and nothing else.\n"
        'Exact keys only: {"topical":true|false,"natural":true|false,"failure_intent_ok":true|false}\n'
        "No notes. No markdown. No explanation."
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an IR mutation validator. "
                "Decide three booleans. Be moderately permissive. "
                "Final answer must be a single compact JSON object with exactly three keys."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    result = chat.complete(
        messages, temperature=0, max_tokens=max_tokens, model=model
    )
    if result.truncated or not _extract_json_obj(result.content):
        result = chat.complete(
            messages,
            temperature=0,
            max_tokens=max(max_tokens * 2, 1024),
            model=model,
        )

    meta = {
        "llm_finish_reason": result.finish_reason,
        "llm_usage": result.usage,
        "llm_truncated": result.truncated,
    }
    data = _extract_json_obj(result.content)
    has_required = (
        "natural" in data
        and "failure_intent_ok" in data
        and ("topical" in data or "relevant" in data)
    )
    if result.truncated or not has_required:
        meta["llm_parse_failed"] = True
        meta["llm_raw"] = result.content[:500]
        return meta

    return {
        **meta,
        "llm_relevant": bool(data.get("topical", data.get("relevant", False))),
        "llm_natural": bool(data.get("natural", False)),
        "llm_failure_intent_ok": bool(data.get("failure_intent_ok", False)),
    }


def _rules_passed(mutation: Mutation, checks: dict) -> bool:
    keys = RULE_KEYS.get(mutation.failure, ())
    return all(bool(checks.get(k)) for k in keys)


def _llm_passed(mutation: Mutation, checks: dict) -> bool:
    keys = LLM_REQUIRED_KEYS.get(mutation.failure)
    if not keys:
        return True
    # Fail-closed: unavailable / error / truncated / unparsable judge => gate fails.
    if (
        checks.get("llm_unavailable")
        or checks.get("llm_error")
        or checks.get("llm_parse_failed")
        or checks.get("llm_truncated")
    ):
        return False
    return all(bool(checks.get(k)) for k in keys)


def validate_semantic(
    mutation: Mutation,
    *,
    chat: ChatClient | None = None,
    model: str = "openai.gpt-oss-120b-1:0",
    use_llm: bool = True,
    judge_max_tokens: int = 1024,
) -> Mutation:
    checks: dict = rule_checks(mutation)

    needs_llm = (
        use_llm
        and chat is not None
        and mutation.failure in LLM_REQUIRED_KEYS
        and bool(mutation.chunk_payloads)
    )
    if needs_llm:
        try:
            llm_checks = llm_semantic_judge(
                mutation, chat, model, max_tokens=judge_max_tokens
            )
            raw = llm_checks.pop("llm_raw", None)
            usage = llm_checks.pop("llm_usage", None)
            finish = llm_checks.pop("llm_finish_reason", None)
            checks.update(llm_checks)
            if raw:
                mutation.metadata["semantic_raw"] = raw
            if usage is not None:
                mutation.metadata["semantic_usage"] = usage
            if finish is not None:
                mutation.metadata["semantic_finish_reason"] = finish
        except Exception as exc:
            checks["llm_error"] = True
            mutation.metadata["semantic_error"] = str(exc)
    elif use_llm and mutation.failure in LLM_REQUIRED_KEYS and chat is None:
        checks["llm_unavailable"] = True

    passed = _rules_passed(mutation, checks) and _llm_passed(mutation, checks)
    checks["rules_passed"] = _rules_passed(mutation, checks)
    checks["llm_gate_passed"] = _llm_passed(mutation, checks)
    mutation.semantic = SemanticInfo(passed=passed, checks=checks)
    return mutation
