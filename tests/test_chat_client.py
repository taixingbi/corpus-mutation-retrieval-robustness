from facm.clients.chat import extract_message_content, is_truncated_completion


def test_extract_prefers_content_over_reasoning():
    msg = {
        "role": "assistant",
        "reasoning_content": "long chain of thought " * 50,
        "content": '{"topical":true,"natural":true,"failure_intent_ok":true}',
    }
    assert extract_message_content(msg).startswith('{"topical"')


def test_extract_skips_reasoning_content_parts():
    msg = {
        "content": [
            {"type": "reasoning", "text": "thinking..."},
            {"type": "text", "text": '{"natural":true}'},
        ]
    }
    assert extract_message_content(msg) == '{"natural":true}'


def test_truncated_when_completion_hits_max_tokens():
    assert is_truncated_completion(
        finish_reason="stop",  # gateway may lie
        usage={"completion_tokens": 64},
        max_tokens=64,
        content='{"topical":true,"natural":true,"',
    )


def test_truncated_finish_reason_length():
    assert is_truncated_completion(
        finish_reason="length",
        usage={"completion_tokens": 10},
        max_tokens=1024,
        content="{}",
    )


def test_not_truncated_complete_json():
    assert not is_truncated_completion(
        finish_reason="stop",
        usage={"completion_tokens": 40},
        max_tokens=1024,
        content='{"topical":true,"natural":true,"failure_intent_ok":true}',
    )
