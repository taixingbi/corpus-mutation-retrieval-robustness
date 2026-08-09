import httpx

from facm.clients.http_retry import request_with_retry


def test_request_with_retry_succeeds_after_502(monkeypatch):
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            req = httpx.Request("POST", "http://example/v1/chat/completions")
            resp = httpx.Response(502, request=req)
            raise httpx.HTTPStatusError("502", request=req, response=resp)
        return "ok"

    monkeypatch.setattr("facm.clients.http_retry.time.sleep", lambda _s: None)
    assert request_with_retry(flaky, retries=5, label="test") == "ok"
    assert calls["n"] == 3


def test_request_with_retry_gives_up(monkeypatch):
    def always_fail() -> str:
        req = httpx.Request("POST", "http://example/v1/chat/completions")
        resp = httpx.Response(502, request=req)
        raise httpx.HTTPStatusError("502", request=req, response=resp)

    monkeypatch.setattr("facm.clients.http_retry.time.sleep", lambda _s: None)
    try:
        request_with_retry(always_fail, retries=2, label="test")
        assert False, "expected HTTPStatusError"
    except httpx.HTTPStatusError:
        pass
