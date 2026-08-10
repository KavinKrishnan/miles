import time

import pytest
import requests


def _modelexpress_payload():
    return {
        "target_training_step": 9,
        "logical_group": "model",
    }


def test_flush_cache_sleeps_between_pending_request_retries(monkeypatch):
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second — nowhere near enough time for
    in-flight generation to drain — and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "fake-host"
    engine.server_port = 1234

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(requests, "get", lambda url: type("Resp", (), {"status_code": 400})())

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        engine.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )


def test_modelexpress_endpoint_forwards_explicit_payload_and_timeout(monkeypatch):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    calls = []
    monkeypatch.setattr(
        engine,
        "_make_request",
        lambda endpoint, payload, timeout=None: calls.append((endpoint, payload, timeout)) or {"success": True},
    )

    payload = _modelexpress_payload()
    assert engine.update_weights_from_modelexpress(payload=payload, timeout=12.5) == {"success": True}
    assert calls == [("update_weights_from_modelexpress", payload, 12.5)]


def test_modelexpress_endpoint_accepts_optional_layout_signature(monkeypatch):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    calls = []
    monkeypatch.setattr(
        engine,
        "_make_request",
        lambda endpoint, payload, timeout=None: calls.append(payload)
        or {
            "success": True,
            "target_training_step": payload["target_training_step"],
            "installed_training_step": payload["target_training_step"],
            "layout_signature": payload["expected_layout_signature"],
            "metrics": {},
            "timing": {},
            "error": None,
            "receiver_poisoned": False,
        },
    )
    payload = _modelexpress_payload() | {"expected_layout_signature": "layout-123"}

    result = engine.update_weights_from_modelexpress(payload=payload)

    assert calls == [payload]
    assert result["installed_training_step"] == 9
    assert result["layout_signature"] == "layout-123"


def test_modelexpress_endpoint_rejects_orchestrator_only_fields():
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    payload = _modelexpress_payload() | {"cohort_id": "not-supported-upstream"}
    with pytest.raises(ValueError, match="Unsupported ModelExpress SGLang request fields"):
        engine.update_weights_from_modelexpress(payload=payload)


def test_modelexpress_endpoint_absence_fails_without_fallback(monkeypatch):
    pytest.importorskip("sglang")
    from miles.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    response = requests.Response()
    response.status_code = 404
    error = requests.exceptions.HTTPError(response=response)
    monkeypatch.setattr(engine, "_make_request", lambda *args, **kwargs: (_ for _ in ()).throw(error))

    with pytest.raises(RuntimeError, match="will not fall back"):
        engine.update_weights_from_modelexpress(payload=_modelexpress_payload())
