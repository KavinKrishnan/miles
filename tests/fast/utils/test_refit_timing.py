import json

import pytest

from miles.utils.refit_timing import refit_span, set_refit_identity, trace_refit


def test_trace_records_nested_failure_and_resets_context(monkeypatch, capsys):
    monkeypatch.setenv("MILES_REFIT_TIMING", "1")

    @trace_refit("round")
    def failing():
        set_refit_identity(rank=1, version_id="v-42", target_training_step=42)
        with refit_span("publish"):
            raise ValueError("failed")

    with pytest.raises(ValueError, match="failed"):
        failing()
    record = json.loads(capsys.readouterr().out.split(" ", 1)[1])
    assert not record["succeeded"]
    assert record["version_id"] == "v-42"
    outer, child = record["spans"]
    assert child["parent"] == outer["id"]
    assert outer["start_ns"] <= child["start_ns"] <= child["end_ns"] <= outer["end_ns"]
    assert child["status"] == "error"
    with refit_span("outside"):
        pass
    assert not capsys.readouterr().out


def test_disabled_trace_preserves_return_and_emits_nothing(monkeypatch, capsys):
    monkeypatch.delenv("MILES_REFIT_TIMING", raising=False)

    @trace_refit("round")
    def operation():
        return 42

    assert operation() == 42
    assert not capsys.readouterr().out
