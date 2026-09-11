"""Optional wall-time intervals for Miles refit benchmark attribution."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import os
import socket
import time

_TRACE = contextvars.ContextVar("miles_refit_trace", default=None)


def set_refit_identity(**values):
    trace = _TRACE.get()
    if trace is not None:
        trace["identity"].update(values)


@contextlib.contextmanager
def refit_span(name):
    trace = _TRACE.get()
    if trace is None:
        yield
        return
    started = time.perf_counter_ns()
    span_id = len(trace["spans"])
    span = {"name": name, "id": span_id, "parent": trace["parent"], "status": "ok"}
    trace["spans"].append(span)
    parent = trace["parent"]
    trace["parent"] = span_id
    try:
        yield
    except BaseException:
        span["status"] = "error"
        raise
    finally:
        ended = time.perf_counter_ns()
        span.update(start_ns=started - trace["start_ns"], end_ns=ended - trace["start_ns"])
        trace["parent"] = parent


def trace_refit(name):
    def decorate(operation):
        @functools.wraps(operation)
        def wrapped(*args, **kwargs):
            if os.environ.get("MILES_REFIT_TIMING", "0") != "1":
                return operation(*args, **kwargs)
            existing = _TRACE.get()
            if existing is not None:
                with refit_span(name):
                    return operation(*args, **kwargs)
            started = time.perf_counter_ns()
            trace = {
                "start_ns": started,
                "wall_start_ns": time.time_ns(),
                "parent": None,
                "spans": [],
                "identity": {"pid": os.getpid(), "host": socket.gethostname(), "run_id": os.environ.get("RUN_ID")},
            }
            token = _TRACE.set(trace)
            succeeded = False
            try:
                with refit_span(name):
                    result = operation(*args, **kwargs)
                succeeded = True
                return result
            finally:
                ended = time.perf_counter_ns()
                _TRACE.reset(token)
                payload = {
                    "schema": "miles-refit-spans-v1",
                    **trace["identity"],
                    "wall_start_ns": trace["wall_start_ns"],
                    "duration_ms": (ended - started) / 1e6,
                    "succeeded": succeeded,
                    "spans": trace["spans"],
                }
                print("MILES_REFIT_TIMING " + json.dumps(payload, separators=(",", ":")), flush=True)

        return wrapped

    return decorate
