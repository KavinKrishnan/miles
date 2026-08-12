import threading

import pytest

from miles.utils.thread_utils import start_and_wait_thread


class TestStartAndWaitThread:
    def test_it_returns_once_the_thread_reports_ready(self):
        """The caller may only proceed after the thing it started is actually usable."""
        ready = threading.Event()

        thread = start_and_wait_thread(
            target=lambda: ready.set(),
            is_ready=ready.is_set,
            description="probe",
            timeout_seconds=5.0,
        )

        assert ready.is_set()
        assert isinstance(thread, threading.Thread)

    def test_a_failure_on_the_thread_reaches_the_caller(self):
        """A daemon thread that dies alone is invisible, which is how a lost port went unnoticed."""

        def _boom() -> None:
            raise ValueError("could not start")

        with pytest.raises(RuntimeError, match="probe failed during startup") as excinfo:
            start_and_wait_thread(target=_boom, is_ready=lambda: False, description="probe", timeout_seconds=5.0)

        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_a_thread_that_exits_without_becoming_ready_fails_the_caller(self):
        """Returning quietly is its own failure: nothing is serving afterwards."""
        with pytest.raises(RuntimeError, match="probe exited during startup"):
            start_and_wait_thread(
                target=lambda: None, is_ready=lambda: False, description="probe", timeout_seconds=5.0
            )

    def test_a_thread_that_never_becomes_ready_times_out(self):
        """A wedged startup must not block the caller forever."""
        stop = threading.Event()

        try:
            with pytest.raises(TimeoutError, match="probe did not finish startup"):
                start_and_wait_thread(
                    target=stop.wait, is_ready=lambda: False, description="probe", timeout_seconds=0.2
                )
        finally:
            stop.set()
