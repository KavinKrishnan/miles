import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

_READY_POLL_INTERVAL_SECONDS = 0.05


def start_and_wait_thread(
    *,
    target: Callable[[], None],
    is_ready: Callable[[], bool],
    description: str,
    timeout_seconds: float,
) -> threading.Thread:
    """Run target on a daemon thread and return only once is_ready holds.

    A daemon thread that fails on its own is invisible to its caller, so anything
    the rest of the run depends on must be started through here: the failure is
    re-raised on the calling thread instead of leaving a half-started system."""
    error: list[BaseException] = []

    def _run() -> None:
        try:
            target()
        except BaseException as err:  # noqa: BLE001 - re-raised on the caller thread below
            logger.error("%s died", description, exc_info=True)
            error.append(err)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout_seconds
    while not is_ready():
        if error:
            raise RuntimeError(f"{description} failed during startup") from error[0]
        if not thread.is_alive():
            raise RuntimeError(f"{description} exited during startup")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{description} did not finish startup within {timeout_seconds}s")
        time.sleep(_READY_POLL_INTERVAL_SECONDS)

    logger.info("%s started", description)
    return thread
