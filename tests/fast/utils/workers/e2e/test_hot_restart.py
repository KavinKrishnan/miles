from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import pytest
from tests.fast.utils.workers.e2e.harness import READY_TIMEOUT_SECONDS, wait_until_serving

from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.worker_handle import WorkerUnreachableError

_OBSERVE_TIMEOUT_SECONDS = 20.0


async def _counter(server, make_handle, tag: str) -> int:
    return await make_handle(server).report_counter(tag=tag)


async def _wait_until(
    observe: Callable[[], Awaitable[int]], *, expected: int, timeout: float = _OBSERVE_TIMEOUT_SECONDS
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await observe() == expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"never observed {expected} within {timeout}s")


class TestCallSurvivesALostConnection:
    async def test_a_sync_call_whose_submit_response_is_lost_still_runs_to_completion(
        self, server, proxy_to, make_handle, tag
    ):
        """A hot restart kills the caller mid-call, and a train step that stops halfway corrupts the trainer."""
        proxy = await proxy_to()
        proxy.drop_next(1)
        handle = make_handle(proxy)

        with pytest.raises(WorkerUnreachableError):
            await handle.demo_count_after_sleep(tag=tag, seconds=1.0)

        await _wait_until(lambda: _counter(server, make_handle, tag), expected=1)

    async def test_a_sync_call_whose_poll_connection_drops_still_returns_its_result(
        self, server, proxy_to, make_handle, tag
    ):
        """Losing the poll connection must not lose the outcome, or the caller reruns a step that already ran."""
        proxy = await proxy_to()
        handle = make_handle(proxy)
        pending = asyncio.create_task(handle.demo_sleep_sync(tag=tag, seconds=2.0))
        await asyncio.sleep(0.5)
        proxy.drop_next(1)

        assert await pending == tag
        events = await make_handle(server).report_events()
        assert any(event.tag == tag and event.phase == "end" for event in events)

    async def test_the_worker_runs_a_dropped_call_exactly_once(self, server, proxy_to, make_handle, tag):
        """A retried poll must not re-execute the call, or an optimizer step lands twice."""
        proxy = await proxy_to()
        handle = make_handle(proxy)
        pending = asyncio.create_task(handle.demo_count_after_sleep(tag=tag, seconds=2.0))
        await asyncio.sleep(0.5)
        proxy.drop_next(1)
        await pending

        assert await make_handle(server).report_counter(tag=tag) == 1


class TestBootUuidAcrossARealRestart:
    async def test_a_pinned_client_refuses_a_replacement_process(self, spawn, make_handle):
        """The whole point of pinning: a new process must not be driven by a script that never initialized it."""
        server = spawn()
        handle = make_handle(server, require_stable_boot_uuid=True)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
        assert await handle.demo_sync(a=1, b=1) == 2

        server.stop()
        replacement = spawn(port=server.port)

        with pytest.raises(ServerRestartedError):
            await handle.demo_sync(a=1, b=1)
        assert replacement.is_running()

    async def test_wait_ready_takes_the_replacement_process_over(self, spawn, make_handle):
        """A restart is expected while waiting for readiness, so the client re-baselines onto what answers."""
        server = spawn()
        handle = make_handle(server, require_stable_boot_uuid=True)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
        assert await handle.demo_sync(a=1, b=1) == 2

        server.stop()
        replacement = spawn(port=server.port, wait=False)
        wait_until_serving(replacement)

        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
        assert await handle.demo_sync(a=2, b=2) == 4

    async def test_a_restart_after_the_re_baseline_is_refused_again(self, spawn, make_handle):
        """Re-baselining widens the tolerated window to wait_ready, it does not disarm the check."""
        server = spawn()
        handle = make_handle(server, require_stable_boot_uuid=True)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)

        server.stop()
        second = spawn(port=server.port)
        await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)
        assert await handle.demo_sync(a=1, b=1) == 2

        second.stop()
        spawn(port=server.port)
        with pytest.raises(ServerRestartedError):
            await handle.demo_sync(a=1, b=1)


class TestWaitIdleAgainstARealServer:
    async def test_a_busy_worker_is_reported_until_its_call_ends(self, server, make_handle, tag):
        """This is the step a restarted orchestration script uses to wait out a running train step."""
        handle = make_handle(server)
        pending = asyncio.create_task(handle.demo_block_sync(tag=tag))
        await asyncio.sleep(0.5)

        assert len(await handle.get_in_flight_call_ids()) >= 1

        await make_handle(server).release(tag=tag)
        await pending
        await handle.wait_idle(timeout=30.0)
