from __future__ import annotations

import asyncio

import pytest

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_provider import simple
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider, parse_host_and_port

_POOL_ID = "trainer-controller-actor"


class FakeController:
    def health(self) -> int:
        return 1


@pytest.fixture(autouse=True)
def dialled(monkeypatch) -> list[tuple[str, int]]:
    seen: list[tuple[str, int]] = []
    monkeypatch.setattr(simple, "wait_tcp_ready", lambda host, port, timeout: seen.append((host, port)))
    return seen


def _provider(*urls: str) -> SimpleWorkerProvider:
    return SimpleWorkerProvider.of_rpc_urls(
        pool_id=_POOL_ID, urls=list(urls or ("10.0.0.1:8000",)), worker_class=f"{__name__}.FakeController"
    )


class TestAddresses:
    def test_answers_the_address_it_was_given(self):
        """The whole point of a static provider is that nothing discovers this address at runtime."""
        addrs = asyncio.run(_provider().get_addrs(f"{_POOL_ID}-0-0"))

        assert addrs["rpc"].addr == "http://10.0.0.1:8000"

    def test_addresses_each_instance_by_its_own_entry(self):
        """A composite controller will be given several urls, and cell one is not cell zero."""
        provider = _provider("10.0.0.1:8000", "10.0.0.2:9000")

        assert asyncio.run(provider.get_addrs(f"{_POOL_ID}-1-0")).get("rpc").addr == "http://10.0.0.2:9000"

    def test_refuses_a_worker_nobody_named(self):
        """Answering an address for an instance that was never given would invent a host out of thin air."""
        with pytest.raises(AssertionError, match="statically given"):
            asyncio.run(_provider().get_addrs(f"{_POOL_ID}-3-0"))

    def test_refuses_a_worker_of_another_pool(self):
        """One provider answers for one pool, so a stray name must not be answered with this pool's address."""
        with pytest.raises(AssertionError, match="answers for pool"):
            asyncio.run(_provider().get_addrs("inference-controller-0-0"))

    def test_refuses_a_second_worker_in_a_cell(self):
        """A statically addressed controller is one process, so worker one of its cell does not exist."""
        with pytest.raises(AssertionError, match="exactly one worker"):
            asyncio.run(_provider().get_addrs(f"{_POOL_ID}-0-1"))


class TestHandles:
    def test_builds_an_rpc_handle_on_the_given_address(self):
        """The orchestration script reaches an independently deployed controller only over rpc."""
        handle = _provider().get_handle(f"{_POOL_ID}-0-0")

        assert isinstance(handle, RpcWorkerHandle)

    def test_a_handle_needs_the_worker_class_the_wire_types_come_from(self):
        """Without the class the client cannot know a single method signature, so it must say so."""
        provider = SimpleWorkerProvider(pool_id=_POOL_ID, addrs=[{"rpc": parse_host_and_port("10.0.0.1:8000")}])

        with pytest.raises(AssertionError, match="no worker class"):
            provider.get_handle(f"{_POOL_ID}-0-0")

    def test_reports_one_worker_per_given_cell(self):
        """Callers reading a cell expect the same shape a watching provider reports."""
        (infos,) = _provider().get_worker_infos(cell_ids=[f"{_POOL_ID}-0"])

        assert [info.name for info in infos] == [f"{_POOL_ID}-0-0"]
        assert infos[0].worker_class == f"{__name__}.FakeController"


class TestWatch:
    @staticmethod
    def _observed(provider: SimpleWorkerProvider) -> list[tuple[str, CellInfo | None]]:
        seen: list[tuple[str, CellInfo | None]] = []

        async def _reconcile(cell_id: str, info: CellInfo | None) -> None:
            seen.append((cell_id, info))

        asyncio.run(provider.watch_cells(_reconcile))
        return seen

    def test_reports_every_given_cell_once_and_then_stops(self):
        """Static addresses never change, so a watch that kept polling would only burn a task."""
        seen = self._observed(_provider("10.0.0.1:8000", "10.0.0.2:9000"))

        assert [cell_id for cell_id, _info in seen] == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]
        assert all(info.alive for _cell_id, info in seen)

    def test_gives_each_cell_the_index_its_readers_key_on(self):
        """A trainer controller builds its cell from meta['cell_index'], and cell one is not cell zero."""
        seen = self._observed(_provider("10.0.0.1:8000", "10.0.0.2:9000"))

        assert [info.meta["cell_index"] for _cell_id, info in seen] == [0, 1]


class TestWaitReady:
    def test_waits_for_every_address_it_was_given_before_handing_out_a_handle(self, dialled):
        """helm install returns long before a pod answers, so using an address unchecked is a startup race."""
        _provider("10.0.0.1:8000", "10.0.0.2:9000").get_handle(f"{_POOL_ID}-0-0")

        assert dialled == [("10.0.0.1", 8000), ("10.0.0.2", 9000)]

    def test_waits_once_however_many_handles_are_asked_for(self, dialled):
        """The wait answers a startup race, so repeating it would dial on every later handle lookup."""
        provider = _provider("10.0.0.1:8000")

        provider.get_handle(f"{_POOL_ID}-0-0")
        provider.get_handle(f"{_POOL_ID}-0-0")

        assert dialled == [("10.0.0.1", 8000)]

    def test_lets_the_wait_fail_loud_when_nobody_ever_answers(self, monkeypatch):
        """A silent skip would push the failure into the first rpc call, which reports something else entirely."""
        monkeypatch.setattr(
            simple,
            "wait_tcp_ready",
            lambda host, port, timeout: (_ for _ in ()).throw(RuntimeError(f"{host}:{port} not ready")),
        )

        with pytest.raises(RuntimeError, match="10.0.0.1:8000 not ready"):
            _provider().get_handle(f"{_POOL_ID}-0-0")


class TestParseHostAndPort:
    @pytest.mark.parametrize(
        ("addr", "expected"),
        [
            ("10.0.0.1:8000", ("10.0.0.1", 8000)),
            ("http://10.0.0.1:8000", ("10.0.0.1", 8000)),
            ("http://host.namespace.svc:8000/", ("host.namespace.svc", 8000)),
            ("http://[::1]:8000", ("::1", 8000)),
        ],
    )
    def test_reads_the_forms_a_user_writes_on_the_command_line(self, addr, expected):
        """Every one of these is what somebody copies out of a log line into the next launch."""
        parsed = parse_host_and_port(addr)

        assert (parsed.host, parsed.port) == expected

    def test_refuses_an_address_without_a_port(self):
        """A controller is reached at a port, and guessing one would fail much later and much less clearly."""
        with pytest.raises(AssertionError, match="host:port"):
            parse_host_and_port("10.0.0.1")
