from __future__ import annotations

import pytest

from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.fan_in import FanInWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts


class _FakeProvider(BaseWorkerProvider):
    def __init__(
        self,
        *,
        pool_id: str,
        cell_indices: list[int],
        expected_extra: int = 0,
        expected_own: int | None = None,
    ) -> None:
        self.pool_id = pool_id
        self.cell_indices = list(cell_indices)
        self.invalidated: list[str] = []
        self.stopped = False
        self.initialized = False
        self._expected_extra = expected_extra
        self._expected_own = expected_own
        self._reconcile: ReconcileFn | None = None

    async def init(self) -> None:
        self.initialized = True

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return {"primary": HostAndPort(host=self.pool_id, port=8000)}

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [
            [
                WorkerInfo(
                    name=f"{cell_id}-0",
                    generation=0,
                    self_addrs={"primary": HostAndPort(host=self.pool_id, port=8000)},
                    gpu_ids=[],
                    handle=None,
                    worker_class=None,
                )
            ]
            for cell_id in cell_ids
        ]

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        self._reconcile = reconcile
        for cell_index in self.cell_indices:
            await reconcile(f"{self.pool_id}-{cell_index}", self._cell_info(cell_index))

        async def _stop() -> None:
            self.stopped = True

        return _stop

    async def drop(self, cell_index: int) -> None:
        await self._reconcile(f"{self.pool_id}-{cell_index}", None)

    def invalidate_cell(self, cell_id: str) -> None:
        self.invalidated.append(cell_id)

    def expected_num_cells(self, *, model_id: str) -> int | None:
        return self._expected_own

    def extra_expected_num_cells(self, *, model_id: str) -> int:
        return self._expected_extra

    def _cell_info(self, cell_index: int) -> CellInfo:
        return CellInfo(
            cell_id=f"{self.pool_id}-{cell_index}",
            pool_id=self.pool_id,
            alive=True,
            worker_names=[f"{self.pool_id}-{cell_index}-0"],
            workers_hash="hash-1",
            meta={},
        )


class _Watcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, CellInfo | None]] = []

    async def __call__(self, cell_id: str, observed: CellInfo | None) -> None:
        self.calls.append((cell_id, observed))


class TestWatchCells:
    async def test_the_watcher_sees_the_cells_of_every_provider(self):
        """Local engines and registered ones must reach the controller through one watch."""
        local = _FakeProvider(pool_id="local", cell_indices=[0])
        remote = _FakeProvider(pool_id="remote", cell_indices=[0, 1])
        watcher = _Watcher()

        await FanInWorkerProvider(providers=[local, remote]).watch_cells(watcher)

        assert [cell_id for cell_id, _ in watcher.calls] == ["local-0", "remote-0", "remote-1"]

    async def test_stopping_the_watch_stops_every_provider(self):
        """A half-stopped watch would keep announcing cells into a disposed controller."""
        local = _FakeProvider(pool_id="local", cell_indices=[])
        remote = _FakeProvider(pool_id="remote", cell_indices=[])

        stop_watch = await FanInWorkerProvider(providers=[local, remote]).watch_cells(_Watcher())
        await stop_watch()

        assert local.stopped and remote.stopped


class TestRoutingByAnnouncedCell:
    async def test_addresses_come_from_the_provider_that_announced_the_cell(self):
        """A registered cell is addressed by what its reporter gave, not by the local platform."""
        provider = FanInWorkerProvider(
            providers=[
                _FakeProvider(pool_id="local", cell_indices=[0]),
                _FakeProvider(pool_id="remote", cell_indices=[0]),
            ]
        )
        await provider.watch_cells(_Watcher())

        assert (await provider.get_addrs("remote-0-0"))["primary"].host == "remote"
        assert (await provider.get_addrs("local-0-0"))["primary"].host == "local"

    async def test_worker_infos_keep_the_order_they_were_asked_in(self):
        """The dashboard zips worker infos against its own cell list."""
        provider = FanInWorkerProvider(
            providers=[
                _FakeProvider(pool_id="local", cell_indices=[0]),
                _FakeProvider(pool_id="remote", cell_indices=[0]),
            ]
        )
        await provider.watch_cells(_Watcher())

        infos = provider.get_worker_infos(cell_ids=["remote-0", "local-0"])

        assert [info.name for (info,) in infos] == ["remote-0-0", "local-0-0"]

    async def test_invalidating_a_cell_reaches_the_provider_that_owns_it(self):
        """Only the owner can make its watch announce the cell again."""
        local = _FakeProvider(pool_id="local", cell_indices=[0])
        remote = _FakeProvider(pool_id="remote", cell_indices=[0])
        provider = FanInWorkerProvider(providers=[local, remote])
        await provider.watch_cells(_Watcher())

        provider.invalidate_cell("remote-0")

        assert remote.invalidated == ["remote-0"] and local.invalidated == []

    async def test_a_cell_nobody_announced_is_a_loud_error(self):
        """Silently answering the wrong provider would address another datacenter's engine."""
        provider = FanInWorkerProvider(providers=[_FakeProvider(pool_id="local", cell_indices=[])])
        await provider.watch_cells(_Watcher())

        with pytest.raises(AssertionError, match="no provider"):
            await provider.get_addrs("local-0-0")

    async def test_a_dropped_cell_is_forgotten_after_it_is_reconciled(self):
        """The removal path still needs the cell's provider while it is being torn down."""
        remote = _FakeProvider(pool_id="remote", cell_indices=[0])
        provider = FanInWorkerProvider(providers=[remote])
        await provider.watch_cells(_Watcher())

        await remote.drop(0)

        with pytest.raises(AssertionError, match="no provider"):
            await provider.get_addrs("remote-0-0")


class TestReconcileFailures:
    async def test_a_cell_whose_reconcile_raised_leaves_no_mapping_behind(self):
        """A mapping to a provider that dropped the cell answers addresses for an engine nobody holds."""
        remote = _FakeProvider(pool_id="remote", cell_indices=[0])
        provider = FanInWorkerProvider(providers=[remote])

        with pytest.raises(RuntimeError):
            await provider.watch_cells(_FailingWatcher())

        with pytest.raises(AssertionError, match="no provider"):
            await provider.get_addrs("remote-0-0")

    async def test_a_failed_update_keeps_the_mapping_the_cell_already_had(self):
        """The cell is still there and still addressable; only its update failed."""
        remote = _FakeProvider(pool_id="remote", cell_indices=[0])
        provider = FanInWorkerProvider(providers=[remote])
        watcher = _FailingWatcher(fail_after=1)
        await provider.watch_cells(watcher)

        with pytest.raises(RuntimeError):
            await remote._reconcile("remote-0", remote._cell_info(0))

        assert (await provider.get_addrs("remote-0-0"))["primary"].host == "remote"


class _FailingWatcher:
    def __init__(self, *, fail_after: int = 0) -> None:
        self._remaining = fail_after

    async def __call__(self, cell_id: str, observed: CellInfo | None) -> None:
        if self._remaining > 0:
            self._remaining -= 1
            return
        raise RuntimeError(f"reconciling {cell_id} failed")


class TestInit:
    async def test_every_provider_is_initialized(self):
        """A provider that discovers its fleet in init() answers nothing until it has run."""
        providers = [
            _FakeProvider(pool_id="local", cell_indices=[]),
            _FakeProvider(pool_id="remote", cell_indices=[]),
        ]

        await FanInWorkerProvider(providers=providers).init()

        assert [provider.initialized for provider in providers] == [True, True]


class TestExpectedNumCells:
    async def test_the_expectations_of_every_provider_are_summed(self):
        """The startup barrier covers the local engines and the registered ones together."""
        provider = FanInWorkerProvider(
            providers=[
                _FakeProvider(pool_id="local", cell_indices=[], expected_extra=0),
                _FakeProvider(pool_id="remote", cell_indices=[], expected_extra=6),
            ]
        )

        assert provider.extra_expected_num_cells(model_id="default") == 6

    def test_a_provider_that_was_handed_a_fleet_still_sizes_it(self):
        """A fan-in over a handed fleet must not fall back onto the gpu formula, which describes another layout."""
        provider = FanInWorkerProvider(
            providers=[
                _FakeProvider(pool_id="static", cell_indices=[], expected_own=3),
                _FakeProvider(pool_id="remote", cell_indices=[], expected_extra=6),
            ]
        )

        assert provider.expected_num_cells(model_id="default") == 3
        assert provider.extra_expected_num_cells(model_id="default") == 6

    def test_without_an_opinion_the_gpu_formula_still_decides(self):
        """Providers that lay out no fleet of their own leave the sizing where it was."""
        provider = FanInWorkerProvider(
            providers=[
                _FakeProvider(pool_id="local", cell_indices=[], expected_extra=0),
                _FakeProvider(pool_id="remote", cell_indices=[], expected_extra=6),
            ]
        )

        assert provider.expected_num_cells(model_id="default") is None
