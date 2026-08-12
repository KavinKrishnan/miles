from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import NamedHostAndPorts

_POOL_ID = "west-inference-engine-0-0"
_CELL_ID = f"{_POOL_ID}-0"


class _StubProvider:
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise AssertionError(f"cell startup is patched out in this module ({worker_name=})")

    def invalidate_cell(self, cell_id: str) -> None:
        raise AssertionError(f"nothing invalidates cells in this module ({cell_id=})")


def _make_cell_info(*, workers_hash: str = "hash-1") -> CellInfo:
    return CellInfo(
        cell_id=_CELL_ID,
        pool_id=_POOL_ID,
        alive=True,
        worker_names=[f"{_CELL_ID}-0"],
        workers_hash=workers_hash,
        meta=dict(
            model_id="default",
            worker_type="regular",
            num_gpus_per_engine=1,
            gpu_offset=0,
            sglang_api_key=None,
            needs_offload=False,
            update_weights=True,
        ),
    )


def _make_server() -> RolloutServer:
    return RolloutServer(
        server_cells={},
        args=SimpleNamespace(colocate=False, ft_components=[]),
        context_lock=ContextLock("InferenceController"),
        engine_provider=_StubProvider(),
    )


def _make_controller(*, server: RolloutServer) -> InferenceController:
    controller = InferenceController.__new__(InferenceController)
    controller.args = SimpleNamespace(debug_train_only=False, colocate=False)
    controller.servers = {"default": server}
    controller.context_lock = server.context_lock
    controller._health_checker_activeness = ActivenessTracker(active=True)
    controller._engine_provider = server.engine_provider
    controller._router_providers = []
    controller._registration_provider = None
    controller._router_addrs = {}
    controller._watcher_disposers = []
    controller._ticker = None
    controller._cell_reconcile_slots = {}
    return controller


class _Gate:
    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.opened = asyncio.Event()
        self.cells_that_dialled: list[str] = []


def _install_gate(monkeypatch) -> _Gate:
    gate = _Gate()

    async def _wait_at_the_gate(self: ServerCell) -> None:
        gate.cells_that_dialled.append(self.meta.cell_id)
        gate.reached.set()
        await gate.opened.wait()

    monkeypatch.setattr(ServerCell, "init", _wait_at_the_gate)
    return gate


async def _dispose(server: RolloutServer) -> None:
    async with server.context_lock:
        await server.dispose()


class TestTheContextLockDuringACellBringUp:
    @pytest.mark.asyncio
    async def test_the_context_lock_is_free_while_a_cell_waits_at_its_launch_gate(self, monkeypatch):
        """A gate that never answers must not freeze weight updates, ft sweeps and every other cell of this run."""
        gate = _install_gate(monkeypatch)
        srv = _make_server()
        controller = _make_controller(server=srv)

        reconciling = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.wait_for(gate.reached.wait(), timeout=5.0)

        await asyncio.wait_for(controller.prepare_eval(), timeout=5.0)
        assert not controller.context_lock.locked

        gate.opened.set()
        await asyncio.wait_for(reconciling, timeout=5.0)
        assert list(srv.server_cells) == [_CELL_ID]
        await _dispose(srv)

    @pytest.mark.asyncio
    async def test_a_cell_reaches_this_run_only_after_its_gate_answered(self, monkeypatch):
        """Committing before the gate answers would put an engine that is not up yet in front of the router."""
        gate = _install_gate(monkeypatch)
        srv = _make_server()
        controller = _make_controller(server=srv)

        reconciling = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.wait_for(gate.reached.wait(), timeout=5.0)
        assert srv.server_cells == {}

        gate.opened.set()
        await asyncio.wait_for(reconciling, timeout=5.0)

        assert list(srv.server_cells) == [_CELL_ID]
        await _dispose(srv)


class TestConcurrentObservationsOfOneCell:
    @pytest.mark.asyncio
    async def test_two_observations_of_one_cell_install_it_once(self, monkeypatch):
        """Two providers announcing one cell at once would leave an engine running that nothing here tracks."""
        gate = _install_gate(monkeypatch)
        srv = _make_server()
        controller = _make_controller(server=srv)

        first = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.wait_for(gate.reached.wait(), timeout=5.0)
        second = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.sleep(0)
        gate.opened.set()

        await asyncio.wait_for(asyncio.gather(first, second), timeout=5.0)

        assert list(srv.server_cells) == [_CELL_ID]
        assert gate.cells_that_dialled == [_CELL_ID, _CELL_ID]
        await _dispose(srv)

    @pytest.mark.asyncio
    async def test_a_removal_that_arrives_during_a_bring_up_wins(self, monkeypatch):
        """The bring-up started before the news that the cell is gone, so its commit must notice and stand down."""
        gate = _install_gate(monkeypatch)
        srv = _make_server()
        controller = _make_controller(server=srv)

        bringing_up = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.wait_for(gate.reached.wait(), timeout=5.0)
        removing = asyncio.create_task(controller._reconcile(_CELL_ID, None))
        await asyncio.sleep(0)
        gate.opened.set()

        await asyncio.wait_for(asyncio.gather(bringing_up, removing), timeout=5.0)

        assert srv.server_cells == {}


class TestABringUpThatOutlivesItsRun:
    @pytest.mark.asyncio
    async def test_a_cell_that_finished_starting_after_dispose_never_joins(self, monkeypatch):
        """Installing into a disposed server leaks a health checker into a run that is already over."""
        gate = _install_gate(monkeypatch)
        srv = _make_server()
        controller = _make_controller(server=srv)

        reconciling = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.wait_for(gate.reached.wait(), timeout=5.0)
        await asyncio.wait_for(controller.dispose(), timeout=5.0)
        gate.opened.set()

        await asyncio.wait_for(reconciling, timeout=5.0)

        assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_a_watcher_that_refuses_to_stop_does_not_keep_the_servers_alive(self):
        """Every step of teardown has to run, or a cell's health checker probes an engine forever."""
        srv = _make_server()
        controller = _make_controller(server=srv)

        async def _refuse_to_stop() -> None:
            raise RuntimeError("injected stop failure")

        controller._watcher_disposers.append(_refuse_to_stop)

        await asyncio.wait_for(controller.dispose(), timeout=5.0)

        assert srv._disposed
