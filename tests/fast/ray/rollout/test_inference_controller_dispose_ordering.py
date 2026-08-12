from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.server_cell import ServerCellMetadata
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.workers.registration.models import (
    RegisteredCell,
    RegisteredWorker,
    RegistrationSnapshot,
    compute_snapshot_digest,
)
from miles.utils.workers.registration.provider import RegistrationWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

_REPORTER = "west"
_POOL_ID = "west-inference-engine-0-0"
_CELL_ID = f"{_POOL_ID}-0"


class _RecordingServer:
    def __init__(self) -> None:
        self.server_cells: dict[str, SimpleNamespace] = {}
        self.model_name = "default"
        self.disposed = False

    async def bring_up_cell(self, cell_meta: ServerCellMetadata) -> SimpleNamespace:
        return SimpleNamespace(meta=cell_meta)

    def commit_cell(self, cell: SimpleNamespace) -> bool:
        self.server_cells[cell.meta.cell_id] = cell
        return True

    async def remove_cell(self, cell_id: str) -> None:
        del self.server_cells[cell_id]

    async def dispose(self) -> None:
        self.disposed = True


def _make_controller(*, registration_provider: RegistrationWorkerProvider, server: _RecordingServer):
    controller = InferenceController.__new__(InferenceController)
    controller.args = SimpleNamespace(debug_train_only=False, colocate=False)
    controller.servers = {"default": server}
    controller.context_lock = ContextLock("InferenceController")
    controller._health_checker_activeness = ActivenessTracker(active=True)
    controller._engine_provider = registration_provider
    controller._router_providers = []
    controller._registration_provider = registration_provider
    controller._router_addrs = {}
    controller._watcher_disposers = []
    controller._ticker = None
    controller._cell_reconcile_slots = {}
    return controller


def _snapshot() -> RegistrationSnapshot:
    cells = [
        RegisteredCell(
            cell_id=_CELL_ID,
            pool_id=_POOL_ID,
            workers_hash="hash-1",
            workers=[
                RegisteredWorker(
                    name=f"{_CELL_ID}-0", addrs={"primary": HostAndPort(host="10.9.0.1", port=8000)}, gpu_ids=[0]
                )
            ],
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
    ]
    expected = {"default": 1}
    return RegistrationSnapshot(
        reporter_id=_REPORTER,
        epoch="epoch-1",
        sequence=1,
        digest=compute_snapshot_digest(cells=cells, expected_num_cells_by_model=expected),
        expected_num_cells_by_model=expected,
        cells=cells,
    )


class TestDisposeAgainstAnInFlightDispatch:
    @pytest.mark.asyncio
    async def test_disposing_while_a_registered_cell_reconciles_still_returns(self):
        """One lock order only: taking the context lock and then the provider's would deadlock both forever."""
        provider = RegistrationWorkerProvider(expected_num_reporters=1)
        server = _RecordingServer()
        controller = _make_controller(registration_provider=provider, server=server)
        controller._watcher_disposers.append(await provider.watch_cells(controller._reconcile))
        await provider.apply_snapshot(_snapshot())
        await provider._wait_pending_dispatches()

        holding = asyncio.Event()
        released = asyncio.Event()

        async def _hold_the_context_lock() -> None:
            async with controller.context_lock:
                holding.set()
                await released.wait()

        holder = asyncio.create_task(_hold_the_context_lock())
        await asyncio.wait_for(holding.wait(), timeout=5.0)
        disposing = asyncio.create_task(controller.dispose())
        await asyncio.sleep(0)
        provider.invalidate_cell(_CELL_ID)
        await asyncio.sleep(0)
        released.set()

        await asyncio.wait_for(disposing, timeout=5.0)
        await asyncio.wait_for(holder, timeout=5.0)

        assert server.disposed
