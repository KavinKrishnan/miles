import asyncio
import logging
import time
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.eval_fleet import EvalFleet, EvalFleetInfo, EvalFleetPin
from miles.ray.rollout.rollout_server import RolloutServer, create_rollout_servers, dispose_uncommitted_cell
from miles.ray.rollout.router_manager import resolve_router_addrs
from miles.ray.rollout.server_cell import ServerCell, compute_server_cell_meta_from_info
from miles.ray.rollout.updatable_engines import UpdatableEngines
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.context_lock import (
    ContextLock,
    acquires_lock,
    enforce_lock_discipline,
    lock_exempt,
    releases_lock,
    requires_lock,
    with_lock,
)
from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.ft_utils.health_checker import ActivenessTracker
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import NodeProbeMixin, SimpleTicker
from miles.utils.workers.registration.models import (
    SNAPSHOT_STALENESS_WARNING_SECONDS,
    RegistrationAck,
    RegistrationSnapshot,
)
from miles.utils.workers.registration.provider import RegistrationWorkerProvider
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.utils import apply_cell_observation
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)

TICK_INTERVAL_SECONDS = 5.0
CELL_TICK_TIMEOUT_SECONDS = 120.0
CELLS_READY_POLL_INTERVAL_SECONDS = 2.0
CELLS_READY_TIMEOUT_SECONDS = 3600.0


@dataclass
class _CellReconcileSlot:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    revision: int = 0


@enforce_lock_discipline
class InferenceController(NodeProbeMixin):
    @lock_exempt
    def __init__(
        self,
        args,
        *,
        engine_provider: BaseWorkerProvider,
        router_providers: Sequence[BaseWorkerProvider],
        registration_provider: RegistrationWorkerProvider | None = None,
    ) -> None:
        self.args = args
        self._engine_provider = engine_provider
        self._router_providers = router_providers
        self._registration_provider = registration_provider
        self.context_lock = ContextLock("InferenceController")
        self.servers: dict[str, RolloutServer] = {}
        self._eval_fleet: EvalFleet | None = None
        self._router_addrs: dict[str, HostAndPort] = {}
        self._watcher_disposers: list[StopWatchFn] = []
        self._health_checker_activeness = ActivenessTracker(active=True)
        self._ticker: SimpleTicker | None = None
        self._cell_reconcile_slots: dict[str, _CellReconcileSlot] = {}

    @lock_exempt
    async def init(self) -> None:
        configure_logger(self.args, source=SimpleProcessIdentity(component="inference_controller"))

        if self.args.debug_train_only:
            return

        await self._engine_provider.init()
        router_addrs = await resolve_router_addrs(self.args, router_providers=self._router_providers)
        self.servers = await create_rollout_servers(
            self.args,
            context_lock=self.context_lock,
            global_health_checker_activeness=self._health_checker_activeness.get,
            engine_provider=self._engine_provider,
            router_addrs=router_addrs,
        )
        if self.args.eval_num_gpus > 0:
            self._eval_fleet = EvalFleet(self.args, srv=self.servers["eval"])

        self._router_addrs = router_addrs
        self._watcher_disposers.append(await self._engine_provider.watch_cells(self._reconcile))
        self._ticker = SimpleTicker(self._tick_cells, interval_seconds=TICK_INTERVAL_SECONDS)

        dashboard_hooks.register_router(self.args)

        await asyncio.gather(*[srv.wait_expected_num_cells() for srv in self.servers.values()])

    # -------------------------- rollout lifecycle hooks -----------------------------

    @with_lock
    async def prepare_rollout(self, rollout_id: int) -> None:
        await self._health_monitoring_resume()
        await dashboard_hooks.register_engines(self.servers, provider=self._engine_provider)

    @with_lock
    async def prepare_eval(self) -> None:
        await self._health_monitoring_resume()

    @lock_exempt
    async def dispose(self) -> None:
        if (ticker := self._ticker) is not None:
            self._ticker = None
            await _stop_while_tearing_down(ticker.dispose(), what="the cell ticker of this run")

        for disposer in self._watcher_disposers:
            await _stop_while_tearing_down(disposer(), what="a cell watch of this run")
        self._watcher_disposers = []

        await self._dispose_servers()

    @with_lock
    async def _dispose_servers(self) -> None:
        for srv in self.servers.values():
            await srv.dispose()

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    @with_lock
    async def offload(self, tags: list[str] | None = None) -> None:
        await self._health_monitoring_pause()
        for srv in self.servers.values():
            await srv.offload(tags=tags)

    @with_lock
    async def onload(self, tags: list[str] | None = None) -> None:
        await self._onload(tags=tags)

    @with_lock
    async def onload_weights(self) -> None:
        await self._onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    @with_lock
    async def onload_kv(self) -> None:
        await self._onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    @requires_lock
    async def _onload(self, tags: list[str] | None):
        for srv in self.servers.values():
            await srv.onload(tags)

    # -------------------------- engine management -----------------------------

    @lock_exempt
    async def get_router_urls(self) -> dict[str, str]:
        return {model_id: addr.addr for model_id, addr in self._router_addrs.items()}

    @lock_exempt
    async def apply_registration_snapshot(self, snapshot: RegistrationSnapshot) -> RegistrationAck:
        assert self._registration_provider is not None, (
            f"reporter {snapshot.reporter_id} registered engines with this run, which does not expect any; pass "
            f"--expected-registration-reporters to the deployment that runs the inference controller"
        )
        return await self._registration_provider.apply_snapshot(snapshot)

    @lock_exempt
    async def updatable_model_ids(self) -> list[str]:
        return [model_id for model_id, srv in self.servers.items() if srv.update_weights]

    @acquires_lock
    async def start_update_weights(self, model_id: str | None = None) -> UpdatableEngines:
        """Return engines eligible for weight updates."""
        await self._health_monitoring_pause()
        await self._ensure_cells_ready()

        srv = self._get_updatable_server(model_id)
        if not srv:
            return UpdatableEngines(
                model_id=model_id,
                rollout_engines=[],
                engine_gpu_counts=[],
                engine_gpu_offsets=[],
                snapshot_cell_id_to_hashes={},
            )

        return UpdatableEngines(
            model_id=srv.model_name,
            rollout_engines=srv.api_clients,
            engine_gpu_counts=srv.engine_gpu_counts,
            engine_gpu_offsets=srv.engine_gpu_offsets,
            snapshot_cell_id_to_hashes={cell_id: cell.meta.workers_hash for cell_id, cell in srv.server_cells.items()},
        )

    @releases_lock
    async def abort_update_weights(self) -> None:
        await self._health_monitoring_resume()

    @releases_lock
    async def end_update_weights(self, snapshot_cell_id_to_hashes: dict[str, str]) -> None:
        await asyncio.gather(
            *[
                cell.mark_weights_ready()
                for srv in self.servers.values()
                for cell_id, cell in srv.server_cells.items()
                if cell_id in snapshot_cell_id_to_hashes
                and snapshot_cell_id_to_hashes[cell_id] == cell.meta.workers_hash
                and cell.is_pending_weights
            ]
        )

    @requires_lock
    async def _ensure_cells_ready(self) -> None:
        deadline = time.monotonic() + CELLS_READY_TIMEOUT_SECONDS
        while True:
            cells = [cell for srv in self.servers.values() for cell in srv.server_cells.values()]
            if self.args.colocate:
                await asyncio.gather(*[cell.init() for cell in cells if cell.is_uninitialized])
            pending = [cell for cell in cells if not cell.is_pending_weights_or_serving]
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {CELLS_READY_TIMEOUT_SECONDS}s waiting for "
                    f"{len(pending)}/{len(cells)} cells to become ready"
                )
            logger.info(f"Waiting for {len(pending)}/{len(cells)} cells to become ready...")
            async with self.context_lock.with_released():
                await asyncio.sleep(CELLS_READY_POLL_INTERVAL_SECONDS)

    @requires_lock
    def _get_updatable_server(self, model_id: str | None = None) -> RolloutServer | None:
        if model_id is not None:
            srv = self.servers.get(model_id)
            assert srv is not None, f"No server for model_id {model_id!r}, known ids: {sorted(self.servers)}"
            assert srv.update_weights, f"Server for model_id {model_id!r} is frozen (update_weights=False)"
            return srv

        updatable = [srv for srv in self.servers.values() if srv.update_weights]
        match updatable:
            case []:
                return None
            case [srv]:
                return srv
            case _:
                raise ValueError(
                    f"Multiple servers have update_weights=True: {[srv.model_name for srv in updatable]}. "
                    f"Pass model_id to update exactly one of them."
                )

    # -------------------------- eval fleet -----------------------------

    @lock_exempt
    async def get_eval_fleet(self) -> EvalFleetInfo | None:
        return self._eval_fleet.info if self._eval_fleet is not None else None

    @lock_exempt
    async def pin_eval_fleet(self, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        if self._eval_fleet is None:
            return EvalFleetPin(skip_reason="no_fleet")
        return await self._eval_fleet.pin(checkpoint_dir=checkpoint_dir, weight_version=weight_version)

    # -------------------------- misc APIs -----------------------------

    @lock_exempt
    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        return {
            cell_id: cell.cell_status()
            for srv in list(self.servers.values())
            for cell_id, cell in list(srv.server_cells.items())
        }

    @with_lock
    async def check_weights(
        self,
        action: str,
        allow_quant_error: bool = False,
        selector: str = "all",
        skip_list: list[str] | None = None,
        model_id: str | None = None,
    ) -> list[Any]:
        # Only the updatable model is re-synced; a frozen model would always mismatch.
        srv = self._get_updatable_server(model_id)
        if srv is None:
            return []
        return await srv.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    # -------------------------- tick -----------------------------

    @with_lock
    async def _tick_cells(self) -> None:
        cells = [cell for srv in list(self.servers.values()) for cell in list(srv.server_cells.values())]
        results = await asyncio.gather(
            *[asyncio.wait_for(cell.tick(), timeout=CELL_TICK_TIMEOUT_SECONDS) for cell in cells],
            return_exceptions=True,
        )
        for cell, result in zip(cells, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Ticking cell {cell.meta.cell_id} failed", exc_info=result)

        for srv in list(self.servers.values()):
            srv.remove_unreachable_cells()
        self._log_stale_reporters()

    @requires_lock
    def _log_stale_reporters(self) -> None:
        if (provider := self._registration_provider) is None:
            return
        for reporter_id in provider.reporter_ids():
            if (seconds := provider.seconds_since_last_snapshot(reporter_id)) < SNAPSHOT_STALENESS_WARNING_SECONDS:
                continue
            logger.warning(
                f"Reporter {reporter_id} last registered its cells {seconds:.0f}s ago, longer than "
                f"{SNAPSHOT_STALENESS_WARNING_SECONDS:.0f}s. Staleness never removes a cell, so the "
                f"{len(provider.cell_ids())} registered cells stay in this run until its own probe finds them dead"
            )

    # -------------------------- reconcile -----------------------------

    @lock_exempt
    async def _reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        slot = self._cell_reconcile_slots.setdefault(cell_id, _CellReconcileSlot())
        slot.users += 1
        slot.revision += 1
        revision = slot.revision
        try:
            async with slot.lock:
                await self._reconcile_one_observation(cell_id=cell_id, observed=observed, revision=revision)
        finally:
            slot.users -= 1
            if slot.users == 0:
                self._cell_reconcile_slots.pop(cell_id, None)

    @lock_exempt
    async def _reconcile_one_observation(self, *, cell_id: str, observed: CellInfo | None, revision: int) -> None:
        async def _add(_cell_id: str, observed_info: CellInfo) -> None:
            await self._bring_up_cell(cell_id=_cell_id, observed=observed_info, revision=revision)

        await apply_cell_observation(
            cell_id=cell_id,
            observed=observed,
            actual_workers_hash=await self._workers_hash_of_cell(cell_id),
            add=_add,
            remove=self._remove_cell,
        )

    @lock_exempt
    async def _bring_up_cell(self, *, cell_id: str, observed: CellInfo, revision: int) -> None:
        cell_meta = compute_server_cell_meta_from_info(observed)
        srv = self.servers[cell_meta.model_id]
        cell = await srv.bring_up_cell(cell_meta)

        if self._cell_reconcile_slots[cell_id].revision != revision:
            logger.warning(
                f"Cell {cell_id} of {srv.model_name} was observed again while it was being brought up, so this "
                f"bring-up is dropped and the newer observation alone decides what this run holds"
            )
            await dispose_uncommitted_cell(cell)
            return

        if not await self._commit_cell(srv=srv, cell=cell):
            logger.warning(
                f"Cell {cell_id} of {srv.model_name} finished starting up after its server was disposed, so it is "
                f"torn down again instead of joining a run that is already over"
            )
            await dispose_uncommitted_cell(cell)

    @with_lock
    async def _commit_cell(self, *, srv: RolloutServer, cell: ServerCell) -> bool:
        return srv.commit_cell(cell)

    @with_lock
    async def _workers_hash_of_cell(self, cell_id: str) -> str | None:
        for srv in self.servers.values():
            if (cell := srv.server_cells.get(cell_id)) is not None:
                return cell.meta.workers_hash
        return None

    @with_lock
    async def _remove_cell(self, cell_id: str) -> None:
        for srv in self.servers.values():
            if cell_id in srv.server_cells:
                await srv.remove_cell(cell_id)
                return

    # -------------------------- utils -----------------------------

    @requires_lock
    async def _health_monitoring_pause(self) -> None:
        self._health_checker_activeness.bump_active(False)

    @requires_lock
    async def _health_monitoring_resume(self) -> None:
        self._health_checker_activeness.bump_active(True)


async def _stop_while_tearing_down(step: Awaitable[None], *, what: str) -> None:
    try:
        await step
    except Exception:
        logger.error(f"Stopping {what} failed, so this run tears the rest of it down anyway", exc_info=True)
