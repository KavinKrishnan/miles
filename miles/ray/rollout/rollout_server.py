import asyncio
import dataclasses
import logging
from collections.abc import Callable
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock, enforce_lock_discipline, lock_exempt, requires_lock
from miles.utils.ft_utils.health_checker import ActiveAndEpoch
from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)

WAIT_CELLS_INITIAL_DELAY_SECONDS = 1.0
WAIT_CELLS_MAX_DELAY_SECONDS = 5.0


async def create_rollout_servers(
    args,
    context_lock: ContextLock,
    global_health_checker_activeness: Callable[[], ActiveAndEpoch],
    *,
    engine_provider: BaseWorkerProvider,
    router_addrs: dict[str, HostAndPort],
) -> dict[str, "RolloutServer"]:
    """Create rollout servers: one per model, each with its own router."""
    config = resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}

    for model_cfg in config.models:
        router_addr = router_addrs[model_cfg.name]

        servers[model_cfg.name] = RolloutServer(
            server_cells={},
            args=args,
            context_lock=context_lock,
            engine_provider=engine_provider,
            router_ip=router_addr.host,
            router_port=router_addr.port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
            global_health_checker_activeness=global_health_checker_activeness,
            expected_num_cells=_compute_expected_num_cells(engine_provider, model_cfg=model_cfg),
        )

    return servers


def _compute_expected_num_cells(engine_provider: BaseWorkerProvider, *, model_cfg) -> int:
    if (n := engine_provider.expected_num_cells(model_id=model_cfg.name)) is not None:
        return n
    return sum(
        group_cfg.num_gpus // group_cfg.num_gpus_per_engine
        for group_cfg in model_cfg.server_groups
        if group_cfg.worker_type != "placeholder"
    )


@dataclasses.dataclass
@enforce_lock_discipline
class RolloutServer:
    """A model served behind a shared router, as a dict of cell id -> cell.

    Each RolloutServer represents one model deployed behind a single router.
    """

    server_cells: dict[str, ServerCell]
    args: Any
    context_lock: ContextLock
    engine_provider: BaseWorkerProvider
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True
    global_health_checker_activeness: Callable[[], ActiveAndEpoch] = lock_exempt(
        lambda: ActiveAndEpoch(active=True, epoch=0)
    )
    expected_num_cells: int = 0
    _disposed: bool = dataclasses.field(default=False, init=False)

    @property
    @requires_lock
    def api_clients(self) -> list[SGLangApiClient]:
        """One client per cell, talking to its primary (node-0) engine."""
        return [cell.api_client for cell in self._cells_by_gpu_offset()]

    @property
    @requires_lock
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [cell.meta.num_gpus_per_engine for cell in self._cells_by_gpu_offset()]

    @property
    @requires_lock
    def engine_gpu_offsets(self) -> list[int]:
        return [cell.meta.gpu_offset for cell in self._cells_by_gpu_offset()]

    @requires_lock
    def _cells_by_gpu_offset(self) -> list[ServerCell]:
        return sorted(self.server_cells.values(), key=lambda cell: cell.meta.gpu_offset)

    @lock_exempt
    async def probe_and_mark_dead(self):
        """Mark unreachable cells stopped so ``recover`` restarts them.

        For servers without a ``RolloutHealthMonitor``, which does the same job.
        """
        for cell in self.server_cells.values():
            await cell.probe_and_mark_dead()

    @requires_lock
    def remove_unreachable_cells(self) -> None:
        for cell_id, cell in list(self.server_cells.items()):
            if not cell.is_unreachable:
                continue
            logger.warning(
                f"Cell {cell_id} of {self.model_name} failed its health checks, so its provider is told to forget "
                f"it; it comes back as soon as that provider observes it again"
            )
            self.engine_provider.invalidate_cell(cell_id)

    @lock_exempt
    async def bring_up_cell(self, cell_meta: ServerCellMetadata) -> ServerCell:
        cell = ServerCell(
            args=self.args,
            router_api_client=self._router_api_client,
            meta=cell_meta,
            provider=self.engine_provider,
            global_health_checker_activeness=self.global_health_checker_activeness,
        )
        if self.args.colocate:
            return cell
        try:
            await cell.init()
        except BaseException:
            logger.error(
                f"Cell {cell_meta.cell_id} of {self.model_name} did not initialize, so it never joins this run; its "
                f"provider rolls it back too and announces it once more",
                exc_info=True,
            )
            await dispose_uncommitted_cell(cell)
            raise
        return cell

    @requires_lock
    def commit_cell(self, cell: ServerCell) -> bool:
        cell_id = cell.meta.cell_id
        assert cell_id not in self.server_cells, (
            f"cell {cell_id} of {self.model_name} is already in this run, and installing a second one would leave "
            f"the engine the first one drives running with nothing tracking it"
        )
        if self._disposed:
            return False
        self.server_cells[cell_id] = cell
        return True

    @requires_lock
    async def remove_cell(self, cell_id: str):
        logger.info(f"Killing server {cell_id=}...")
        await self.server_cells[cell_id].dispose()
        del self.server_cells[cell_id]

    @requires_lock
    async def dispose(self) -> None:
        self._disposed = True
        for cell_id in list(self.server_cells.keys()):
            await self.remove_cell(cell_id)

    @requires_lock
    async def offload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.offload(tags=tags) for cell in self._addressable_cells() if cell.meta.needs_offload]
        )

    @requires_lock
    async def onload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.onload(tags=tags) for cell in self._addressable_cells() if cell.meta.needs_offload]
        )

    @requires_lock
    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        return await asyncio.gather(
            *[
                cell.check_weights(
                    action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
                )
                for cell in self._addressable_cells()
            ]
        )

    @requires_lock
    def _addressable_cells(self) -> list[ServerCell]:
        return [cell for cell in self.server_cells.values() if cell.is_pending_weights_or_serving]

    @lock_exempt
    async def wait_expected_num_cells(self, timeout: float = 3600):
        async def _check(remaining_seconds: float) -> None:
            expected = self.total_expected_num_cells()
            count = self._count_startable_cells()
            if count < expected:
                raise Exception(f"Only {count}/{expected} cells of {self.model_name} are ready")

        await retry_until_deadline(
            _check,
            total_seconds=timeout,
            retry_on=Exception,
            initial_delay=WAIT_CELLS_INITIAL_DELAY_SECONDS,
            max_delay=WAIT_CELLS_MAX_DELAY_SECONDS,
            log_fields=dict(op="wait_expected_num_cells", model_name=self.model_name),
        )

    @lock_exempt
    def total_expected_num_cells(self) -> int:
        """The cells this deployment launches itself plus the ones other deployments register into it."""
        return self.expected_num_cells + self.engine_provider.extra_expected_num_cells(model_id=self.model_name)

    @lock_exempt
    def _count_startable_cells(self) -> int:
        if self.args.colocate:
            return len(self.server_cells)
        return sum(1 for cell in self.server_cells.values() if cell.is_pending_weights_or_serving)

    @property
    @lock_exempt
    def _router_api_client(self) -> SGLangRouterApiClient:
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")


async def dispose_uncommitted_cell(cell: ServerCell) -> None:
    try:
        await cell.dispose()
    except Exception:
        logger.error(
            f"Disposing cell {cell.meta.cell_id}, which never joined this run, failed too, so its health checker "
            f"may keep probing an engine nothing here drives",
            exc_info=True,
        )
