from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial

from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import (
    BaseWorkerProvider,
    CellInfo,
    ReconcileFn,
    StopWatchFn,
    cell_id_of_worker,
)
from miles.utils.workers.worker_spec import NamedHostAndPorts

logger = logging.getLogger(__name__)


class FanInWorkerProvider(BaseWorkerProvider):
    def __init__(self, *, providers: Sequence[BaseWorkerProvider]) -> None:
        assert providers, "a fan-in provider needs at least one provider to observe"
        self._providers = list(providers)
        self._provider_by_cell: dict[str, BaseWorkerProvider] = {}

    async def init(self) -> None:
        for provider in self._providers:
            await provider.init()

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return await self._provider_of(cell_id_of_worker(worker_name)).get_addrs(worker_name)

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [self._provider_of(cell_id).get_worker_infos(cell_ids=[cell_id])[0] for cell_id in cell_ids]

    async def get_handle_async(self, worker_name: str) -> BaseWorkerHandle:
        return await self._provider_of(cell_id_of_worker(worker_name)).get_handle_async(worker_name)

    def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        return self._provider_of(cell_id_of_worker(worker_name)).get_handle(worker_name)

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        stop_watches: list[StopWatchFn] = []
        for provider in self._providers:
            stop_watches.append(await provider.watch_cells(partial(self._reconcile, provider, reconcile)))

        async def _stop() -> None:
            for stop_watch in stop_watches:
                try:
                    await stop_watch()
                except Exception:
                    logger.error(
                        "Stopping one of the cell watches of this run failed, so the remaining ones are stopped "
                        "anyway rather than left running against a controller that is tearing down",
                        exc_info=True,
                    )

        return _stop

    def invalidate_cell(self, cell_id: str) -> None:
        if (provider := self._provider_by_cell.get(cell_id)) is not None:
            provider.invalidate_cell(cell_id)

    def expected_num_cells(self, *, model_id: str) -> int | None:
        counts = [
            count
            for provider in self._providers
            if (count := provider.expected_num_cells(model_id=model_id)) is not None
        ]
        return sum(counts) if counts else None

    def extra_expected_num_cells(self, *, model_id: str) -> int:
        return sum(provider.extra_expected_num_cells(model_id=model_id) for provider in self._providers)

    async def _reconcile(
        self, provider: BaseWorkerProvider, reconcile: ReconcileFn, cell_id: str, observed: CellInfo | None
    ) -> None:
        previous = self._provider_by_cell.get(cell_id)
        if observed is not None:
            self._provider_by_cell[cell_id] = provider

        try:
            await reconcile(cell_id, observed)
        except BaseException:
            if previous is None:
                self._provider_by_cell.pop(cell_id, None)
            else:
                self._provider_by_cell[cell_id] = previous
            raise

        if observed is None:
            self._provider_by_cell.pop(cell_id, None)

    def _provider_of(self, cell_id: str) -> BaseWorkerProvider:
        provider = self._provider_by_cell.get(cell_id)
        assert provider is not None, (
            f"no provider of this run announced cell {cell_id}, so its addresses are unknown; known cells are "
            f"{sorted(self._provider_by_cell)}"
        )
        return provider
