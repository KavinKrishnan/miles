from __future__ import annotations

import asyncio
import logging
from typing import Any

from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.workers.worker_handle import BaseWorkerHandle

logger = logging.getLogger(__name__)

TRAINER_READY_TIMEOUT_SECONDS = 3600.0


class CompositeTrainerController:
    def __init__(self, *, trainers: dict[str, BaseWorkerHandle]) -> None:
        assert trainers, "a composite trainer controller fans out over at least one trainer"
        self._trainers = dict(trainers)

    @property
    def model_ids(self) -> list[str]:
        return list(self._trainers)

    async def wait_ready(self, *, timeout: float = TRAINER_READY_TIMEOUT_SECONDS) -> None:
        await asyncio.gather(*[trainer.wait_ready(timeout=timeout) for trainer in self._trainers.values()])
        logger.info(f"Every trainer of this run is ready: {sorted(self._trainers)}")

    async def init(self, args, model_id: str | None = None) -> list[Any]:
        return await self._route(model_id).init(args)

    async def train(
        self,
        rollout_id: int,
        rollout_data_pack: dict[str, Any],
        external_data: list[Any] | None = None,
        model_id: str | None = None,
    ) -> list[Any]:
        return await self._route(model_id).train(
            rollout_id=rollout_id, rollout_data_pack=rollout_data_pack, external_data=external_data
        )

    async def save_model(self, rollout_id: int, force_sync: bool = False, model_id: str | None = None) -> None:
        await self._route(model_id).save_model(rollout_id=rollout_id, force_sync=force_sync)

    async def export_hf(self, rollout_id: int, path: str, model_id: str | None = None) -> None:
        await self._route(model_id).export_hf(rollout_id=rollout_id, path=path)

    async def update_weights(self, info, rollout_id: int | None = None, model_id: str | None = None) -> int | None:
        return await self._route(model_id).update_weights(info=info, rollout_id=rollout_id)

    async def onload(self, model_id: str | None = None) -> None:
        await self._fan_out("onload", model_id=model_id)

    async def offload(self, model_id: str | None = None) -> None:
        await self._fan_out("offload", model_id=model_id)

    async def clear_memory(self, model_id: str | None = None) -> None:
        await self._fan_out("clear_memory", model_id=model_id)

    async def reconcile_adapters(self, model_id: str | None = None) -> None:
        await self._fan_out("reconcile_adapters", model_id=model_id)

    async def dispose(self, model_id: str | None = None) -> None:
        await self._fan_out("dispose", model_id=model_id)

    async def get_train_parallel_config(self, model_id: str | None = None) -> dict[str, Any]:
        return await self._route(model_id).get_train_parallel_config()

    async def get_cell_statuses(self, model_id: str | None = None) -> dict[str, CellStatus]:
        if model_id is not None:
            return await self._route(model_id).get_cell_statuses()
        per_trainer = await asyncio.gather(*[trainer.get_cell_statuses() for trainer in self._trainers.values()])
        return {cell_id: status for statuses in per_trainer for cell_id, status in statuses.items()}

    async def _fan_out(self, method_name: str, *, model_id: str | None) -> None:
        if model_id is not None:
            await getattr(self._route(model_id), method_name)()
            return
        await asyncio.gather(*[getattr(trainer, method_name)() for trainer in self._trainers.values()])

    def _route(self, model_id: str | None) -> BaseWorkerHandle:
        if model_id is None:
            assert (
                len(self._trainers) == 1
            ), f"this run trains {sorted(self._trainers)}, so every call has to name the model it drives"
            return next(iter(self._trainers.values()))

        trainer = self._trainers.get(model_id)
        assert (
            trainer is not None
        ), f"no trainer is deployed for model {model_id!r}, known models: {sorted(self._trainers)}"
        return trainer
