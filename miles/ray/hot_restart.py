from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, Protocol, TypeVar

from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.rpc.client.misc import RETRYABLE_ERRORS, ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError

logger = logging.getLogger(__name__)

TRAINER_IDLE_TIMEOUT_SECONDS = 3600.0
EXECUTOR_FREE_TIMEOUT_SECONDS = 1800.0
INFERENCE_TAKE_OVER_TIMEOUT_SECONDS = 300.0

_EXECUTOR_POLL_INTERVAL_SECONDS = 5.0

_T = TypeVar("_T")


class ResumableTrainer(Protocol):
    async def is_initialized(self, model_id: str | None = None) -> bool: ...

    async def init(self, args, model_id: str | None = None) -> list[Any]: ...

    async def load_state(self, model_id: str | None = None) -> list[Any]: ...

    async def wait_idle(self, *, timeout: float) -> None: ...


class _StillInitializedError(Exception):
    pass


_EXECUTOR_WAITABLE_ERRORS = (_StillInitializedError, WorkerUnreachableError, *RETRYABLE_ERRORS)


async def init_or_resume_trainer(trainer: ResumableTrainer, args, *, model_id: str | None = None) -> list[Any]:
    """Initialize a trainer, or - when it survived the orchestration script - roll it back to its checkpoint."""
    if not await trainer.is_initialized(model_id=model_id):
        return await trainer.init(args, model_id=model_id)

    logger.info(
        "This trainer is already initialized, so a previous orchestration script built it; waiting for whatever it "
        "is still running, then rolling it back to its checkpoint"
    )
    await trainer.wait_idle(timeout=TRAINER_IDLE_TIMEOUT_SECONDS)
    start_rollout_ids = await trainer.load_state(model_id=model_id)
    logger.info(f"Resumed an already-initialized trainer at rollout ids {start_rollout_ids}")
    return start_rollout_ids


async def wait_until_rollout_executor_is_free(
    handle: BaseWorkerHandle, *, timeout: float = EXECUTOR_FREE_TIMEOUT_SECONDS
) -> None:
    """Wait until the rollout executor answering us is a fresh one, not the process the previous script drove."""

    async def attempt(remaining: float) -> None:
        try:
            initialized = await handle.is_initialized()
        except ServerRestartedError:
            await handle.wait_ready(timeout=min(remaining, _EXECUTOR_POLL_INTERVAL_SECONDS))
            raise _StillInitializedError("the rollout executor is being replaced right now") from None
        if initialized:
            raise _StillInitializedError("the rollout executor still belongs to the previous orchestration script")

    try:
        await retry_until_deadline(
            attempt,
            total_seconds=timeout,
            retry_on=_EXECUTOR_WAITABLE_ERRORS,
            initial_delay=_EXECUTOR_POLL_INTERVAL_SECONDS,
            max_delay=_EXECUTOR_POLL_INTERVAL_SECONDS,
            log_fields=dict(tag="hot_restart", op="wait_rollout_executor_free"),
        )
    except _EXECUTOR_WAITABLE_ERRORS as e:
        raise TimeoutError(
            f"The rollout executor answering us was still not a fresh process after {timeout}s ({e!r}); a hot "
            f"restart replaces its pod, and everything up to and including that replacement has to fit in this "
            f"budget, so either the pod is not being replaced or the previous script's executor never went away"
        ) from e


async def init_or_resume_inference_controller(inference_controller) -> None:
    """Initialize the inference side, or take over the one a previous orchestration script left running."""
    if not await inference_controller.is_initialized():
        await inference_controller.init()
        return

    logger.info("The inference controller outlived a previous orchestration script; taking it over as it is")
    await _wait_until_previous_calls_end(inference_controller)
    await _close_stale_update_weights_window(inference_controller)
    await _abort_inflight_rollouts(inference_controller)


async def _wait_until_previous_calls_end(inference_controller) -> None:
    try:
        await inference_controller.wait_idle(timeout=INFERENCE_TAKE_OVER_TIMEOUT_SECONDS)
    except TimeoutError as e:
        raise TimeoutError(
            f"A call of the previous orchestration script was still running on the inference controller after "
            f"{INFERENCE_TAKE_OVER_TIMEOUT_SECONDS}s ({e!r}); whether it holds the controller's lock can only be "
            f"read once it ended, so this run cannot take the controller over and its deployment has to be "
            f"restarted before this run is hot restarted again"
        ) from e


async def _abort_inflight_rollouts(inference_controller) -> None:
    """Stop every generation a previous orchestration script left behind; this run owns the fleet now."""
    refused = await _within(
        inference_controller.abort_all(),
        timeout=INFERENCE_TAKE_OVER_TIMEOUT_SECONDS,
        action="abort the generations the previous orchestration script left in flight",
    )
    if refused:
        logger.error(
            f"Cells {sorted(refused)} refused the abort, so a generation of the previous orchestration script may "
            f"still be running on them and its samples may reach this run; the other cells were aborted anyway"
        )
        return

    logger.info("Aborted every in-flight generation, so this run starts from a quiet inference fleet")


async def _close_stale_update_weights_window(inference_controller) -> None:
    if not await inference_controller.is_update_weights_window_open():
        return

    logger.warning(
        "The previous orchestration script stopped inside a weight update, so the inference controller still holds "
        "the lock that update opened; aborting that window before this script drives the fleet"
    )
    await _within(
        inference_controller.abort_update_weights(),
        timeout=INFERENCE_TAKE_OVER_TIMEOUT_SECONDS,
        action="abort the weight-update window the previous orchestration script left open",
    )


async def _within(call: Coroutine[Any, Any, _T], *, timeout: float, action: str) -> _T:
    try:
        return await asyncio.wait_for(call, timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError) as e:
        raise TimeoutError(
            f"Timed out after {timeout}s trying to {action}; the surviving inference controller is not answering "
            f"lifecycle calls, most likely because one of its own calls still holds its lock, so it cannot be taken "
            f"over and its deployment has to be restarted before this run is hot restarted again"
        ) from e


def log_startup_weight_alignment(*, weight_version: int | None) -> None:
    logger.info(
        f"Pushed the trainer's weights into the rollout engines at startup, so both sides now agree on weight "
        f"version {weight_version}; a trainer that outlived the orchestration script keeps counting from where it "
        f"was, so this number does not restart at zero"
    )
