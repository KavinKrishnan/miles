import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from miles.utils.file_utils import atomic_write_text
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)

MULTI_POLICY_STATE_DIRNAME = "multi_policy_state"

SAVE_PARK_TIMEOUT_SECONDS = 3600.0


class MultiPolicyCheckpointState(FrozenStrictBaseModel):
    primary_model_id: str
    rollout_ids: dict[str, int]
    finished_model_ids: list[str] = []


def multi_policy_state_path(save_dir: Path, primary_rollout_id: int) -> Path:
    return Path(save_dir) / MULTI_POLICY_STATE_DIRNAME / f"rollout_ids_{primary_rollout_id}.json"


def save_multi_policy_state(save_dir: Path, state: MultiPolicyCheckpointState) -> None:
    primary_rollout_id = state.rollout_ids[state.primary_model_id]
    path = multi_policy_state_path(save_dir, primary_rollout_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, state.model_dump_json())
    logger.info(f"Saved multi policy checkpoint state {state.rollout_ids} to {path}")


def load_multi_policy_state(save_dir: Path, primary_rollout_id: int) -> MultiPolicyCheckpointState | None:
    path = multi_policy_state_path(save_dir, primary_rollout_id)
    if not path.exists():
        return None
    return MultiPolicyCheckpointState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def assert_restored_rollout_ids(expected: MultiPolicyCheckpointState, restored: dict[str, int]) -> None:
    recorded = expected.rollout_ids
    known = set(recorded) | set(expected.finished_model_ids)
    assert set(restored) == known, (
        f"multi policy checkpoint mismatch: the primary model {expected.primary_model_id!r} recorded the "
        f"policies {sorted(known)}, but this run trains {sorted(restored)}; a policy the record never saw "
        f"would start from zero against a data source and a rollout executor restored to the primary's "
        f"position, and a policy the record names but this run drops leaves that position unproven"
    )
    disagreeing = {
        model_id: restored.get(model_id) for model_id, value in recorded.items() if restored.get(model_id) != value
    }
    assert not disagreeing, (
        f"multi policy checkpoint mismatch: the primary model {expected.primary_model_id!r} recorded "
        f"{recorded}, but the policies restored {disagreeing} for those ids; loading inconsistent "
        f"positions would train each policy against the wrong global state "
        f"(policies that had already finished when the checkpoint was written: {expected.finished_model_ids})"
    )


class MultiPolicySaveCoordinator:
    def __init__(self, *, model_ids: list[str], primary_model_id: str) -> None:
        assert primary_model_id in model_ids
        self._primary_model_id = primary_model_id
        self._active: set[str] = set(model_ids)
        self._finished: list[str] = []
        self._parked: set[str] = set()
        self._rollout_ids: dict[str, int] = {}
        self._save_requested = False
        self._force_sync = False
        self._final_save_in_flight = False
        self._cond = asyncio.Condition()

    @property
    def rollout_ids(self) -> dict[str, int]:
        return dict(self._rollout_ids)

    @property
    def finished_model_ids(self) -> list[str]:
        return list(self._finished)

    async def leave(self, model_id: str) -> None:
        async with self._cond:
            self._active.discard(model_id)
            self._parked.discard(model_id)
            if model_id not in self._finished:
                self._finished.append(model_id)
            self._cond.notify_all()

    @asynccontextmanager
    async def saving(self, primary_rollout_id: int, *, force_sync: bool) -> AsyncIterator[None]:
        try:
            await self.begin_save(primary_rollout_id, force_sync=force_sync)
            yield
        finally:
            await asyncio.shield(self.end_save())

    async def begin_save(
        self, primary_rollout_id: int, *, force_sync: bool = False, timeout: float = SAVE_PARK_TIMEOUT_SECONDS
    ) -> None:
        async with self._cond:
            assert not self._save_requested, "a save is already in flight"
            await self._cond.wait_for(lambda: not self._final_save_in_flight and not self._parked)
            self._save_requested = True
            self._force_sync = force_sync
            self._rollout_ids = {self._primary_model_id: primary_rollout_id}
            self._cond.notify_all()
            try:
                await asyncio.wait_for(self._cond.wait_for(self._others_parked), timeout=timeout)
            except TimeoutError as e:
                raise TimeoutError(
                    f"the primary model {self._primary_model_id!r} waited {timeout}s at rollout "
                    f"{primary_rollout_id} for the other policies to park; still running: "
                    f"{sorted(self._active - self._parked - {self._primary_model_id})}, "
                    f"parked: {sorted(self._parked)}, finished: {self._finished}"
                ) from e
        logger.info(f"All policy models parked at {self._rollout_ids} for the global checkpoint")

    @asynccontextmanager
    async def final_saving(self, model_id: str, rollout_id: int) -> AsyncIterator[None]:
        async with self._cond:
            await self._cond.wait_for(lambda: not self._save_requested and not self._final_save_in_flight)
            self._final_save_in_flight = True
            self._rollout_ids[model_id] = rollout_id
        logger.info(f"Policy model {model_id!r} is saving its last rollout {rollout_id} on its own")
        try:
            yield
        finally:
            async with self._cond:
                self._final_save_in_flight = False
                self._cond.notify_all()

    async def end_save(self) -> None:
        async with self._cond:
            self._save_requested = False
            self._cond.notify_all()

    async def maybe_park(
        self, model_id: str, rollout_id: int, save_model_fn: Callable[[bool], Awaitable[None]]
    ) -> bool:
        async with self._cond:
            if not self._save_requested:
                return False
            force_sync = self._force_sync

        await save_model_fn(force_sync)

        async with self._cond:
            self._rollout_ids[model_id] = rollout_id
            self._parked.add(model_id)
            self._cond.notify_all()
            await self._cond.wait_for(lambda: not self._save_requested)
            self._parked.discard(model_id)
            self._cond.notify_all()
        return True

    def _others_parked(self) -> bool:
        return self._parked == self._active - {self._primary_model_id}
