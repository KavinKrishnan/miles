from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest

from miles.ray import hot_restart as hot_restart_module
from miles.ray.hot_restart import (
    _abort_inflight_rollouts,
    init_or_resume_inference_controller,
    init_or_resume_trainer,
    wait_until_rollout_executor_is_free,
)
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.worker_handle import WorkerUnreachableError


class _FakeTrainer:
    def __init__(self, *, initialized: bool) -> None:
        self.initialized = initialized
        self.calls: list[str] = []
        self.idle_timeouts: list[float] = []

    async def is_initialized(self, model_id: str | None = None) -> bool:
        self.calls.append("is_initialized")
        return self.initialized

    async def init(self, args, model_id: str | None = None) -> list[Any]:
        self.calls.append("init")
        return [7]

    async def load_state(self, model_id: str | None = None) -> list[Any]:
        self.calls.append("load_state")
        return [3]

    async def wait_idle(self, *, timeout: float) -> None:
        self.calls.append("wait_idle")
        self.idle_timeouts.append(timeout)


class _FakeInferenceController:
    def __init__(
        self,
        *,
        initialized: bool,
        update_weights_window_open: bool = False,
        wedged: bool = False,
        busy: bool = False,
        cells_refusing_the_abort: list[str] | None = None,
    ) -> None:
        self.initialized = initialized
        self.update_weights_window_open = update_weights_window_open
        self.wedged = wedged
        self.busy = busy
        self.cells_refusing_the_abort = cells_refusing_the_abort or []
        self.calls: list[str] = []
        self.idle_timeouts: list[float] = []

    async def is_initialized(self) -> bool:
        return self.initialized

    async def wait_idle(self, *, timeout: float) -> None:
        self.calls.append("wait_idle")
        self.idle_timeouts.append(timeout)
        if self.busy:
            raise TimeoutError("InferenceController was still busy")

    async def is_update_weights_window_open(self) -> bool:
        self.calls.append("is_update_weights_window_open")
        return self.update_weights_window_open

    async def init(self) -> None:
        self.calls.append("init")

    async def abort_update_weights(self) -> None:
        self.calls.append("abort_update_weights")
        await self._maybe_hang()

    async def abort_all(self) -> list[str]:
        self.calls.append("abort_all")
        await self._maybe_hang()
        return self.cells_refusing_the_abort

    async def _maybe_hang(self) -> None:
        if self.wedged:
            await asyncio.sleep(3600)


class _FakeExecutor:
    def __init__(self, answers: list[bool | Exception], *, ready_error: Exception | None = None) -> None:
        self._answers = list(answers)
        self._ready_error = ready_error
        self.ready_calls = 0

    async def is_initialized(self) -> bool:
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def wait_ready(self, *, timeout: float) -> None:
        self.ready_calls += 1
        if self._ready_error is not None:
            raise self._ready_error


@pytest.fixture
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hot_restart_module, "_EXECUTOR_POLL_INTERVAL_SECONDS", 0.01)


@pytest.fixture
def short_take_over_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hot_restart_module, "INFERENCE_TAKE_OVER_TIMEOUT_SECONDS", 0.05)


class TestInitOrResumeTrainer:
    async def test_a_trainer_that_never_ran_is_initialized(self):
        """A cold start must be untouched by the resume protocol."""
        trainer = _FakeTrainer(initialized=False)

        assert await init_or_resume_trainer(trainer, object()) == [7]
        assert trainer.calls == ["is_initialized", "init"]

    async def test_a_surviving_trainer_is_waited_out_and_reloaded(self):
        """The whole point: reattach to a live trainer instead of rebuilding it from scratch."""
        trainer = _FakeTrainer(initialized=True)

        assert await init_or_resume_trainer(trainer, object()) == [3]
        assert trainer.calls == ["is_initialized", "wait_idle", "load_state"]

    async def test_the_in_flight_wait_happens_before_the_reload(self):
        """Reloading a checkpoint into a model a train step is still writing would corrupt it."""
        trainer = _FakeTrainer(initialized=True)

        await init_or_resume_trainer(trainer, object())

        assert trainer.calls.index("wait_idle") < trainer.calls.index("load_state")

    async def test_the_wait_is_bounded(self):
        """A wedged trainer must fail the restart rather than hang the new script forever."""
        trainer = _FakeTrainer(initialized=True)

        await init_or_resume_trainer(trainer, object())

        assert trainer.idle_timeouts == [hot_restart_module.TRAINER_IDLE_TIMEOUT_SECONDS]

    async def test_the_model_id_is_passed_through(self):
        """A multi policy run resumes one policy at a time."""
        trainer = _FakeTrainer(initialized=False)

        await init_or_resume_trainer(trainer, object(), model_id="policy_a")

        assert trainer.calls == ["is_initialized", "init"]


class TestInitOrResumeInferenceController:
    async def test_a_fresh_controller_is_initialized(self):
        """A cold start initializes the inference side as it always did."""
        controller = _FakeInferenceController(initialized=False)

        await init_or_resume_inference_controller(controller)

        assert controller.calls == ["init"]

    async def test_a_surviving_controller_is_taken_over_without_a_second_init(self):
        """An inference deployment that outlived the script must not be re-initialized under its own engines."""
        controller = _FakeInferenceController(initialized=True)

        await init_or_resume_inference_controller(controller)

        assert "init" not in controller.calls

    async def test_a_quiet_controller_keeps_its_lock_untouched(self):
        """Aborting a weight-update window nobody opened would reattach to a lock nobody holds."""
        controller = _FakeInferenceController(initialized=True, update_weights_window_open=False)

        await init_or_resume_inference_controller(controller)

        assert controller.calls == ["wait_idle", "is_update_weights_window_open", "abort_all"]

    async def test_a_weight_update_the_previous_script_left_open_is_aborted(self):
        """start_update_weights holds the lock across calls, so abort_all would wait on it forever."""
        controller = _FakeInferenceController(initialized=True, update_weights_window_open=True)

        await init_or_resume_inference_controller(controller)

        assert controller.calls == ["wait_idle", "is_update_weights_window_open", "abort_update_weights", "abort_all"]

    async def test_a_controller_that_cannot_be_freed_fails_loud(self, short_take_over_budget: None):
        """Hanging here would leave the operator with a silent hot restart that never starts training."""
        controller = _FakeInferenceController(initialized=True, update_weights_window_open=True, wedged=True)

        with pytest.raises(TimeoutError, match="weight-update window"):
            await init_or_resume_inference_controller(controller)

    async def test_a_fresh_controller_is_never_asked_about_a_weight_update_window(self):
        """A controller this script is about to initialize cannot be holding a predecessor's lock."""
        controller = _FakeInferenceController(initialized=False, update_weights_window_open=True)

        await init_or_resume_inference_controller(controller)

        assert controller.calls == ["init"]


class TestAbortInflightRollouts:
    async def test_every_generation_is_aborted(self):
        """Orphan requests would otherwise resume under the new weights the startup update pushes."""
        controller = _FakeInferenceController(initialized=True)

        await _abort_inflight_rollouts(controller)

        assert controller.calls == ["abort_all"]

    async def test_a_quiet_fleet_is_announced_only_when_every_cell_answered(self, caplog):
        """A cell that kept generating pollutes this run's data, so the success line must not cover for it."""
        controller = _FakeInferenceController(initialized=True, cells_refusing_the_abort=["west-engine-0-0-0"])

        with caplog.at_level(logging.ERROR):
            await _abort_inflight_rollouts(controller)

        assert "west-engine-0-0-0" in caplog.text
        assert "quiet inference fleet" not in caplog.text

    async def test_a_fleet_that_answered_every_abort_is_announced_quiet(self, caplog):
        """The ordinary take-over says so, and an operator reads that line as a fleet with no request left on it."""
        controller = _FakeInferenceController(initialized=True)

        with caplog.at_level(logging.INFO):
            await _abort_inflight_rollouts(controller)

        assert "quiet inference fleet" in caplog.text

    async def test_a_controller_that_never_answers_fails_loud(self, short_take_over_budget: None):
        """An unbounded abort would hang the new script instead of naming the deployment to restart."""
        controller = _FakeInferenceController(initialized=True, wedged=True)

        with pytest.raises(TimeoutError, match="in flight"):
            await _abort_inflight_rollouts(controller)

    async def test_a_cold_start_never_aborts_anything(self):
        """A run that deploys its own engines would make an unanswered abort a new startup dependency."""
        controller = _FakeInferenceController(initialized=False)

        await init_or_resume_inference_controller(controller)

        assert controller.calls == ["init"]

    async def test_a_take_over_aborts_before_the_trainers_are_built(self):
        """The reload of a large checkpoint takes minutes, and until then the fleet still serves the dead script."""
        controller = _FakeInferenceController(initialized=True)

        await init_or_resume_inference_controller(controller)

        assert "abort_all" in controller.calls


class TestTheTakeOverWaitsForTheCallsOfThePreviousScript:
    async def test_the_wait_comes_before_the_window_is_read(self):
        """A start_update_weights still running holds the lock without having detached it, so the window reads shut."""
        controller = _FakeInferenceController(initialized=True, update_weights_window_open=True)

        await init_or_resume_inference_controller(controller)

        assert controller.calls.index("wait_idle") < controller.calls.index("is_update_weights_window_open")

    async def test_the_wait_is_bounded_by_the_take_over_budget(self):
        """A controller nobody can take over has to fail the restart rather than hang it."""
        controller = _FakeInferenceController(initialized=True)

        await init_or_resume_inference_controller(controller)

        assert controller.idle_timeouts == [hot_restart_module.INFERENCE_TAKE_OVER_TIMEOUT_SECONDS]

    async def test_a_call_that_never_ends_fails_loud(self):
        """The script that died inside start_update_weights is exactly the case this wait exists for."""
        controller = _FakeInferenceController(initialized=True, busy=True)

        with pytest.raises(TimeoutError, match="still running on the inference controller"):
            await init_or_resume_inference_controller(controller)

        assert "abort_all" not in controller.calls


class TestWaitUntilRolloutExecutorIsFree:
    async def test_a_fresh_executor_is_accepted_at_once(self):
        """The normal case must not pay a polling delay."""
        executor = _FakeExecutor([False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_an_executor_of_the_previous_script_is_waited_out(self, fast_polling: None):
        """Initializing the old executor a second time would drive the process that is about to die."""
        executor = _FakeExecutor([True, True, False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_an_executor_being_replaced_mid_wait_is_re_baselined(self, fast_polling: None):
        """The executor is expected to restart during this wait, so its boot uuid change is not a violation."""
        executor = _FakeExecutor([True, ServerRestartedError("replaced"), False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

        assert executor.ready_calls == 1

    async def test_an_executor_that_never_frees_up_times_out(self, fast_polling: None):
        """A new script must not silently share a run with the executor of its predecessor."""
        executor = _FakeExecutor([True] * 100)

        with pytest.raises(TimeoutError, match="not a fresh process"):
            await wait_until_rollout_executor_is_free(executor, timeout=0.05)

    async def test_an_executor_pod_being_recreated_is_waited_out(self, fast_polling: None):
        """Replacing the pod is the whole point, and it is unreachable for as long as that takes."""
        executor = _FakeExecutor([True, WorkerUnreachableError("pod is gone"), False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_a_transport_error_is_waited_out_too(self, fast_polling: None):
        """A connection refused while kubernetes reschedules the pod is the expected state, not a failure."""
        executor = _FakeExecutor([httpx.ConnectError("refused"), False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_an_executor_that_stays_unreachable_reports_a_timeout(self, fast_polling: None):
        """The message has to say the executor never came back, not leak a transport error."""
        executor = _FakeExecutor([WorkerUnreachableError("pod is gone")] * 100)

        with pytest.raises(TimeoutError, match="not a fresh process"):
            await wait_until_rollout_executor_is_free(executor, timeout=0.05)

    async def test_a_readiness_wait_that_fails_is_retried_rather_than_fatal(self, fast_polling: None):
        """wait_ready itself throws while the replacement pod is still being scheduled."""
        executor = _FakeExecutor([ServerRestartedError("replaced"), False], ready_error=WorkerUnreachableError("no"))

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)


class TestTheResumeSurfaceCrossesTheWire:
    def test_the_trainer_controller_answers_whether_it_is_initialized(self):
        """A restarted orchestration script asks this over rpc before it decides how to start."""
        from miles.ray.train.group import TrainerController

        assert "is_initialized" in collect_rpc_method_specs(TrainerController)

    def test_the_trainer_controller_exposes_the_reload(self):
        """load_state is the resume path, and an unexposed method makes the pool unreachable for it."""
        from miles.ray.train.group import TrainerController

        assert "load_state" in collect_rpc_method_specs(TrainerController)

    def test_the_rollout_executor_answers_whether_it_is_initialized(self):
        """The new script waits on exactly this answer until the old executor is gone."""
        from miles.ray.rollout.rollout_executor import RolloutExecutor

        assert "is_initialized" in collect_rpc_method_specs(RolloutExecutor)

    def test_the_inference_controller_exposes_the_take_over_surface(self):
        """A surviving inference deployment is taken over and quiesced through these two calls."""
        from miles.ray.rollout.inference_controller import InferenceController

        specs = collect_rpc_method_specs(InferenceController)

        assert {"is_initialized", "abort_all", "is_update_weights_window_open", "abort_update_weights"} <= set(specs)
