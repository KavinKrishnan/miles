from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tests.fast.ray.train import conftest as train_conftest

from miles.ray.specs.trainer_identity import DEFAULT_TRAINER_ROLE
from miles.ray.train.group import TrainerController

pytestmark = pytest.mark.asyncio

_NUM_CELLS = 2


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        indep_dp=True,
        enable_witness=False,
        witness_buffer_size=100,
        save_debug_event_data=None,
        trainer_heartbeat_checker_interval=10.0,
        trainer_heartbeat_checker_timeout=10.0,
        trainer_heartbeat_checker_first_wait=300.0,
        trainer_heartbeat_checker_failure_threshold=3,
        ci_ft_test_actions=None,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        actor_num_nodes=1,
        actor_num_gpus_per_node=_NUM_CELLS,
    )


def _make_controller() -> TrainerController:
    train_conftest.fake_worker_manager.num_cells = _NUM_CELLS
    return TrainerController(
        cell_provider=train_conftest.make_provider(),
        cell_operations=MagicMock(),
        role=DEFAULT_TRAINER_ROLE,
        with_ref=False,
    )


class TestInitExactlyOnce:
    async def test_a_controller_that_never_ran_reports_it_is_not_initialized(self):
        """This is the answer that sends a restarted orchestration script down the cold-start path."""
        assert await _make_controller().is_initialized() is False

    async def test_a_controller_reports_initialized_after_init(self):
        """A trainer that outlived its orchestration script must say so, or the new one rebuilds it."""
        controller = _make_controller()

        await controller.init(_make_args())

        assert await controller.is_initialized() is True

    async def test_a_second_init_is_refused(self):
        """It would create a second TCPStore, a second cell watcher and re-init every live megatron rank."""
        controller = _make_controller()
        await controller.init(_make_args())

        with pytest.raises(AssertionError, match="already been initialized"):
            await controller.init(_make_args())

    async def test_the_refusal_names_the_trainer_role(self):
        """A run has one controller per policy, so the failure has to say which one was initialized twice."""
        controller = _make_controller()
        await controller.init(_make_args())

        with pytest.raises(AssertionError, match=DEFAULT_TRAINER_ROLE):
            await controller.init(_make_args())

    async def test_a_call_for_another_model_id_is_still_refused_first(self):
        """Routing errors must not be masked by the initialization guard."""
        controller = _make_controller()

        with pytest.raises(AssertionError, match="cannot answer for model"):
            await controller.init(_make_args(), model_id="someone-else")


class TestLoadState:
    async def test_load_state_before_init_is_refused(self):
        """There is no model to reload into yet, and the failure must say that rather than crash on an attribute."""
        with pytest.raises(AssertionError, match="not initialized yet"):
            await _make_controller().load_state()

    async def test_load_state_reaches_every_rank_of_every_cell(self):
        """A checkpoint reload that skips a rank leaves the training world inconsistent."""
        controller = _make_controller()
        await controller.init(_make_args())

        await controller.load_state()

        for cell in controller._cells:
            for handle in cell._get_worker_handles():
                assert "load_state" in [name for name, _, _ in await handle.get_calls()]

    async def test_load_state_answers_the_rollout_id_of_every_rank(self):
        """The orchestration script derives the rollout id to resume at from exactly this answer."""
        controller = _make_controller()
        await controller.init(_make_args())

        answers = await controller.load_state()

        assert answers and set(answers) == {11}

    async def test_load_state_answers_for_its_own_model_only(self):
        """A composite routes by model id, and a misrouted reload would roll back the wrong policy."""
        controller = _make_controller()
        await controller.init(_make_args())

        with pytest.raises(AssertionError, match="cannot answer for model"):
            await controller.load_state(model_id="someone-else")
