from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import asyncio
from argparse import Namespace

import pytest

from miles.ray.rollout import rollout_executor as rollout_executor_module
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.timer import Timer


@pytest.fixture(autouse=True)
def _quiet_rollout_pipeline(monkeypatch):
    Timer().timers.clear()
    Timer().start_time.clear()
    monkeypatch.setattr(rollout_executor_module, "save_debug_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rollout_executor_module, "convert_samples_to_train_data", lambda *a, **kw: {})
    monkeypatch.setattr(rollout_executor_module, "split_train_data_by_dp", lambda *a, **kw: None)
    yield
    Timer().timers.clear()
    Timer().start_time.clear()


def _make_executor(*, multi_policy: bool) -> RolloutExecutor:
    executor = RolloutExecutor.__new__(RolloutExecutor)
    executor.args = Namespace(delay_split_train_data_by_dp=False)
    executor.data_source = Namespace()
    executor.train_parallel_config = {}
    executor.custom_convert_samples_to_train_data_func = None
    executor.custom_reward_post_process_func = None
    executor.weight_version = None
    executor._weight_versions = {}
    executor._rollout_ids = {}
    executor._multi_policy = multi_policy
    return executor


class TestRolloutTimerNaming:
    async def test_two_policies_may_be_generating_at_the_same_time(self, monkeypatch):
        """The rollout timer is a process singleton that refuses a second start under the same name."""
        logged: list = []
        monkeypatch.setattr(
            rollout_executor_module,
            "log_rollout_data",
            lambda *a, trainer_model_id=None, **kw: logged.append(trainer_model_id),
        )
        executor = _make_executor(multi_policy=True)
        both_arrived = asyncio.Event()
        arrivals = 0

        async def _get_rollout_data(rollout_id, trainer_model_id=None):
            nonlocal arrivals
            arrivals += 1
            if arrivals == 2:
                both_arrived.set()
            await asyncio.wait_for(both_arrived.wait(), timeout=5)
            return [], None, None

        executor._get_rollout_data = _get_rollout_data

        await asyncio.wait_for(
            asyncio.gather(executor.get(0, trainer_model_id="a"), executor.get(0, trainer_model_id="b")), timeout=5
        )

        assert sorted(Timer().log_dict()) == ["rollout/a", "rollout/b"]
        assert sorted(logged) == ["a", "b"]

    async def test_a_single_policy_run_keeps_the_timer_and_metric_names_it_had(self, monkeypatch):
        """Every existing dashboard query is written against the unprefixed names."""
        logged: list = []
        monkeypatch.setattr(
            rollout_executor_module,
            "log_rollout_data",
            lambda *a, trainer_model_id=None, **kw: logged.append(trainer_model_id),
        )
        executor = _make_executor(multi_policy=False)

        async def _get_rollout_data(rollout_id, trainer_model_id=None):
            return [], None, None

        executor._get_rollout_data = _get_rollout_data

        await executor.get(0, trainer_model_id="default")

        assert list(Timer().log_dict()) == ["rollout"]
        assert logged == [None]
