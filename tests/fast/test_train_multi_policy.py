from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="stage-a-cpu", labels=[])

import asyncio
import base64
import json
from argparse import Namespace
from unittest.mock import AsyncMock

import pytest
import train_multi_policy
import yaml
from train_multi_policy import _MultiPolicyRun, _Policy

from miles.ray.train.multi_policy import multi_policy_state_path
from miles.utils.megatron_config import MegatronConfig, resolve_megatron_config


def _make_config(*model_ids: str) -> MegatronConfig:
    payload = yaml.dump({"megatron": [{"name": model_id} for model_id in model_ids]}).encode()
    return resolve_megatron_config(Namespace(megatron_config=f"base64:{base64.b64encode(payload).decode()}"))


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        num_rollout=2,
        update_weights_interval=1,
        save=None,
        save_interval=None,
        save_trigger_sentinel=None,
        debug_exit_after_rollout=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_run(args, *, model_ids: tuple[str, ...], trainers: AsyncMock) -> _MultiPolicyRun:
    config = _make_config(*model_ids)
    return _MultiPolicyRun(
        args,
        config=config,
        inference_controller=AsyncMock(),
        rollout_executor=AsyncMock(),
        num_rollout_per_epoch=None,
        trainers=trainers,
        policies=[_Policy(model_id=model_id, start_rollout_id=0) for model_id in model_ids],
    )


@pytest.fixture(autouse=True)
def _no_object_store(monkeypatch):
    monkeypatch.setattr(train_multi_policy, "remove_rollout_data_refs", lambda args, ref: None)


class TestMultiPolicyRun:
    async def test_every_policy_drains_and_updates_only_its_own_model(self, monkeypatch):
        """Two policies sharing one executor must never train on, or publish into, each other's model."""
        updated: list[tuple[str, int]] = []
        monkeypatch.setattr(
            train_multi_policy,
            "update_weights",
            AsyncMock(side_effect=lambda *a, model_id, rollout_id=None, **kw: updated.append((model_id, rollout_id))),
        )
        trainers = AsyncMock()
        run = _make_run(_make_args(), model_ids=("a", "b"), trainers=trainers)

        await run.run()

        drained = [call.kwargs["trainer_model_id"] for call in run.rollout_executor.get.await_args_list]
        trained = [call.kwargs["model_id"] for call in trainers.train.await_args_list]
        assert sorted(drained) == ["a", "a", "b", "b"]
        assert sorted(trained) == ["a", "a", "b", "b"]
        assert sorted(updated) == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]

    async def test_two_policies_are_inside_the_executor_at_the_same_time(self, monkeypatch):
        """The whole point of one loop per policy is that they overlap; the executor must tolerate it."""
        monkeypatch.setattr(train_multi_policy, "update_weights", AsyncMock())
        arrivals = 0
        both_arrived = asyncio.Event()

        async def _get(rollout_id: int, trainer_model_id: str | None = None):
            nonlocal arrivals
            arrivals += 1
            if arrivals == 2:
                both_arrived.set()
            await asyncio.wait_for(both_arrived.wait(), timeout=5)
            return dict(data_ref=None)

        run = _make_run(_make_args(num_rollout=1), model_ids=("a", "b"), trainers=AsyncMock())
        run.rollout_executor.get = _get

        await asyncio.wait_for(run.run(), timeout=5)

        assert both_arrived.is_set()

    async def test_a_failing_policy_stops_the_others_instead_of_orphaning_them(self, monkeypatch):
        """A surviving loop keeps training and writing checkpoints while the run is already dead."""
        monkeypatch.setattr(train_multi_policy, "update_weights", AsyncMock())
        rounds_of_b = 0

        async def _train(rollout_id: int, rollout_data_ref, *, model_id: str) -> None:
            nonlocal rounds_of_b
            if model_id == "a":
                raise RuntimeError("trainer a died")
            rounds_of_b += 1
            await asyncio.sleep(0.05)

        trainers = AsyncMock()
        trainers.train = _train
        run = _make_run(_make_args(num_rollout=100), model_ids=("a", "b"), trainers=trainers)

        with pytest.raises(RuntimeError, match="trainer a died"):
            await asyncio.wait_for(run.run(), timeout=5)

        assert rounds_of_b <= 2

    async def test_the_sidecar_records_where_every_policy_stood(self, monkeypatch, tmp_path):
        """A resume asserts against this record, so a missing or stale entry rejects a legal checkpoint."""
        monkeypatch.setattr(train_multi_policy, "update_weights", AsyncMock())
        args = _make_args(num_rollout=1, save=str(tmp_path), save_interval=1)

        async def _train(rollout_id: int, rollout_data_ref, *, model_id: str) -> None:
            if model_id == "b":
                await asyncio.sleep(0.05)

        trainers = AsyncMock()
        trainers.train = _train
        run = _make_run(args, model_ids=("a", "b"), trainers=trainers)

        await asyncio.wait_for(run.run(), timeout=5)

        state = json.loads(multi_policy_state_path(tmp_path, 0).read_text())
        assert state["primary_model_id"] == "a"
        assert state["rollout_ids"] == {"a": 0, "b": 0}

    async def test_a_run_without_a_save_directory_never_reaches_the_sidecar(self, monkeypatch):
        """--save-trigger-sentinel without --save used to build a Path(None) and crash the primary."""
        monkeypatch.setattr(train_multi_policy, "update_weights", AsyncMock())
        args = _make_args(num_rollout=1, save=None, save_interval=1, save_trigger_sentinel="/nonexistent/sentinel")
        run = _make_run(args, model_ids=("a", "b"), trainers=AsyncMock())
        run.coordinator.begin_save = AsyncMock(side_effect=AssertionError("must not start a save"))

        await asyncio.wait_for(run.run(), timeout=5)

        run.rollout_executor.save.assert_not_awaited()


class TestFinalSave:
    async def test_a_policy_that_outlives_the_primary_still_lands_its_last_rounds(self, monkeypatch, tmp_path):
        """The primary stops driving saves when it exits, so a slower policy used to lose its whole tail."""
        monkeypatch.setattr(train_multi_policy, "update_weights", AsyncMock())
        args = _make_args(num_rollout=4, save=str(tmp_path), save_interval=2)

        async def _slow_train(rollout_id: int, rollout_data_ref) -> None:
            await asyncio.sleep(0.02)

        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["b"].train = _slow_train
        run = _make_run(args, model_ids=("a", "b"), trainers=trainers)

        await asyncio.wait_for(run.run(), timeout=10)

        assert trainers["b"].save_model.await_args_list[-1].args[0] == args.num_rollout - 1
        assert run.saved_rollout_ids == {"a": 3, "b": 3}

    async def test_the_primary_does_not_save_its_last_round_twice(self, monkeypatch, tmp_path):
        """The periodic save already covers the last round; a second one would rewrite the same iteration."""
        monkeypatch.setattr(train_multi_policy, "update_weights", AsyncMock())
        args = _make_args(num_rollout=2, save=str(tmp_path), save_interval=1)
        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        run = _make_run(args, model_ids=("a", "b"), trainers=trainers)

        await asyncio.wait_for(run.run(), timeout=10)

        saved_rollout_ids = [call.args[0] for call in trainers["a"].save_model.await_args_list]
        assert saved_rollout_ids == [0, 1]

    async def test_the_record_names_the_last_position_of_every_policy(self, monkeypatch, tmp_path):
        """A resume asserts against the record, so a tail save the record never saw rejects the checkpoint."""
        monkeypatch.setattr(train_multi_policy, "update_weights", AsyncMock())
        args = _make_args(num_rollout=4, save=str(tmp_path), save_interval=2)

        async def _slow_train(rollout_id: int, rollout_data_ref) -> None:
            await asyncio.sleep(0.02)

        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        trainers["b"].train = _slow_train
        run = _make_run(args, model_ids=("a", "b"), trainers=trainers)

        await asyncio.wait_for(run.run(), timeout=10)

        state = json.loads(multi_policy_state_path(tmp_path, 3).read_text())
        assert state["rollout_ids"] == {"a": 3, "b": 3}

    async def test_a_run_without_a_save_directory_writes_no_final_checkpoint(self, monkeypatch):
        """--save is what asks for checkpoints at all; the tail save must not invent one."""
        monkeypatch.setattr(train_multi_policy, "update_weights", AsyncMock())
        trainers = {"a": AsyncMock(), "b": AsyncMock()}
        run = _make_run(_make_args(num_rollout=2), model_ids=("a", "b"), trainers=trainers)

        await asyncio.wait_for(run.run(), timeout=10)

        trainers["a"].save_model.assert_not_awaited()
        trainers["b"].save_model.assert_not_awaited()


class TestDefinePolicyMetricGroups:
    def test_a_single_policy_run_declares_no_extra_metric_axes(self, monkeypatch):
        """Its metric names are unchanged, so the axes the tracking backend already knows still apply."""
        calls: list[dict] = []
        monkeypatch.setattr(train_multi_policy, "define_step_key_metric_group", lambda **kwargs: calls.append(kwargs))

        train_multi_policy.define_policy_metric_groups(_make_config("only"))

        assert calls == []

    def test_a_multi_policy_run_binds_every_prefixed_curve_to_its_own_step(self, monkeypatch):
        """An undeclared prefix is plotted against wandb's internal counter instead of its own rollout step."""
        calls: list[dict] = []
        monkeypatch.setattr(train_multi_policy, "define_step_key_metric_group", lambda **kwargs: calls.append(kwargs))

        train_multi_policy.define_policy_metric_groups(_make_config("a", "b"))

        assert calls == [
            dict(prefix="a", step_key="a/rollout/step"),
            dict(prefix="a/train", step_key="a/train/step"),
            dict(prefix="b", step_key="b/rollout/step"),
            dict(prefix="b/train", step_key="b/train/step"),
        ]


class TestAssertConsistentRestore:
    @staticmethod
    def _policies(**start_rollout_ids: int) -> list[_Policy]:
        return [_Policy(model_id=model_id, start_rollout_id=value) for model_id, value in start_rollout_ids.items()]

    def test_a_fresh_run_needs_no_record(self, tmp_path):
        """Nothing was ever saved, so there is nothing to be consistent with."""
        args = Namespace(save=str(tmp_path), load=None)

        train_multi_policy._assert_consistent_restore(
            args, config=_make_config("a", "b"), policies=self._policies(a=0, b=0)
        )

    def test_the_record_is_read_from_the_load_directory(self, tmp_path):
        """Resuming with --load elsewhere and a fresh --save is the common shape and used to skip the check."""
        load_dir = tmp_path / "old"
        path = multi_policy_state_path(load_dir, 4)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(primary_model_id="a", rollout_ids={"a": 4, "b": 2}, finished_model_ids=[])))
        args = Namespace(save=str(tmp_path / "new"), load=str(load_dir))

        with pytest.raises(AssertionError, match="multi policy checkpoint mismatch"):
            train_multi_policy._assert_consistent_restore(
                args, config=_make_config("a", "b"), policies=self._policies(a=5, b=5)
            )

    def test_a_multi_policy_resume_without_a_record_fails_loudly(self, tmp_path):
        """Silently skipping the check is exactly the mixture of checkpoints the record exists to refuse."""
        args = Namespace(save=str(tmp_path), load=None)

        with pytest.raises(AssertionError, match="no record of"):
            train_multi_policy._assert_consistent_restore(
                args, config=_make_config("a", "b"), policies=self._policies(a=5, b=3)
            )

    def test_a_record_written_under_another_primary_is_refused(self, tmp_path):
        """The global rollout index means whatever the primary's index meant when it was written."""
        path = multi_policy_state_path(tmp_path, 4)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(primary_model_id="b", rollout_ids={"a": 4, "b": 4}, finished_model_ids=[])))
        args = Namespace(save=str(tmp_path), load=None)

        with pytest.raises(AssertionError, match="as the primary policy"):
            train_multi_policy._assert_consistent_restore(
                args, config=_make_config("a", "b"), policies=self._policies(a=5, b=5)
            )

    def test_a_single_policy_resume_predating_the_sidecar_is_allowed(self, tmp_path):
        """Checkpoints written by train_async.py carry no record and must stay loadable."""
        args = Namespace(save=str(tmp_path), load=None)

        train_multi_policy._assert_consistent_restore(
            args, config=_make_config("only"), policies=self._policies(only=5)
        )
