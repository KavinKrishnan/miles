from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args, make_samples_grouped

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.utils.multi_lora import EmptyBatchTimeoutError
from miles.utils.types import WeightVersionSpan, WeightVersionsPerCall


@pytest.fixture
def http_client_calls(monkeypatch) -> list[str]:
    import miles.ray.rollout.rollout_executor as rexec

    recorded: list[str] = []
    monkeypatch.setattr(rexec, "init_http_client", lambda args: recorded.append("init_http_client"))
    return recorded


@pytest.fixture
def own_args_resolutions(monkeypatch) -> list[tuple[str, object]]:
    import miles.ray.rollout.rollout_executor as rexec

    recorded: list[tuple[str, object]] = []

    async def _record_router(args, **kwargs):
        recorded.append(("resolve_router_addrs", args))
        return {}

    async def _record_session(args, **kwargs):
        recorded.append(("wait_session_server_ready", args))
        return {}

    monkeypatch.setattr(rexec, "resolve_router_addrs", _record_router)
    monkeypatch.setattr(rexec, "wait_session_server_ready", _record_session)
    return recorded


@pytest.fixture
def patch_low_level(monkeypatch, http_client_calls, own_args_resolutions):
    import miles.ray.rollout.rollout_executor as rexec

    monkeypatch.setattr(rexec, "configure_logger", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "init_tracking", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "load_function", lambda path: lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "load_rollout_function", lambda input, path: lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "log_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "log_eval_rollout_data", lambda *a, **kw: None)
    monkeypatch.setattr(rexec, "save_debug_rollout_data", lambda *a, **kw: None)


class _NeverUsedProvider:
    async def get_addrs(self, worker_name: str):
        raise AssertionError("the stubbed resolution must not ask the provider for an address")


async def _make_executor(args):
    executor = RolloutExecutor(
        args=args,
        router_providers=[_NeverUsedProvider()],
        session_server_provider=None,
        inference_controller_provider=_NeverUsedProvider(),
    )
    await executor.init()
    return executor


def _make_test_args(**overrides):
    return make_args(
        sglang_model_routers={"default": ("127.0.0.1", 30000)},
        use_wandb=False,
        use_tensorboard=False,
        use_mlflow=False,
        use_distributed_post=False,
        **overrides,
    )


@pytest.mark.asyncio
class TestProcessSetup:
    async def test_initializes_the_http_client(self, ray_local_mode, patch_low_level, http_client_calls):
        """The rollout functions issue their HTTP from this actor, so the client is created here."""
        await _make_executor(_make_test_args())

        assert http_client_calls == ["init_http_client"]

    async def test_skips_the_http_client_in_debug_train_only(self, ray_local_mode, patch_low_level, http_client_calls):
        """No engines exist in this mode, so there is nothing to talk to."""
        args = _make_test_args()
        args.debug_train_only = True

        await _make_executor(args)

        assert http_client_calls == []

    async def test_it_resolves_the_router_and_session_servers_on_its_own_args(
        self, ray_local_mode, patch_low_level, own_args_resolutions
    ):
        """The platform builds the executor before the driver resolves anything, so it must resolve for itself."""
        args = _make_test_args()

        executor = await _make_executor(args)

        assert [name for name, _ in own_args_resolutions] == ["resolve_router_addrs", "wait_session_server_ready"]
        assert all(seen is executor.args for _, seen in own_args_resolutions)

    async def test_it_resolves_nothing_in_debug_train_only(
        self, ray_local_mode, patch_low_level, own_args_resolutions
    ):
        """No engines and no session servers exist in this mode, so there is nothing to wait for."""
        args = _make_test_args()
        args.debug_train_only = True

        await _make_executor(args)

        assert own_args_resolutions == []


@pytest.mark.asyncio
class TestGenerate:
    async def test_invokes_rollout_fn_with_correct_input_and_returns_dp_split(self, ray_local_mode, patch_low_level):
        """generate passes a train input and returns the samples split per dp rank."""
        args = _make_test_args()
        args.global_batch_size = 8

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})

        captured: list = []

        def fake_rollout_fn(input):
            captured.append(input)
            return RolloutFnTrainOutput(
                samples=[make_samples_grouped(n_groups=2, group_size=4)],
                metrics={"my_metric": 1.23},
            )

        executor.generate_rollout = fake_rollout_fn

        result = await executor.get(rollout_id=42)

        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnTrainInput)
        assert captured[0].rollout_id == 42
        assert set(result) == {"sample_indices", "data_ref", "empty_batch_timeout"}
        assert result["empty_batch_timeout"] is False
        data_refs = result["data_ref"]
        assert len(data_refs) == 2
        partitions = ray.get([box.inner for box in data_refs])
        for partition in partitions:
            assert "tokens" in partition
            assert "rewards" in partition
            assert "loss_masks" in partition
            assert len(partition["tokens"]) == 4

    async def test_an_empty_batch_timeout_is_reported_as_a_field_rather_than_an_exception(
        self, ray_local_mode, patch_low_level
    ):
        """Under rpc a remote exception arrives as RpcWorkerCallError, so the multi-LoRA driver reads a field."""
        args = _make_test_args()
        args.global_batch_size = 8

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})

        def timing_out_rollout_fn(input):
            raise EmptyBatchTimeoutError("no trainable group arrived")

        executor.generate_rollout = timing_out_rollout_fn

        result = await executor.get(rollout_id=11)

        assert result == dict(sample_indices=None, data_ref=None, empty_batch_timeout=True)

    async def test_rejects_samples_generated_under_the_default_weight_version(self, ray_local_mode, patch_low_level):
        """A batch carrying the sglang never-updated version must fail get(), not reach training."""
        args = _make_test_args()
        args.global_batch_size = 8

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 2})

        samples = make_samples_grouped(n_groups=2, group_size=4)
        samples[0].weight_versions = [
            WeightVersionsPerCall(spans=[WeightVersionSpan(version="default", abs_start=0, abs_end=1)])
        ]

        executor.generate_rollout = lambda input: RolloutFnTrainOutput(samples=[samples], metrics=None)

        with pytest.raises(AssertionError, match="never updated"):
            await executor.get(rollout_id=42)

    async def test_does_not_touch_the_inference_side(self, ray_local_mode, patch_low_level):
        """The controller is a driver-side object the executor cannot reach, so generate must not need it."""
        args = _make_test_args()
        args.global_batch_size = 4

        executor = await _make_executor(args)
        executor.set_train_parallel_config({"dp_size": 1})
        executor.generate_rollout = lambda input: RolloutFnTrainOutput(
            samples=[make_samples_grouped(n_groups=1, group_size=4)], metrics={}
        )

        await executor.get(rollout_id=7)

        assert not hasattr(executor, "servers")
        assert not hasattr(executor, "_health_monitors")


class _RecordingRolloutFn(BaseRolloutFn):
    def __init__(self, name: str, log: list[tuple[str, str, object]]) -> None:
        self._name = name
        self._log = log

    def __call__(self, input):
        raise AssertionError("not exercised by the checkpointing tests")

    def save(self, rollout_id: int) -> None:
        self._log.append((self._name, "save", rollout_id))

    def load(self, rollout_id: int | None) -> None:
        self._log.append((self._name, "load", rollout_id))


@pytest.mark.asyncio
class TestCheckpointing:
    async def test_save_and_load_reach_only_the_train_rollout_function(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """One checkpoint is enough: the train and eval instances are becoming a single object."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=False)

        executor = await _make_executor(args)
        executor.use_experimental_refactor = True
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)
        executor.data_source = MagicMock()

        executor.save(rollout_id=7)
        executor.load(rollout_id=7)

        assert calls == [
            ("train", "save", 7),
            ("train", "load", 7),
        ]

    async def test_save_forwards_to_the_data_source_for_a_global_dataset(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """With a global dataset both the data source and the rollout functions are checkpointed."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=True)

        executor = await _make_executor(args)
        executor.use_experimental_refactor = True
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)
        executor.data_source = MagicMock()

        executor.save(rollout_id=5)

        executor.data_source.save.assert_called_once_with(5)
        assert ("train", "save", 5) in calls

    async def test_save_forwards_to_the_data_source_without_a_global_dataset(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """A custom data source is saved as unconditionally as it is loaded, so its state can be restored."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=False)

        executor = await _make_executor(args)
        executor.use_experimental_refactor = True
        calls: list[tuple[str, str, object]] = []
        executor.generate_rollout = _RecordingRolloutFn("train", calls)
        executor.eval_generate_rollout = _RecordingRolloutFn("eval", calls)
        executor.data_source = MagicMock()

        executor.save(rollout_id=3)
        executor.load(rollout_id=3)

        executor.data_source.save.assert_called_once_with(3)
        executor.data_source.load.assert_called_once_with(3)
        assert ("train", "save", 3) in calls

    async def test_legacy_function_path_does_not_get_save_load(
        self,
        ray_local_mode,
        patch_low_level,
        monkeypatch,
    ):
        """Without the experimental flag the rollout functions are bare callables, so they are not checkpointed."""
        import miles.ray.rollout.rollout_executor as rexec

        monkeypatch.setattr(rexec, "event_logger_checkpoint", MagicMock())
        args = _make_test_args(rollout_global_dataset=False)

        executor = await _make_executor(args)
        executor.use_experimental_refactor = False
        executor.generate_rollout = lambda *a, **kw: None
        executor.eval_generate_rollout = lambda *a, **kw: None
        executor.data_source = MagicMock()

        executor.save(rollout_id=1)
        executor.load(rollout_id=1)

        executor.data_source.load.assert_called_once_with(1)


@pytest.mark.asyncio
class TestEval:
    async def test_invokes_eval_fn_with_eval_input(self, ray_local_mode, patch_low_level):
        """eval passes an eval input carrying the rollout id."""
        executor = await _make_executor(_make_test_args())

        captured: list = []

        def fake_eval_fn(input):
            captured.append(input)
            return RolloutFnEvalOutput(data={"my_dataset": {"rewards": [0.5, 1.0]}}, metrics={})

        executor.eval_generate_rollout = fake_eval_fn

        await executor.eval(rollout_id=10)

        assert len(captured) == 1
        assert isinstance(captured[0], RolloutFnEvalInput)
        assert captured[0].rollout_id == 10

    async def test_skipped_in_debug_train_only_mode(self, ray_local_mode, patch_low_level):
        """debug_train_only short-circuits eval before the rollout function runs."""
        args = _make_test_args()
        args.debug_train_only = True

        executor = await _make_executor(args)

        called: list = []
        executor.eval_generate_rollout = lambda inp: called.append(inp)

        await executor.eval(rollout_id=10)

        assert called == []
