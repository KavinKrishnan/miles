import sys
import types
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest


def _stub_module(name: str, attrs: dict[str, object] | None = None, is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    if attrs is not None:
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
    sys.modules[name] = module
    return module


class _DummyDDP:
    pass


class _DummyModel:
    pass


class _DummyOptimizer:
    pass


class _DummyChainedOptimizer:
    pass


class _DummyDistributedOptimizer:
    pass


class _DummyScheduler:
    pass


class _DummyOptimizerConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeModelChunk:
    role: str | None = None


@pytest.fixture(scope="module", autouse=True)
def _mock_megatron_environment():
    original_modules = dict(sys.modules)
    try:
        _stub_module("megatron", is_package=True)
        core_module = _stub_module("megatron.core", is_package=True)
        core_module.mpu = types.SimpleNamespace()
        core_module.tensor_parallel = types.SimpleNamespace(model_parallel_cuda_manual_seed=MagicMock())
        _stub_module(
            "megatron.core.distributed",
            {
                "DistributedDataParallel": _DummyDDP,
                "finalize_model_grads": MagicMock(),
            },
        )
        _stub_module(
            "megatron.core.enums",
            {"ModelType": types.SimpleNamespace(encoder_or_decoder="encoder_or_decoder")},
        )
        _stub_module("megatron.core.models", is_package=True)
        _stub_module("megatron.core.models.gpt", {"GPTModel": _DummyModel})
        _stub_module(
            "megatron.core.optimizer",
            {
                "OptimizerConfig": _DummyOptimizerConfig,
                "get_megatron_optimizer": MagicMock(),
            },
            is_package=True,
        )
        _stub_module("megatron.core.optimizer.muon", {"get_megatron_muon_optimizer": MagicMock()})
        _stub_module("megatron.core.optimizer.distrib_optimizer", {"DistributedOptimizer": _DummyDistributedOptimizer})
        _stub_module(
            "megatron.core.optimizer.optimizer",
            {
                "ChainedOptimizer": _DummyChainedOptimizer,
                "MegatronOptimizer": _DummyOptimizer,
            },
        )
        _stub_module("megatron.core.optimizer_param_scheduler", {"OptimizerParamScheduler": _DummyScheduler})
        _stub_module("megatron.core.packed_seq_params", {"PackedSeqParams": MagicMock()})
        _stub_module("megatron.core.pipeline_parallel", {"get_forward_backward_func": MagicMock()})
        _stub_module("megatron.core.transformer", is_package=True)
        _stub_module("megatron.core.transformer.utils", {"sharded_state_dict_default": MagicMock()})
        _stub_module("megatron.core.utils", {"get_model_config": MagicMock()})
        _stub_module("megatron.core.config", {"set_experimental_flag": MagicMock()})
        _stub_module("megatron.core.num_microbatches_calculator", {"init_num_microbatches_calculator": MagicMock()})
        _stub_module("megatron.training", is_package=True)
        _stub_module(
            "megatron.training.global_vars",
            {
                "get_args": MagicMock(),
                "_build_tokenizer": MagicMock(),
                "set_args": MagicMock(),
            },
        )
        _stub_module("megatron.training.training", {"get_model": MagicMock()})
        _stub_module(
            "megatron.training.checkpointing",
            {
                "load_checkpoint": MagicMock(),
                "save_checkpoint": MagicMock(),
            },
        )
        _stub_module("sglang.srt.debug_utils", is_package=True)
        _stub_module(
            "sglang.srt.debug_utils.dumper",
            {
                "DumperConfig": MagicMock(),
                "_get_rank": MagicMock(return_value=0),
                "dumper": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.bridge_lora_helpers",
            {
                "_ensure_model_list": MagicMock(),
                "_setup_lora_model_via_bridge": MagicMock(),
            },
        )
        _stub_module("miles.backends.megatron_utils.model_provider", {"get_model_provider_func": MagicMock()})
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def _patch_initialize_side_effects(stack: ExitStack) -> None:
    stack.enter_context(patch("miles.backends.megatron_utils.model.clear_memory"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_peak_gpu_memory_after_load"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_model_hashes"))


def test_initialize_does_not_step_scheduler_restored_from_checkpoint():
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=True, global_batch_size=8)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (model, optimizer, opt_param_scheduler, 100)
    opt_param_scheduler.step.assert_not_called()


def test_initialize_steps_scheduler_when_checkpoint_did_not_restore_it():
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=False, global_batch_size=8)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (model, optimizer, opt_param_scheduler, 100)
    opt_param_scheduler.step.assert_called_once_with(increment=800)


_GLOBAL_BATCH_SIZE = 8
_CHECKPOINT_NUM_STEPS = 500
_CHECKPOINT_ITERATION = 100


class _FakeScheduler:
    def __init__(self) -> None:
        self.num_steps = 0

    def step(self, increment: int) -> None:
        self.num_steps += increment


def _accumulating_load_checkpoint(iteration: int):
    def _load(model, optimizer, opt_param_scheduler, **kwargs):
        if opt_param_scheduler is not None and iteration > 0:
            opt_param_scheduler.step(increment=_CHECKPOINT_NUM_STEPS)
        return iteration, 0

    return _load


def _load_state(scheduler: _FakeScheduler, *, use_checkpoint_scheduler: bool, iteration: int) -> int:
    from miles.backends.megatron_utils.model import load_model_state

    args = Namespace(use_checkpoint_opt_param_scheduler=use_checkpoint_scheduler, global_batch_size=_GLOBAL_BATCH_SIZE)
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.load_checkpoint",
                side_effect=_accumulating_load_checkpoint(iteration),
            )
        )
        _patch_initialize_side_effects(stack)
        return load_model_state(
            args,
            model=[_FakeModelChunk()],
            optimizer=object(),
            opt_param_scheduler=scheduler,
            role="actor",
        )


class TestTheSchedulerStateALoadLandsOn:
    def test_a_cold_load_keeps_the_checkpoints_own_steps_and_adds_the_iteration(self):
        """Megatron's load_state_dict accumulates, so this sum is the schedule every resume has always used."""
        scheduler = _FakeScheduler()

        _load_state(scheduler, use_checkpoint_scheduler=False, iteration=_CHECKPOINT_ITERATION)

        assert scheduler.num_steps == _CHECKPOINT_NUM_STEPS + _CHECKPOINT_ITERATION * _GLOBAL_BATCH_SIZE

    def test_a_reload_into_a_live_scheduler_lands_exactly_where_a_cold_load_does(self):
        """A hot restart reloads into the scheduler object init built, and must not drift from a cold restart."""
        cold = _FakeScheduler()
        _load_state(cold, use_checkpoint_scheduler=False, iteration=_CHECKPOINT_ITERATION)

        reloaded = _FakeScheduler()
        _load_state(reloaded, use_checkpoint_scheduler=False, iteration=_CHECKPOINT_ITERATION)
        _load_state(reloaded, use_checkpoint_scheduler=False, iteration=_CHECKPOINT_ITERATION)

        assert reloaded.num_steps == cold.num_steps

    def test_a_checkpoint_restored_schedule_is_not_stepped_a_second_time(self):
        """--use-checkpoint-opt-param-scheduler takes the schedule from the checkpoint and nothing else."""
        scheduler = _FakeScheduler()

        _load_state(scheduler, use_checkpoint_scheduler=True, iteration=_CHECKPOINT_ITERATION)

        assert scheduler.num_steps == _CHECKPOINT_NUM_STEPS

    def test_a_checkpoint_restored_schedule_does_not_double_on_a_reload(self):
        """This branch skips the step entirely, so only resetting before the load keeps a reload idempotent."""
        scheduler = _FakeScheduler()

        _load_state(scheduler, use_checkpoint_scheduler=True, iteration=_CHECKPOINT_ITERATION)
        _load_state(scheduler, use_checkpoint_scheduler=True, iteration=_CHECKPOINT_ITERATION)

        assert scheduler.num_steps == _CHECKPOINT_NUM_STEPS

    def test_a_run_that_loads_no_checkpoint_starts_its_schedule_at_zero(self):
        """A fresh run must not inherit a schedule position from anywhere."""
        scheduler = _FakeScheduler()

        _load_state(scheduler, use_checkpoint_scheduler=False, iteration=0)

        assert scheduler.num_steps == 0

    def test_the_cold_path_through_initialize_lands_on_the_same_value(self):
        """load_model_state is the load half of initialize, and extracting it must not move the cold path."""
        from miles.backends.megatron_utils.model import initialize_model_and_optimizer

        scheduler = _FakeScheduler()
        args = Namespace(use_checkpoint_opt_param_scheduler=False, global_batch_size=_GLOBAL_BATCH_SIZE)

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                    return_value=([_FakeModelChunk()], object(), scheduler),
                )
            )
            stack.enter_context(
                patch(
                    "miles.backends.megatron_utils.model.load_checkpoint",
                    side_effect=_accumulating_load_checkpoint(_CHECKPOINT_ITERATION),
                )
            )
            _patch_initialize_side_effects(stack)
            _, _, _, iteration = initialize_model_and_optimizer(args)

        assert iteration == _CHECKPOINT_ITERATION
        assert scheduler.num_steps == _CHECKPOINT_NUM_STEPS + _CHECKPOINT_ITERATION * _GLOBAL_BATCH_SIZE
