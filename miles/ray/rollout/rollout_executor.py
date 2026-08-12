import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.debug_data import RolloutDataInjectionUtil, load_debug_rollout_data, save_debug_rollout_data
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetSession
from miles.ray.rollout.metrics import log_eval_rollout_data, log_eval_skip, log_rollout_data
from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data
from miles.ray.rollout.router_manager import resolve_router_addrs, wait_session_server_ready
from miles.ray.rollout.train_data_conversion import (
    ROLLOUT_DATA_VALUE_SPEC,
    convert_samples_to_train_data,
    split_train_data_by_dp,
)
from miles.ray.specs.inference import inference_controller_worker_name
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnTrainInput,
    call_rollout_fn,
)
from miles.rollout.checkpoint_eval import CheckpointEvalFn, EvalSkip
from miles.rollout.inference_rollout.compatibility import call_rollout_function, load_rollout_function
from miles.rollout.router_addressing import compute_any_router_url
from miles.utils import object_store
from miles.utils.audit_utils.event_analyzer import analyzer as event_analyzer
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.environ import enable_experimental_rollout_refactor
from miles.utils.function_registry import load_function
from miles.utils.hf_config import is_complete_hf_export
from miles.utils.http_utils import init_http_client
from miles.utils.logging_utils import configure_logger
from miles.utils.megatron_config import resolve_megatron_config
from miles.utils.metric_checker import MetricChecker
from miles.utils.misc import NodeProbeMixin
from miles.utils.multi_lora import EmptyBatchTimeoutError
from miles.utils.timer import timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.weight_version import assert_samples_weight_version_sane
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


class RolloutExecutor(NodeProbeMixin):
    """The class to run rollout and convert rollout data to training data."""

    def __init__(
        self,
        *,
        args,
        router_providers: Sequence[BaseWorkerProvider],
        router_worker_names: Sequence[str] | None = None,
        session_server_provider: BaseWorkerProvider | None,
        inference_controller_provider: BaseWorkerProvider,
    ):
        event_logger_checkpoint.restore(args)
        configure_logger(args, source=SimpleProcessIdentity(component="rollout_executor"))

        self.args = args
        # set by the training actor after each weight update
        self.weight_version: int | None = None
        self._weight_versions: dict[str | None, int] = {}
        self._rollout_ids: dict[str | None, int] = {}
        self._multi_policy = resolve_megatron_config(args).is_multi_policy
        self._router_providers = router_providers
        self._router_worker_names = router_worker_names
        self._session_server_provider = session_server_provider
        self._inference_controller_provider = inference_controller_provider

    async def init(self) -> None:
        args = self.args
        if not args.debug_train_only:
            await resolve_router_addrs(
                args, router_providers=self._router_providers, router_worker_names=self._router_worker_names
            )
            await wait_session_server_ready(args, provider=self._session_server_provider)

        # TODO make args immutable
        init_tracking(args, primary=False, router_addr=compute_any_router_url(args))
        object_store.init_instance(args, contribute_segment=False)

        if not self.args.debug_train_only:
            init_http_client(args)

        data_source_cls = load_function(self.args.data_source_path)
        self.data_source = data_source_cls(args)

        self.use_experimental_refactor = enable_experimental_rollout_refactor()
        if self.use_experimental_refactor:
            input = RolloutFnConstructorInput(args=args, data_source=self.data_source)
            self.generate_rollout = load_rollout_function(input, self.args.rollout_function_path)
            if self.args.eval_function_path == self.args.rollout_function_path:
                # Reuse the instance so train and eval share one state (and stateful
                # rollout fns like FullyAsyncRolloutFn are not constructed twice).
                self.eval_generate_rollout = self.generate_rollout
            else:
                self.eval_generate_rollout = load_rollout_function(input, self.args.eval_function_path)
        else:
            self.generate_rollout = load_function(self.args.rollout_function_path)
            self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if (x := self.args.custom_reward_post_process_path) is not None:
            self.custom_reward_post_process_func = load_function(x)
        self.custom_convert_samples_to_train_data_func = None
        if (x := self.args.custom_convert_samples_to_train_data_path) is not None:
            self.custom_convert_samples_to_train_data_func = load_function(x)
        logger.info(f"import {self.args.rollout_function_path} as generate_rollout function.")
        logger.info(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        self.newest_rollout_id = -1
        self._eval_lock = asyncio.Lock()
        self._eval_fleet: EvalFleetSession | None = None

        self._metric_checker = MetricChecker.maybe_create(args)

    # -------------------------- lifecycle -----------------------------

    def dispose(self) -> None:
        if (close := getattr(self.data_source, "close", None)) is not None:
            close()
        event_analyzer.run_analysis_from_args(self.args)
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        if isinstance(self.eval_generate_rollout, CheckpointEvalFn):
            self.eval_generate_rollout.dispose()

    # -------------------------- data generation -----------------------------

    async def get(self, rollout_id: int, trainer_model_id: str | None = None) -> dict[str, Any]:
        start_time = time.time()
        metric_model_id = trainer_model_id if self._multi_policy else None
        self._rollout_ids[trainer_model_id] = rollout_id
        self.newest_rollout_id = max(self._rollout_ids.values())
        if (get_buffer_length := getattr(self.data_source, "get_buffer_length", None)) is not None:
            dashboard_hooks.report_data_buffer(get_buffer_length())
        with timer("rollout" if metric_model_id is None else f"rollout/{metric_model_id}"):
            try:
                data, metadata, metrics = await self._get_rollout_data(
                    rollout_id=rollout_id, trainer_model_id=trainer_model_id
                )
            except EmptyBatchTimeoutError as e:
                logger.warning(f"Rollout {rollout_id} produced no trainable group before the empty-wait timeout: {e}")
                return dict(sample_indices=None, data_ref=None, empty_batch_timeout=True)
        save_debug_rollout_data(
            self.args,
            data,
            rollout_id=rollout_id,
            evaluation=False,
            metadata=metadata,
            trainer_model_id=trainer_model_id,
        )
        log_rollout_data(
            rollout_id, self.args, data, metrics, time.time() - start_time, trainer_model_id=metric_model_id
        )
        data = convert_samples_to_train_data(
            self.args,
            data,
            metadata=metadata,
            custom_convert_samples_to_train_data_func=self.custom_convert_samples_to_train_data_func,
            custom_reward_post_process_func=self.custom_reward_post_process_func,
        )
        sample_indices = data.get("sample_indices")
        if self.args.delay_split_train_data_by_dp:
            data_ref = object_store.get_instance().put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        else:
            data_ref = split_train_data_by_dp(self.args, data, self.train_parallel_config)
        return dict(sample_indices=sample_indices, data_ref=data_ref, empty_batch_timeout=False)

    async def eval(
        self,
        rollout_id: int,
        hf_dir: str | None = None,
        export_time_seconds: float | None = None,
        require_marker: bool = True,
    ) -> None:
        if self.args.debug_train_only:
            # if debug train only, we don't generate evaluation data
            return

        if self.args.eval_uses_snapshots:
            return await self._eval_checkpoint(rollout_id, hf_dir, export_time_seconds, require_marker)

        with timer("eval_rollout"):
            if self.use_experimental_refactor:
                result = await asyncio.to_thread(
                    call_rollout_function, self.eval_generate_rollout, RolloutFnEvalInput(rollout_id=rollout_id)
                )
            else:
                result = await asyncio.to_thread(
                    call_rollout_fn,
                    self.eval_generate_rollout,
                    self.args,
                    rollout_id,
                    self.data_source,
                    evaluation=True,
                )
        data = result.data
        save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=True)
        metrics = log_eval_rollout_data(rollout_id, self.args, data, result.metrics)
        if self._metric_checker is not None:
            self._metric_checker.on_eval(metrics)

    async def _eval_checkpoint(
        self, rollout_id: int, hf_dir: str | None, export_time_seconds: float | None, require_marker: bool
    ):
        """Evaluate a snapshot through the checkpoint eval fn (fleet or external
        backend) and log at ``rollout_id``. Every failure degrades to a skipped
        point; the lock serializes pins against a single backend."""
        assert hf_dir is not None, "checkpoint eval requires an HF snapshot dir"
        start_time = time.time()
        async with self._eval_lock:
            if require_marker and not is_complete_hf_export(hf_dir):
                logger.warning(f"Eval snapshot {hf_dir} missing or incomplete, skipping eval {rollout_id}")
                return self.report_eval_skip(rollout_id, "ckpt_missing")

            version = str(rollout_id)
            try:
                state = await self._eval_fleet.pin(hf_dir, version) if self._eval_fleet else None
                eval_input = RolloutFnEvalInput(
                    rollout_id=rollout_id, weight_version=version, hf_dir=hf_dir, generate_state=state
                )
                result = await asyncio.to_thread(call_rollout_function, self.eval_generate_rollout, eval_input)
            except EvalSkip as e:
                return self.report_eval_skip(rollout_id, e.reason)

            data = result.data
            save_debug_rollout_data(self.args, data, rollout_id=rollout_id, evaluation=True)
            extra_metrics = dict(result.metrics or {})
            extra_metrics["eval/lag_steps"] = max(self.newest_rollout_id - rollout_id, 0)
            extra_metrics["eval/duration_seconds"] = time.time() - start_time
            if export_time_seconds is not None:
                extra_metrics["eval/export_time_seconds"] = export_time_seconds
            metrics = log_eval_rollout_data(rollout_id, self.args, data, extra_metrics)
            if self._metric_checker is not None:
                self._metric_checker.on_eval(metrics)

    def report_eval_skip(self, rollout_id: int, reason: str) -> None:
        log_eval_skip(rollout_id, self.args, reason)

    async def _get_rollout_data(self, rollout_id, trainer_model_id: str | None = None):
        if self.args.load_debug_rollout_data:
            data, metadata = load_debug_rollout_data(self.args, rollout_id=rollout_id)
            metrics = None
        else:
            if self.use_experimental_refactor:
                data = await asyncio.to_thread(
                    call_rollout_function,
                    self.generate_rollout,
                    RolloutFnTrainInput(
                        rollout_id=rollout_id,
                        weight_version=self._weight_versions.get(trainer_model_id, self.weight_version),
                        trainer_model_id=trainer_model_id,
                    ),
                )
            else:
                data = await asyncio.to_thread(
                    call_rollout_fn, self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False
                )
            metrics = data.metrics
            data = data.samples
            data, metadata = postprocess_rollout_data(
                self.args, data, train_parallel_config=self.train_parallel_config
            )
            assert_samples_weight_version_sane(self.args, samples=data)
            if RolloutDataInjectionUtil.should_inject(self.args, rollout_id):
                generated_data = data
                data, metadata = RolloutDataInjectionUtil.load(self.args, rollout_id=rollout_id)
                RolloutDataInjectionUtil.assert_matches_generated(
                    self.args, generated=generated_data, injected=data, rollout_id=rollout_id
                )
                metrics = None

        return data, metadata, metrics

    # -------------------------- checkpointing -----------------------------

    # TODO the train and eval rollout functions will become one object, so one save/load is enough here
    def save(self, rollout_id: int) -> None:
        self.data_source.save(rollout_id)
        if self.use_experimental_refactor:
            self.generate_rollout.save(rollout_id)
        event_logger_checkpoint.snapshot(self.args, rollout_id)

    def load(self, rollout_id: int | None = None) -> None:
        self.data_source.load(rollout_id)
        if self.use_experimental_refactor:
            self.generate_rollout.load(rollout_id)

    # -------------------------- misc APIs -----------------------------

    def get_num_rollout_per_epoch(self) -> int:
        assert self.args.rollout_global_dataset
        return len(self.data_source.dataset) // self.args.rollout_batch_size

    def set_weight_version(self, weight_version: int, trainer_model_id: str | None = None) -> None:
        # warning instead of assert when use indep_dp ft
        if (previous := self._weight_versions.get(trainer_model_id)) is not None and weight_version < previous:
            message = (
                f"Engine weight version of model {trainer_model_id} went backwards: {previous} -> {weight_version}"
            )
            assert self.args.indep_dp, message
            logger.warning(message)
        self._weight_versions[trainer_model_id] = weight_version
        if trainer_model_id is None:
            self.weight_version = weight_version

    def set_train_parallel_config(self, config: dict[str, Any]) -> None:
        self.train_parallel_config = config

    async def set_eval_fleet(self, eval_fleet: EvalFleetInfo | None) -> None:
        if eval_fleet is None:
            self._eval_fleet = None
            return

        inference_controller = await self._inference_controller_provider.get_handle_async(
            inference_controller_worker_name()
        )
        self._eval_fleet = EvalFleetSession(self.args, info=eval_fleet, inference_controller=inference_controller)
