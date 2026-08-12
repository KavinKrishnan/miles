import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from miles.ray.deployment import run_deployment
from miles.ray.placement_group import create_rollout_components, maybe_start_api_server, update_weights
from miles.ray.specs.train import create_composite_trainer_controller
from miles.ray.train.composite import CompositeTrainerController
from miles.ray.train.multi_policy import (
    MultiPolicyCheckpointState,
    MultiPolicySaveCoordinator,
    assert_restored_rollout_ids,
    load_multi_policy_state,
    save_multi_policy_state,
)
from miles.ray.wiring import get_backend_capability, launch_worker_manager
from miles.utils import object_store
from miles.utils.arguments import parse_args, validate_async_off_policy_correction
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.data import remove_rollout_data_refs
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.megatron_config import MegatronConfig, compute_model_args, resolve_megatron_config
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils.tracking import define_step_key_metric_group, finish_tracking, init_tracking
from miles.utils.workers.worker_handle import BaseWorkerHandle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Policy:
    model_id: str
    start_rollout_id: int


class _MultiPolicyRun:
    def __init__(
        self,
        args,
        *,
        config: MegatronConfig,
        inference_controller,
        rollout_executor: BaseWorkerHandle,
        num_rollout_per_epoch: int | None,
        trainers: CompositeTrainerController,
        policies: list[_Policy],
    ) -> None:
        self.args = args
        self.config = config
        self.inference_controller = inference_controller
        self.trainers = trainers
        self.rollout_executor = rollout_executor
        self.num_rollout_per_epoch = num_rollout_per_epoch
        self.policies = policies
        self.coordinator = MultiPolicySaveCoordinator(
            model_ids=config.model_ids, primary_model_id=config.primary_model_id
        )
        self.saved_rollout_ids: dict[str, int] = {}

    async def run(self) -> None:
        tasks = [asyncio.create_task(self._run_policy(policy)) for policy in self.policies]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        for task in done:
            task.result()

    async def _run_policy(self, policy: _Policy) -> None:
        args = self.args
        last_rollout_id: int | None = None
        try:
            for rollout_id in range(policy.start_rollout_id, args.num_rollout):
                await self.inference_controller.prepare_rollout(rollout_id)
                rollout_data_ref = await self.rollout_executor.get(rollout_id, trainer_model_id=policy.model_id)
                await self.trainers.train(rollout_id, rollout_data_ref, model_id=policy.model_id)
                remove_rollout_data_refs(args, rollout_data_ref)
                last_rollout_id = rollout_id

                await self._maybe_save(policy, rollout_id)

                if (rollout_id + 1) % args.update_weights_interval == 0:
                    await update_weights(
                        args,
                        actor_model=self.trainers,
                        rollout_executor=self.rollout_executor,
                        inference_controller=self.inference_controller,
                        rollout_id=rollout_id,
                        model_id=policy.model_id,
                    )

                if (x := args.debug_exit_after_rollout) is not None and (
                    rollout_id - policy.start_rollout_id + 1
                ) >= x:
                    logger.info(f"debug_exit_after_rollout={x} reached at rollout_id={rollout_id}, exiting")
                    break

            await self._final_save(policy, last_rollout_id)
        finally:
            await asyncio.shield(self.coordinator.leave(policy.model_id))

    async def _maybe_save(self, policy: _Policy, rollout_id: int) -> None:
        args = self.args
        if args.save is None:
            return

        if policy.model_id != self.config.primary_model_id:
            if await self.coordinator.maybe_park(
                policy.model_id,
                rollout_id,
                lambda force_sync: self.trainers.save_model(
                    rollout_id, force_sync=force_sync, model_id=policy.model_id
                ),
            ):
                self.saved_rollout_ids[policy.model_id] = rollout_id
            return

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if not external_save and not should_run_periodic_action(
            rollout_id, args.save_interval, self.num_rollout_per_epoch, args.num_rollout
        ):
            return

        force_sync = external_save or rollout_id == args.num_rollout - 1
        async with self.coordinator.saving(rollout_id, force_sync=force_sync):
            await self.trainers.save_model(rollout_id, force_sync=force_sync, model_id=policy.model_id)
            self.saved_rollout_ids[policy.model_id] = rollout_id
            await self.rollout_executor.save(rollout_id)
            self._write_checkpoint_state()
        if external_save:
            os.remove(args.save_trigger_sentinel)

    async def _final_save(self, policy: _Policy, last_rollout_id: int | None) -> None:
        args = self.args
        if args.save is None or last_rollout_id is None:
            return
        if self.saved_rollout_ids.get(policy.model_id) == last_rollout_id:
            return

        async with self.coordinator.final_saving(policy.model_id, last_rollout_id):
            await self.trainers.save_model(last_rollout_id, force_sync=True, model_id=policy.model_id)
            self.saved_rollout_ids[policy.model_id] = last_rollout_id
            if policy.model_id == self.config.primary_model_id:
                await self.rollout_executor.save(last_rollout_id)
            self._write_checkpoint_state()

    def _write_checkpoint_state(self) -> None:
        rollout_ids = self.coordinator.rollout_ids
        primary_model_id = self.config.primary_model_id
        if primary_model_id not in rollout_ids:
            logger.warning(
                f"Not recording where the policies stand at {rollout_ids}: the primary model "
                f"{primary_model_id!r} never reached a checkpoint, so the record has no global rollout id to "
                f"be indexed by and a resume of this run will be refused"
            )
            return

        save_multi_policy_state(
            Path(self.args.save),
            MultiPolicyCheckpointState(
                primary_model_id=primary_model_id,
                rollout_ids=rollout_ids,
                finished_model_ids=self.coordinator.finished_model_ids,
            ),
        )


async def train_multi_policy(args) -> None:
    assert not args.colocate, "Colocation is not supported for multi policy training."
    assert args.fully_async, "Multi policy training is only supported for --fully-async"
    assert args.eval_interval is None, (
        "train_multi_policy.py does not evaluate: it has no eval dispatcher, so --eval-interval and the "
        "--eval-* arguments beside it would be accepted and never used. Drop them and read the per policy "
        "training curves instead."
    )
    validate_async_off_policy_correction(args)
    configure_logger(args, source=SimpleProcessIdentity(component="main"))
    maybe_start_periodic_pyspy_dump()
    init_tracking(args)
    config = resolve_megatron_config(args)
    define_policy_metric_groups(config)
    _worker_manager = launch_worker_manager(args)
    object_store.init_instance(args, contribute_segment=False)

    inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
    trainers = create_composite_trainer_controller(args, capability=get_backend_capability(args))

    try:
        await trainers.wait_ready()
        policies = await _create_policies(args, config=config, trainers=trainers)
        primary = policies[0]

        _assert_consistent_restore(args, config=config, policies=policies)
        await rollout_executor.set_train_parallel_config(
            await trainers.get_train_parallel_config(model_id=primary.model_id)
        )
        await rollout_executor.load(primary.start_rollout_id - 1)

        maybe_start_api_server(args, actor_model=trainers, inference_controller=inference_controller)
        maybe_start_mini_ft_controller(args)

        for policy in policies:
            await update_weights(
                args,
                actor_model=trainers,
                rollout_executor=rollout_executor,
                inference_controller=inference_controller,
                model_id=policy.model_id,
            )

        run = _MultiPolicyRun(
            args,
            config=config,
            inference_controller=inference_controller,
            rollout_executor=rollout_executor,
            num_rollout_per_epoch=num_rollout_per_epoch,
            trainers=trainers,
            policies=policies,
        )
        await run.run()
    finally:
        await rollout_executor.dispose()
        await inference_controller.dispose()
        await trainers.dispose()


def define_policy_metric_groups(config: MegatronConfig) -> None:
    if not config.is_multi_policy:
        return
    for model_id in config.model_ids:
        define_step_key_metric_group(prefix=model_id, step_key=f"{model_id}/rollout/step")
        define_step_key_metric_group(prefix=f"{model_id}/train", step_key=f"{model_id}/train/step")


async def _create_policies(args, *, config: MegatronConfig, trainers: CompositeTrainerController) -> list[_Policy]:
    ans: list[_Policy] = []
    for model_id in config.model_ids:
        model_args = compute_model_args(args, model_id)
        start_rollout_ids = await trainers.init(model_args, model_id=model_id)
        assert len(set(start_rollout_ids)) == 1, f"model {model_id} restored to {start_rollout_ids}"
        ans.append(_Policy(model_id=model_id, start_rollout_id=start_rollout_ids[0]))
    return ans


def _assert_consistent_restore(args, *, config: MegatronConfig, policies: list[_Policy]) -> None:
    primary_rollout_id = policies[0].start_rollout_id - 1
    if primary_rollout_id < 0:
        return

    state_dir = args.load or args.save
    if state_dir is None:
        return

    state = load_multi_policy_state(Path(state_dir), primary_rollout_id)
    assert state is not None or not config.is_multi_policy, (
        f"resuming a multi policy run at rollout {primary_rollout_id} but {state_dir} holds no record of "
        f"where the other policies stood; the checkpoint was not written by train_multi_policy.py, so the "
        f"policies cannot be proven to resume at consistent positions"
    )
    if state is None:
        return

    assert state.primary_model_id == config.primary_model_id, (
        f"the checkpoint was written with {state.primary_model_id!r} as the primary policy, but this run "
        f"makes {config.primary_model_id!r} primary; the global rollout index would change meaning"
    )
    assert_restored_rollout_ids(state, {p.model_id: p.start_rollout_id - 1 for p in policies})
    logger.info(f"Restored multi policy run at {state.rollout_ids} (primary {config.primary_model_id})")


if __name__ == "__main__":
    args = parse_args()
    try:
        run_deployment(args, run_orchestration_script=train_multi_policy)
    finally:
        finish_tracking()
