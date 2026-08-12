import copy
import os
from argparse import Namespace
from dataclasses import dataclass, replace
from pathlib import Path

from miles.ray.specs.static_addrs import trainer_controller_urls
from miles.ray.specs.trainer_identity import (
    CRITIC_TRAINER_ROLE,
    compute_trainer_controller_pool_id,
    compute_trainer_pool_id,
    compute_trainer_role,
)
from miles.ray.train.composite import CompositeTrainerController
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.environ import default_fp8_block_scaling_fp32_scales
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.megatron_config import compute_model_args, resolve_megatron_config
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.types import DeployComponent, DeploySelector
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider
from miles.utils.workers.worker_spec import (
    MASTER_PORT_NAME,
    PortInfo,
    SchedulingSpec,
    ServeWorkerSpec,
    WorkerLaunchContext,
)

POOL_CATEGORY_TRAINER_ENGINE = "trainer_engine"

TRAINER_CONCURRENCY_GROUPS = {"heartbeat_status": 1, "default": 1, "fault_injector": 1, "kill_self": 1}

TRAINER_CONTROLLER_WORKER_CLASS = "miles.ray.train.group.TrainerController"

_TRAINER_ACTOR_CLASSES = {
    "megatron": "miles.backends.megatron_utils.actor.MegatronTrainRayActor",
    "fsdp": "miles.backends.fsdp_utils.actor.FSDPTrainRayActor",
}

_NUM_GPUS_PER_TRAINER_WORKER = 0.4


@dataclass(frozen=True)
class TrainerInstance:
    role: str
    args: Namespace
    num_nodes: int
    num_gpus_per_node: int
    gpu_offset: int
    with_ref: bool
    with_opd_teacher: bool

    @property
    def num_gpus(self) -> int:
        return self.num_nodes * self.num_gpus_per_node


def compute_trainer_instances(args) -> list[TrainerInstance]:
    config = resolve_megatron_config(args)

    ans: list[TrainerInstance] = []
    gpu_offset = 0
    for model_id in config.model_ids:
        model_args = compute_model_args(args, model_id) if config.is_multi_policy else args
        instance = TrainerInstance(
            role=compute_trainer_role(config, model_id),
            args=model_args,
            num_nodes=model_args.actor_num_nodes,
            num_gpus_per_node=model_args.actor_num_gpus_per_node,
            gpu_offset=gpu_offset,
            with_ref=model_args.kl_coef != 0 or model_args.use_kl_loss,
            with_opd_teacher=model_args.use_opd and model_args.opd_type == "megatron",
        )
        gpu_offset += instance.num_gpus
        ans.append(instance)

    if args.use_critic:
        ans.append(
            TrainerInstance(
                role=CRITIC_TRAINER_ROLE,
                args=compute_critic_args(args),
                num_nodes=args.critic_num_nodes,
                num_gpus_per_node=args.critic_num_gpus_per_node,
                gpu_offset=0,
                with_ref=False,
                with_opd_teacher=False,
            )
        )

    return ans


def compute_deployed_trainer_instances(args) -> list[TrainerInstance]:
    selector = DeploySelector.of(args)
    all_instances = compute_trainer_instances(args)
    selected = [
        instance for instance in all_instances if selector.selects(DeployComponent.TRAINER, instance=instance.role)
    ]
    assert selected, (
        f"--deploy-component {selector.value} names a trainer instance this run does not train; its trainers are "
        f"{[instance.role for instance in all_instances]}"
    )
    return _rebase_gpu_offsets(selected)


def _rebase_gpu_offsets(instances: list[TrainerInstance]) -> list[TrainerInstance]:
    ans: list[TrainerInstance] = []
    gpu_offset = 0
    for instance in instances:
        if instance.role == CRITIC_TRAINER_ROLE:
            ans.append(replace(instance, gpu_offset=0))
            continue
        ans.append(replace(instance, gpu_offset=gpu_offset))
        gpu_offset += instance.num_gpus
    return ans


def compute_trainer_gpu_budget(args) -> int:
    if not DeploySelector.of(args).component.selects(DeployComponent.TRAINER):
        return 0
    return sum(
        instance.num_gpus
        for instance in compute_deployed_trainer_instances(args)
        if instance.role != CRITIC_TRAINER_ROLE
    )


def specs_trainer_controller(args) -> list[ServeWorkerSpec]:
    return [_compute_spec_trainer_controller(instance) for instance in compute_deployed_trainer_instances(args)]


def create_composite_trainer_controller(args, *, capability: BackendCapability) -> CompositeTrainerController:
    config = resolve_megatron_config(args)
    return CompositeTrainerController(
        trainers={
            model_id: create_trainer_controller_handle(
                args, capability=capability, role=compute_trainer_role(config, model_id)
            )
            for model_id in config.model_ids
        }
    )


def create_trainer_controller_handle(args, *, capability: BackendCapability, role: str) -> BaseWorkerHandle:
    provider = compute_trainer_controller_provider(args, capability=capability, role=role)
    return provider.get_handle(trainer_controller_worker_name(role))


def compute_trainer_controller_provider(args, *, capability: BackendCapability, role: str) -> BaseWorkerProvider:
    pool_id = compute_trainer_controller_pool_id(role)
    if (urls := trainer_controller_urls(args, role=role)) is not None:
        return SimpleWorkerProvider.of_rpc_urls(
            pool_id=pool_id, urls=urls, worker_class=TRAINER_CONTROLLER_WORKER_CLASS
        )
    return capability.static_worker_provider(pool_id=pool_id)


def trainer_controller_worker_name(role: str) -> str:
    return compute_worker_name(pool_id=compute_trainer_controller_pool_id(role))


def trainer_controller_cell_id(role: str) -> str:
    return compute_cell_id(pool_id=compute_trainer_controller_pool_id(role), cell_index=0)


def _compute_spec_trainer_controller(instance: TrainerInstance) -> ServeWorkerSpec:
    role = instance.role
    return ServeWorkerSpec(
        name=compute_trainer_controller_pool_id(role),
        deploy_component=DeployComponent.TRAINER,
        deploy_instance=role,
        port_infos=[],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
        ),
        worker_class=TRAINER_CONTROLLER_WORKER_CLASS,
        ctor_kwargs=lambda ctx: dict(
            role=role,
            with_ref=instance.with_ref,
            with_opd_teacher=instance.with_opd_teacher,
            cell_provider=ctx.capability.dynamic_worker_provider(pool_ids=[compute_trainer_pool_id(role)]),
            cell_operations=ctx.capability.cell_operations(),
        ),
    )


def specs_trainer(args) -> list[ServeWorkerSpec]:
    return [_compute_spec_trainer(instance) for instance in compute_deployed_trainer_instances(args)]


def compute_trainer_num_cells(args, *, role: str) -> int:
    num_nodes, num_gpus_per_node = (
        (args.critic_num_nodes, args.critic_num_gpus_per_node)
        if role == CRITIC_TRAINER_ROLE
        else (args.actor_num_nodes, args.actor_num_gpus_per_node)
    )
    total_gpus = num_nodes * num_gpus_per_node
    return (total_gpus // compute_megatron_world_size_except_dp(args)) if args.indep_dp else 1


def compute_critic_args(args):
    critic_args = copy.deepcopy(args)
    critic_args.kl_coef = 0
    critic_args.use_opd = False
    critic_args.disable_param_buffers_cpu_backup = False
    return critic_args


def _compute_spec_trainer(instance: TrainerInstance) -> ServeWorkerSpec:
    args = instance.args
    role = instance.role
    num_gpus_per_node = instance.num_gpus_per_node
    total_gpus = instance.num_gpus
    num_cells = compute_trainer_num_cells(args, role=role)
    assert total_gpus % num_cells == 0, f"{total_gpus=} must be divisible by {num_cells=}"
    gpus_per_cell = total_gpus // num_cells

    return ServeWorkerSpec(
        name=compute_trainer_pool_id(role),
        category=POOL_CATEGORY_TRAINER_ENGINE,
        deploy_component=DeployComponent.TRAINER,
        deploy_instance=role,
        port_infos=[PortInfo(name=MASTER_PORT_NAME, static_port=9000, mode="master", allow_dynamic=True)],
        env_var=lambda ctx: compute_trainer_env_vars(args, ctx),
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=gpus_per_cell,
            num_gpus_per_worker=_NUM_GPUS_PER_TRAINER_WORKER,
            num_cpus_per_worker=_NUM_GPUS_PER_TRAINER_WORKER,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=num_gpus_per_node,
            pg_name="actor",
            pg_slot_offset=instance.gpu_offset,
        ),
        worker_class=_TRAINER_ACTOR_CLASSES[args.train_backend],
        ctor_kwargs=lambda ctx: dict(
            args=args,
            world_size=gpus_per_cell,
            rank=ctx.worker_in_cell_index,
            role=role,
            cell_index=ctx.cell_index,
        ),
        concurrency_groups=TRAINER_CONCURRENCY_GROUPS,
        meta=lambda ctx: dict(role=role, cell_index=ctx.cell_index),
    )


def compute_trainer_env_vars(args, ctx: WorkerLaunchContext) -> dict[str, str]:
    env_vars = {
        # because sglang will always set NCCL_CUMEM_ENABLE to 0
        # we need also set it to 0 to prevent nccl error.
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get(
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES", default_fp8_block_scaling_fp32_scales()
        ),
        # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
        "NVSHMEM_DISABLE_NCCL": os.environ.get("NVSHMEM_DISABLE_NCCL", "1"),
        **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
        **args.train_env_vars,
    }

    if source_patcher_config := args.dumper_source_patcher_config_train:
        env_vars["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config

    if args.offload_train and args.train_backend == "megatron":
        from torch_memory_saver.utils import get_binary_path_from_package

        dynlib_path = str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))

        env_vars["LD_PRELOAD"] = dynlib_path
        env_vars["TMS_INIT_ENABLE"] = "1"
        if args.offload_train_target == "disk":
            assert b"TMS_INIT_ENABLE_DISK_BACKUP" in Path(dynlib_path).read_bytes(), (
                f"{dynlib_path} has no disk backend; reinstall torch_memory_saver at the commit "
                f"docker/Dockerfile pins."
            )
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "0"
            env_vars["TMS_INIT_ENABLE_DISK_BACKUP"] = "1"
            env_vars["TMS_DISK_BACKUP_CHUNK_MB"] = str(args.offload_train_disk_chunk_mb)
            env_vars["TMS_DISK_BACKUP_DIR"] = os.path.join(
                args.offload_train_disk_dir, f"cell{ctx.cell_index}_rank{ctx.worker_in_cell_index}"
            )
        else:
            env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

    return env_vars
