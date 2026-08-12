import dataclasses
import json
import os
import shlex

from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import MOONCAKE_MASTER_PORT
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.workers.types import ClusterBackend, DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_worker_host

register_cuda_ci(est_time=900, suite="stage-c-8-gpu-h100", labels=["short"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_TRAIN_GPUS = 2
NUM_ROLLOUT_GPUS = 2

RPC_PORT = 8000
ROUTER_PORT = 8000


def prepare():
    config = command_utils.default_config()
    if config.cluster_backend is not ClusterBackend.KUBERNETES:
        return

    U = config.create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    config = command_utils.default_config()
    if config.cluster_backend is not ClusterBackend.KUBERNETES:
        print(
            "Skipping: deploying the parts of a run separately is one release per part, "
            "which only the kubernetes backend installs"
        )
        return

    trainer_release = RunNames.release(run_id=config.run_id, deploy_component=DeployComponent.TRAINER)
    inference_release = RunNames.release(run_id=config.run_id, deploy_component=DeployComponent.INFERENCE)

    _launch(config, component=DeployComponent.TRAINER)
    _launch(config, component=DeployComponent.INFERENCE)
    try:
        _launch(config, component=DeployComponent.PRIMARY)
        _assert_still_installed(config, releases=[trainer_release, inference_release])
    finally:
        for release in (trainer_release, inference_release):
            Helm.uninstall(release=release, namespace=config.namespace)


def _assert_still_installed(config, *, releases: list[str]) -> None:
    missing = [release for release in releases if Helm.get_manifest(release, config.namespace) is None]
    assert not missing, (
        f"{missing} disappeared while the orchestration script ran: a release only ends when its own launch or its "
        f"own user uninstalls it, never because another release of the run finished or failed"
    )


def _launch(config, *, component: DeployComponent) -> None:
    config = dataclasses.replace(config, deploy_component=component)
    U = config.create_backend()
    U.execute_train(
        train_args=_train_args(config, component=component),
        num_gpus_per_node=NUM_TRAIN_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


def _train_args(config, *, component: DeployComponent) -> str:
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
        f"--rollout-num-gpus {NUM_ROLLOUT_GPUS} "
        "--rollout-num-gpus-per-engine 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    optimizer_args = "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_TRAIN_GPUS} "
        "--megatron-to-hf-mode bridge "
        "--api-server-port 0 "
    )

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{misc_args} "
        f"{_object_store_args(config, component=component)} "
        f"{_static_addr_args(config) if component is DeployComponent.PRIMARY else ''} "
    )


def _object_store_args(config, *, component: DeployComponent) -> str:
    if component is DeployComponent.PRIMARY:
        return command_utils.get_mooncake_object_store_args()

    primary_release = RunNames.release(run_id=config.run_id, deploy_component=DeployComponent.PRIMARY)
    master = MooncakeInfo.master_service_host(primary_release, config.namespace)
    init_kwargs = {
        "protocol": "tcp",
        "master_server_address": f"{master}:{MOONCAKE_MASTER_PORT}",
        "global_segment_size": "2gb",
        "local_buffer_size": "2gb",
    }
    return f"--object-store-backend mooncake --mooncake-store-init-kwargs {shlex.quote(json.dumps(init_kwargs))} "


def _static_addr_args(config) -> str:
    trainer_release = RunNames.release(run_id=config.run_id, deploy_component=DeployComponent.TRAINER)
    inference_release = RunNames.release(run_id=config.run_id, deploy_component=DeployComponent.INFERENCE)

    trainer_controller = static_worker_host(trainer_release, "trainer-controller-actor", 0)
    inference_controller = static_worker_host(inference_release, "inference-controller", 0)
    router = static_worker_host(inference_release, "inference-router-0", 0)

    return (
        f"--trainer-controller-addrs {trainer_controller}:{RPC_PORT} "
        f"--inference-controller-addrs {inference_controller}:{RPC_PORT} "
        f"--inference-router-addrs {router}:{ROUTER_PORT} "
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
