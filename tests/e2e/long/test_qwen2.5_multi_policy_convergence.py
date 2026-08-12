import os
import tempfile

from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=7200,
    suite="stage-c-8-gpu-h100",
    labels=["long", "fully-async", "multi-policy"],
    nightly=True,
)

MODEL_A = "Qwen2.5-0.5B-Instruct"
MODEL_B = "Qwen2.5-1.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8

SGLANG_CONFIG_YAML = f"""\
sglang:
  - name: policy_a
    model_path: /root/models/{MODEL_A}
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 2
        num_gpus_per_engine: 1
  - name: policy_b
    model_path: /root/models/{MODEL_B}
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 2
        num_gpus_per_engine: 1
"""

MEGATRON_CONFIG_YAML = f"""\
megatron:
  - name: policy_a
    args: --hf-checkpoint /root/models/{MODEL_A} --ref-load /root/models/{MODEL_A} --lr 1e-6
  - name: policy_b
    args: --hf-checkpoint /root/models/{MODEL_B} --ref-load /root/models/{MODEL_B} --lr 5e-7
"""


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    for model_name in (MODEL_A, MODEL_B):
        U.exec_command_cpu(f"hf download Qwen/{model_name} --local-dir /root/models/{model_name}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def _write_config(payload: str, prefix: str) -> str:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix=prefix, delete=False)
    handle.write(payload)
    handle.flush()
    return handle.name


def execute():
    U = command_utils.default_config().create_backend()
    sglang_config_path = _write_config(SGLANG_CONFIG_YAML, "sglang_config_")
    megatron_config_path = _write_config(MEGATRON_CONFIG_YAML, "megatron_config_")

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_A}/ " f"--ref-load /root/models/{MODEL_A}/ "

    rollout_args = (
        "--fully-async "
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 60 "
        "--rollout-batch-size 16 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 128 "
        "--pause-generation-mode in_place "
        "--custom-generate-function-path examples.multi_policy.round_robin_generate.generate "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-enable-metrics "
        f"--sglang-config {sglang_config_path} "
        f"--megatron-config {megatron_config_path} "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 4 "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_multi_policy.py",
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
