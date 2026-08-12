from argparse import Namespace


def compute_use_critic(args: Namespace) -> bool:
    return args.advantage_estimator == "ppo"


def compute_global_batch_size(args: Namespace, *, model_id: str | None = None) -> int | None:
    if args.num_steps_per_rollout is None:
        return args.global_batch_size

    ans = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
    owner = "" if model_id is None else f"--megatron-config model {model_id!r}: "
    assert args.global_batch_size is None or args.global_batch_size == ans, (
        f"{owner}global_batch_size {args.global_batch_size} is not equal to "
        f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
        f"// num_steps_per_rollout {args.num_steps_per_rollout}"
    )
    return ans


def apply_derived_args(args: Namespace, *, model_id: str | None = None) -> None:
    args.use_critic = compute_use_critic(args)
    args.global_batch_size = compute_global_batch_size(args, model_id=model_id)
