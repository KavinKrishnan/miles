from miles.utils.megatron_config import MegatronConfig, resolve_megatron_config

CRITIC_TRAINER_ROLE = "critic"
DEFAULT_TRAINER_ROLE = "actor"


def compute_trainer_role(config: MegatronConfig, model_id: str) -> str:
    return model_id if config.is_multi_policy else DEFAULT_TRAINER_ROLE


def compute_policy_trainer_roles(args) -> list[str]:
    config = resolve_megatron_config(args)
    return [compute_trainer_role(config, model_id) for model_id in config.model_ids]


def compute_trainer_roles(args) -> list[str]:
    return [*compute_policy_trainer_roles(args), *([CRITIC_TRAINER_ROLE] if args.use_critic else [])]


def compute_trainer_model_id(args, *, role: str) -> str | None:
    config = resolve_megatron_config(args)
    if role == CRITIC_TRAINER_ROLE or not config.is_multi_policy:
        return None
    return role


def compute_trainer_controller_pool_id(role: str) -> str:
    return f"trainer-controller-{role}"


def compute_trainer_pool_id(role: str) -> str:
    return f"trainer-engine-{role}"
