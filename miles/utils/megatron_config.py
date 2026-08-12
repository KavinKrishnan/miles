import copy
import functools
import logging
import re
import shlex
from argparse import Namespace
from pathlib import Path
from typing import Any

import pydantic
import yaml

from miles.utils.derived_args import apply_derived_args
from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)

DEFAULT_TRAINER_MODEL_ID = "default"

POLICY_CHECKPOINT_DIRNAME = "policies"

MODEL_ID_PATTERN = re.compile(r"\A[A-Za-z0-9_-]+\Z")

PER_POLICY_ARGS: dict[str, type] = {
    "hf_checkpoint": str,
    "ref_load": str,
    "megatron_to_hf_mode": str,
    "optimizer": str,
    "lr": float,
    "min_lr": float,
    "lr_decay_style": str,
    "lr_warmup_iters": int,
    "lr_warmup_fraction": float,
    "weight_decay": float,
    "adam_beta1": float,
    "adam_beta2": float,
    "clip_grad": float,
    "tensor_model_parallel_size": int,
    "pipeline_model_parallel_size": int,
    "context_parallel_size": int,
    "expert_model_parallel_size": int,
    "expert_tensor_parallel_size": int,
    "sequence_parallel": bool,
    "global_batch_size": int,
    "micro_batch_size": int,
    "max_tokens_per_gpu": int,
    "use_dynamic_batch_size": bool,
    "advantage_estimator": str,
    "use_kl_loss": bool,
    "kl_loss_coef": float,
    "kl_loss_type": str,
    "entropy_coef": float,
    "eps_clip": float,
    "eps_clip_high": float,
}


class _RawMegatronModelConfig(FrozenStrictBaseModel):
    name: str
    args: str = ""


class _RawMegatronConfig(FrozenStrictBaseModel):
    models: list[_RawMegatronModelConfig] = pydantic.Field(
        validation_alias=pydantic.AliasChoices("models", "megatron")
    )

    @classmethod
    def from_file_arg(cls, value: str) -> "_RawMegatronConfig":
        return cls.model_validate(yaml.safe_load(resolve_file_arg(value)))


class MegatronModelConfig(FrozenStrictBaseModel):
    name: str
    extra_args: list[str]

    @classmethod
    def resolve(cls, raw: _RawMegatronModelConfig) -> "MegatronModelConfig":
        return cls(name=raw.name, extra_args=shlex.split(raw.args))


class MegatronConfig(FrozenStrictBaseModel):
    models: list[MegatronModelConfig]

    @classmethod
    def resolve(cls, raw: _RawMegatronConfig) -> "MegatronConfig":
        models = [MegatronModelConfig.resolve(m) for m in raw.models]
        assert models, "--megatron-config must declare at least one model"
        names = [m.name for m in models]
        assert len(set(names)) == len(names), f"--megatron-config model names must be unique, got {names}"
        bad_names = [name for name in names if MODEL_ID_PATTERN.match(name) is None]
        assert not bad_names, (
            f"--megatron-config model names {bad_names} are not usable as path components: a model id names "
            f"this policy's checkpoint directory under --save and --load, so it must match "
            f"{MODEL_ID_PATTERN.pattern}"
        )
        return cls(models=models)

    @property
    def model_ids(self) -> list[str]:
        return [m.name for m in self.models]

    @property
    def primary_model_id(self) -> str:
        return self.models[0].name

    @property
    def is_multi_policy(self) -> bool:
        return len(self.models) > 1

    def get(self, model_id: str) -> MegatronModelConfig:
        for model in self.models:
            if model.name == model_id:
                return model
        raise KeyError(f"Unknown trainer model id {model_id!r}, known ids: {self.model_ids}")


def resolve_megatron_config(args) -> MegatronConfig:
    return _resolve_megatron_config_value(args.megatron_config)


@functools.cache
def _resolve_megatron_config_value(value: str | None) -> MegatronConfig:
    if value is None:
        raw = _RawMegatronConfig(models=[_RawMegatronModelConfig(name=DEFAULT_TRAINER_MODEL_ID)])
    else:
        raw = _RawMegatronConfig.from_file_arg(value)
    return MegatronConfig.resolve(raw)


def compute_model_args(args, model_id: str) -> Namespace:
    config = resolve_megatron_config(args)
    ans = copy.deepcopy(args)
    ans.trainer_model_id = model_id if config.is_multi_policy else None
    for key, value in compute_model_arg_overrides(config, model_id).items():
        assert hasattr(ans, key), (
            f"--megatron-config model {model_id!r} sets '--{key.replace('_', '-')}', which this run's "
            f"argument parser does not know"
        )
        setattr(ans, key, value)

    apply_derived_args(ans, model_id=model_id)

    if config.is_multi_policy:
        ans.save = compute_policy_checkpoint_dir(args.save, model_id)
        ans.load = compute_policy_checkpoint_dir(args.load, model_id)
    return ans


def compute_model_arg_overrides(config: MegatronConfig, model_id: str) -> dict[str, Any]:
    parsed = _parse_extra_args(config.get(model_id).extra_args)
    return {key: _coerce_value(value, key=key, model_id=model_id) for key, value in parsed.items()}


def compute_policy_checkpoint_dir(base_dir: str | None, model_id: str) -> str | None:
    if base_dir is None:
        return None
    return str(Path(base_dir) / POLICY_CHECKPOINT_DIRNAME / model_id)


def _parse_extra_args(tokens: list[str]) -> dict[str, str | list[str] | None]:
    ans: dict[str, str | list[str] | None] = {}
    key: str | None = None
    for token in tokens:
        if token.startswith("--"):
            body = token[2:]
            if "=" in body:
                name, _, value = body.partition("=")
                ans[name.replace("-", "_")] = value
                key = None
                continue
            key = body.replace("-", "_")
            ans[key] = None
            continue
        assert key is not None, f"--megatron-config args must start with a flag, got a bare value {token!r}"
        previous = ans[key]
        if previous is None:
            ans[key] = token
        elif isinstance(previous, list):
            previous.append(token)
        else:
            ans[key] = [previous, token]
    return ans


def _coerce_value(value: str | list[str] | None, *, key: str, model_id: str) -> Any:
    flag = f"--{key.replace('_', '-')}"
    expected = PER_POLICY_ARGS.get(key)
    assert expected is not None, (
        f"--megatron-config model {model_id!r} sets {flag!r}, which is not a per-policy argument; "
        f"only these may differ between policies: {sorted(PER_POLICY_ARGS)}. Everything else is read "
        f"from the base command line, so overriding it here would be silently ignored"
    )
    assert not isinstance(value, list), (
        f"--megatron-config model {model_id!r} passes {flag!r} several values {value}, "
        f"but a per-policy argument takes exactly one"
    )

    if expected is bool:
        return True if value is None else _parse_bool(value)
    assert value is not None, f"--megatron-config model {model_id!r} passes {flag!r} without a value"
    return expected(value)


def _parse_bool(value: str) -> bool:
    lowered = value.lower()
    assert lowered in ("true", "false", "1", "0"), f"cannot read {value!r} as a boolean"
    return lowered in ("true", "1")
