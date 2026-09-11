from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import re
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import torch
import torch.distributed as dist
from ray.actor import ActorHandle

from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.misc import load_function
from miles.utils.types import ParamInfo

from .common import get_atomic_update_groups, get_named_update_units, named_params_and_buffers

MODELEXPRESS_LOGICAL_GROUP = "model"


def _qwen3_model_kind(args: Namespace, model_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", model_name.lower())
    if "qwen3" not in normalized or any(token in normalized for token in ("qwen3next", "qwen35", "qwen36")):
        return "unsupported"
    if "qwen3moe" in normalized or getattr(args, "num_experts", None):
        return "moe"
    return "dense"


@dataclasses.dataclass(frozen=True)
class ModelExpressTrainerRegistration:
    model_name: str
    worker_id: str
    cohort_id: str
    source_geometry: Mapping[str, int | None]
    rollout_workers: tuple[Mapping[str, object], ...]
    logical_groups: tuple[str, ...] = (MODELEXPRESS_LOGICAL_GROUP,)


@dataclasses.dataclass(frozen=True)
class ModelExpressAliasSpec:
    hf_name: str
    role: str
    global_shape: tuple[int, ...]
    shard_axis: int | None
    local_shard_range: tuple[int, int] | None


@dataclasses.dataclass(frozen=True)
class ModelExpressPublishedTensorSpec:
    native_name: str
    tensor: torch.Tensor
    hf_names: tuple[str, ...]
    role: str
    global_shape: tuple[int, ...]
    placement_kind: str
    shard_axis: int | None
    local_shard_range: tuple[int, int] | None
    tensor_model_parallel: bool
    partition_dim: int
    partition_stride: int
    parallel_mode: str | None
    source_rank: int
    aliases: tuple[ModelExpressAliasSpec, ...]
    conversion_metadata: Mapping[str, object]


@dataclasses.dataclass(frozen=True)
class ModelExpressPublishRequest:
    version: str
    training_step: int
    logical_group: str
    cohort_id: str
    worker_id: str
    source_geometry: Mapping[str, int | None]
    tensors: Sequence[ModelExpressPublishedTensorSpec]
    atomic_units: tuple[tuple[str, ...], ...]


@runtime_checkable
class ModelExpressPublisher(Protocol):
    """The only ModelExpress surface Miles needs on a trainer worker."""

    def configure(self, registration: ModelExpressTrainerRegistration) -> None: ...

    def prepare(self, tensors: Sequence[ModelExpressPublishedTensorSpec]) -> str: ...

    def create_version(self, *, source_slots: Sequence[str], step: int, update_id: str) -> str: ...

    def publish_and_execute(self, request: ModelExpressPublishRequest) -> None: ...

    def mark_ready(self, version_id: str) -> None: ...

    def retire(self, version_id: str) -> None: ...

    def release(self, version_id: str) -> None: ...


PublisherFactory = Callable[[], ModelExpressPublisher]


def validate_modelexpress_configuration(
    args: Namespace,
    *,
    model_name: str,
    quantization_config: Mapping | None,
    is_lora: bool,
) -> None:
    problems = []
    if getattr(args, "update_weight_transfer_mode", None) != "modelexpress":
        problems.append("--update-weight-transfer-mode must be modelexpress")
    if getattr(args, "colocate", False):
        problems.append("colocated training/rollout is not supported")
    if is_lora or getattr(args, "lora_rank", 0) > 0:
        problems.append("LoRA updates are not supported")
    if quantization_config is not None or getattr(args, "fp8", False):
        problems.append("FP8/quantized updates are not supported")
    if getattr(args, "fp16", False) or not getattr(args, "bf16", False):
        problems.append("only BF16 source weights are supported")
    if _qwen3_model_kind(args, model_name) == "unsupported":
        problems.append(f"only Qwen3 and Qwen3MoE models are supported (got {model_name!r})")
    if getattr(args, "update_weight_disk_dir", None) or getattr(args, "update_weight_local_checkpoint_dir", None):
        problems.append("disk fallback is not supported")
    if getattr(args, "multi_lora_n_adapters", 0):
        problems.append("multi-LoRA updates are not supported")
    if problems:
        raise ValueError("Invalid ModelExpress weight transfer configuration: " + "; ".join(problems))


def load_modelexpress_publisher_factory(adapter_path: str | None) -> PublisherFactory:
    """Load a narrow runtime adapter without making modelexpress a Miles dependency."""
    if adapter_path:
        try:
            factory = load_function(adapter_path)
        except (ImportError, AttributeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot load ModelExpress publisher adapter {adapter_path!r}. "
                "It must be an importable zero-argument factory implementing configure() "
                "and publish_and_execute()."
            ) from exc
        return factory

    try:
        module = importlib.import_module("modelexpress_rl.integrations.miles")
    except ImportError as exc:
        raise RuntimeError(
            "ModelExpress transfer was selected, but the 'modelexpress' package is not installed. "
            "Install a compatible runtime or set --modelexpress-publisher-adapter to a Miles adapter factory."
        ) from exc

    factory = getattr(module, "create_miles_publisher", None)
    if factory is None:
        raise RuntimeError(
            "The installed ModelExpress runtime is incompatible with the Miles publisher protocol; "
            "modelexpress_rl.integrations.miles.create_miles_publisher is missing."
        )
    return factory


def _dense_qwen3_hf_names(native_name: str) -> tuple[str, tuple[str, ...], str]:
    if native_name == "module.module.embedding.word_embeddings.weight":
        return "vocab", ("model.embed_tokens.weight",), "embedding"
    if native_name == "module.module.output_layer.weight":
        return "vocab", ("lm_head.weight",), "output"
    if native_name == "module.module.decoder.final_layernorm.weight":
        return "replicated", ("model.norm.weight",), "final_norm"

    match = re.fullmatch(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", native_name)
    if match is None:
        raise ValueError(f"Unsupported dense Qwen3 Megatron parameter for ModelExpress: {native_name}")
    layer, rest = match.groups()
    prefix = f"model.layers.{layer}"
    mappings = {
        "self_attention.linear_proj.weight": ("row", (f"{prefix}.self_attn.o_proj.weight",), "o_proj"),
        "mlp.linear_fc2.weight": ("row", (f"{prefix}.mlp.down_proj.weight",), "down_proj"),
        "self_attention.linear_qkv.layer_norm_weight": (
            "replicated",
            (f"{prefix}.input_layernorm.weight",),
            "input_norm",
        ),
        "input_layernorm.weight": ("replicated", (f"{prefix}.input_layernorm.weight",), "input_norm"),
        "mlp.linear_fc1.layer_norm_weight": (
            "replicated",
            (f"{prefix}.post_attention_layernorm.weight",),
            "post_attention_norm",
        ),
        "pre_mlp_layernorm.weight": (
            "replicated",
            (f"{prefix}.post_attention_layernorm.weight",),
            "post_attention_norm",
        ),
        "self_attention.q_layernorm.weight": (
            "replicated",
            (f"{prefix}.self_attn.q_norm.weight",),
            "q_norm",
        ),
        "self_attention.k_layernorm.weight": (
            "replicated",
            (f"{prefix}.self_attn.k_norm.weight",),
            "k_norm",
        ),
    }
    if rest == "self_attention.linear_qkv.weight":
        return (
            "qkv",
            (
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ),
            "qkv",
        )
    if rest == "mlp.linear_fc1.weight":
        return "gate_up", (f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight"), "gate_up"
    try:
        return mappings[rest]
    except KeyError as exc:
        raise ValueError(f"Unsupported dense Qwen3 Megatron parameter for ModelExpress: {native_name}") from exc


def _alias(
    hf_name: str,
    role: str,
    global_shape: tuple[int, ...],
    shard_axis: int | None,
    local_shard_range: tuple[int, int] | None,
) -> ModelExpressAliasSpec:
    return ModelExpressAliasSpec(
        hf_name=hf_name,
        role=role,
        global_shape=global_shape,
        shard_axis=shard_axis,
        local_shard_range=local_shard_range,
    )


def build_qwen3_published_tensor_spec(
    args: Namespace,
    info: ParamInfo,
    tensor: torch.Tensor,
    *,
    tp_rank: int,
    tp_size: int,
) -> ModelExpressPublishedTensorSpec:
    """Lower one local dense-Qwen3 Megatron shard to explicit HF alias geometry."""
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"ModelExpress requires BF16 tensors; {info.name} has {tensor.dtype}")
    if tuple(tensor.shape) != tuple(info.shape):
        raise ValueError(
            f"ModelExpress metadata shape mismatch for {info.name}: {tuple(info.shape)} != {tuple(tensor.shape)}"
        )

    attrs = info.attrs
    tensor_parallel = bool(attrs.get("tensor_model_parallel", False))
    parallel_mode = attrs.get("parallel_mode")
    if parallel_mode == "duplicated":
        tensor_parallel = False
    partition_dim = int(attrs.get("partition_dim", -1))
    partition_stride = int(attrs.get("partition_stride", 1))
    if tensor_parallel and not (0 <= partition_dim < tensor.ndim):
        raise ValueError(f"ModelExpress cannot register TP shard {info.name}: invalid partition_dim={partition_dim}")
    if partition_stride < 1:
        raise ValueError(f"ModelExpress cannot register {info.name}: invalid partition_stride={partition_stride}")

    native_global_shape = list(info.shape)
    local_range = None
    if tensor_parallel:
        native_global_shape[partition_dim] *= tp_size
        local_extent = int(info.shape[partition_dim])
        local_range = (tp_rank * local_extent, (tp_rank + 1) * local_extent)
    native_global_shape = tuple(int(dim) for dim in native_global_shape)

    placement_role, hf_names, role = _dense_qwen3_hf_names(info.name)
    placement_kind = "replicated"
    aliases: tuple[ModelExpressAliasSpec, ...]
    conversion_metadata: dict[str, object] = {}
    shard_axis = partition_dim if tensor_parallel else None

    if placement_role == "qkv":
        if not tensor_parallel or partition_dim != 0:
            raise ValueError(f"Dense Qwen3 QKV must be TP-sharded on axis 0: {info.name}")
        hidden_size = int(args.hidden_size)
        num_heads = int(args.num_attention_heads)
        num_query_groups = int(args.num_query_groups)
        head_dim = int(getattr(args, "kv_channels", None) or hidden_size // num_heads)
        if num_query_groups % tp_size or num_heads % tp_size:
            raise ValueError("Dense Qwen3 QKV heads/query groups must divide the trainer TP size")
        expected_shape = ((num_heads + 2 * num_query_groups) * head_dim, hidden_size)
        if native_global_shape != expected_shape:
            raise ValueError(
                f"Dense Qwen3 QKV global shape mismatch for {info.name}: "
                f"metadata gives {native_global_shape}, config requires {expected_shape}"
            )
        q_shape = (num_heads * head_dim, hidden_size)
        kv_shape = (num_query_groups * head_dim, hidden_size)
        q_extent = q_shape[0] // tp_size
        kv_extent = kv_shape[0] // tp_size
        aliases = (
            _alias(hf_names[0], "q", q_shape, 0, (tp_rank * q_extent, (tp_rank + 1) * q_extent)),
            _alias(hf_names[1], "k", kv_shape, 0, (tp_rank * kv_extent, (tp_rank + 1) * kv_extent)),
            _alias(hf_names[2], "v", kv_shape, 0, (tp_rank * kv_extent, (tp_rank + 1) * kv_extent)),
        )
        placement_kind = "qkv_grouped_tp"
        conversion_metadata = {
            "layout": "grouped_qkv",
            "projection_order": ("q", "k", "v"),
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_query_groups": num_query_groups,
            "head_dim": head_dim,
            "query_heads_per_group": num_heads // num_query_groups,
            "local_query_groups": num_query_groups // tp_size,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "partition_stride": partition_stride,
        }
    elif placement_role == "gate_up":
        if not tensor_parallel or partition_dim != 0 or partition_stride != 2:
            raise ValueError(f"Dense Qwen3 gate/up must be TP-sharded on axis 0 with partition_stride=2: {info.name}")
        hidden_size = int(args.hidden_size)
        ffn_hidden_size = int(args.ffn_hidden_size)
        expected_shape = (2 * ffn_hidden_size, hidden_size)
        if native_global_shape != expected_shape:
            raise ValueError(
                f"Dense Qwen3 gate/up global shape mismatch for {info.name}: "
                f"metadata gives {native_global_shape}, config requires {expected_shape}"
            )
        alias_extent = ffn_hidden_size // tp_size
        aliases = tuple(
            _alias(
                name,
                alias_role,
                (ffn_hidden_size, hidden_size),
                0,
                (tp_rank * alias_extent, (tp_rank + 1) * alias_extent),
            )
            for name, alias_role in zip(hf_names, ("gate", "up"), strict=True)
        )
        placement_kind = "gate_up_tp"
        conversion_metadata = {
            "layout": "gate_then_up_per_tp_rank",
            "split_dim": 0,
            "split_sizes": (ffn_hidden_size, ffn_hidden_size),
            "local_split_size": ffn_hidden_size // tp_size,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
        }
    elif placement_role == "vocab":
        if tensor_parallel and partition_dim != 0:
            raise ValueError(f"Dense Qwen3 vocabulary tensor must be TP-sharded on axis 0: {info.name}")
        vocab_size = int(args.vocab_size)
        if len(native_global_shape) != 2 or vocab_size > native_global_shape[0]:
            raise ValueError(
                f"Dense Qwen3 vocabulary geometry mismatch for {info.name}: "
                f"vocab_size={vocab_size}, source shape={native_global_shape}"
            )
        canonical_range = None
        if local_range is not None:
            canonical_range = (min(local_range[0], vocab_size), min(local_range[1], vocab_size))
        aliases = (
            _alias(
                hf_names[0],
                role,
                (vocab_size, native_global_shape[1]),
                0 if tensor_parallel else None,
                canonical_range,
            ),
        )
        if tensor_parallel:
            placement_kind = "contiguous_tp"
        conversion_metadata = {
            "layout": "padded_vocab",
            "padded_vocab_size": native_global_shape[0],
            "vocab_size": vocab_size,
        }
    else:
        if placement_role == "row" and (not tensor_parallel or partition_dim != 1):
            raise ValueError(f"Dense Qwen3 row-parallel tensor must be TP-sharded on axis 1: {info.name}")
        if placement_role == "replicated" and tensor_parallel:
            raise ValueError(f"Dense Qwen3 replicated tensor unexpectedly reports TP sharding: {info.name}")
        aliases = tuple(_alias(name, role, native_global_shape, shard_axis, local_range) for name in hf_names)
        if tensor_parallel:
            placement_kind = "strided_tp" if partition_stride > 1 else "contiguous_tp"

    return ModelExpressPublishedTensorSpec(
        native_name=info.name,
        tensor=tensor,
        hf_names=hf_names,
        role=role,
        global_shape=native_global_shape,
        placement_kind=placement_kind,
        shard_axis=shard_axis,
        local_shard_range=local_range,
        tensor_model_parallel=tensor_parallel,
        partition_dim=partition_dim,
        partition_stride=partition_stride,
        parallel_mode=parallel_mode,
        source_rank=info.src_rank,
        aliases=aliases,
        conversion_metadata=conversion_metadata,
    )


def _required_positive_int(args: Namespace, name: str) -> int:
    value = getattr(args, name, None)
    if value is None or int(value) <= 0:
        raise ValueError(f"Qwen3MoE ModelExpress publication requires positive args.{name}")
    return int(value)


def _validate_moe_local_shape(
    info: ParamInfo,
    tensor: torch.Tensor,
    global_shape: tuple[int, ...],
    *,
    shard_axis: int | None,
    rank: int,
    size: int,
    label: str,
) -> tuple[tuple[int, int] | None, tuple[int, ...]]:
    if shard_axis is None:
        expected_local = global_shape
        local_range = None
    else:
        if size < 1 or not 0 <= rank < size:
            raise ValueError(f"{label} has invalid parallel rank/size {rank}/{size}")
        if global_shape[shard_axis] % size:
            raise ValueError(f"{label} global shape {global_shape} does not divide parallel size {size}")
        expected_local_list = list(global_shape)
        expected_local_list[shard_axis] //= size
        expected_local = tuple(expected_local_list)
        extent = expected_local[shard_axis]
        local_range = (rank * extent, (rank + 1) * extent)
    if tuple(tensor.shape) != expected_local:
        raise ValueError(
            f"{label} local shape mismatch for {info.name}: "
            f"got {tuple(tensor.shape)}, expected {expected_local} from global shape {global_shape}"
        )
    return local_range, expected_local


def build_qwen3moe_published_tensor_spec(
    args: Namespace,
    info: ParamInfo,
    tensor: torch.Tensor,
    *,
    tp_rank: int,
    tp_size: int,
    ep_rank: int,
    ep_size: int,
    etp_rank: int,
    etp_size: int,
) -> ModelExpressPublishedTensorSpec:
    """Lower one Qwen3MoE shard while keeping EP and ETP geometry explicit."""
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"ModelExpress requires BF16 tensors; {info.name} has {tensor.dtype}")
    if tuple(tensor.shape) != tuple(info.shape):
        raise ValueError(
            f"ModelExpress metadata shape mismatch for {info.name}: {tuple(info.shape)} != {tuple(tensor.shape)}"
        )

    hidden_size = _required_positive_int(args, "hidden_size")
    num_experts = _required_positive_int(args, "num_experts")
    if ep_size < 1 or num_experts % ep_size:
        raise ValueError(f"Qwen3MoE num_experts={num_experts} must divide EP size {ep_size}")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"Qwen3MoE has invalid EP rank/size {ep_rank}/{ep_size}")

    attrs = info.attrs
    reported_tp = bool(attrs.get("tensor_model_parallel", False))
    parallel_mode = attrs.get("parallel_mode")
    if parallel_mode == "duplicated":
        reported_tp = False
    reported_dim = int(attrs.get("partition_dim", -1))
    reported_stride = int(attrs.get("partition_stride", 1))
    if reported_stride < 1:
        raise ValueError(f"ModelExpress cannot register {info.name}: invalid partition_stride={reported_stride}")

    if info.name == "module.module.embedding.word_embeddings.weight":
        placement, hf_names, role = "vocab", ("model.embed_tokens.weight",), "embedding"
        expected_global = (int(args.vocab_size), hidden_size)
    elif info.name == "module.module.output_layer.weight":
        placement, hf_names, role = "vocab", ("lm_head.weight",), "output"
        expected_global = (int(args.vocab_size), hidden_size)
    elif info.name == "module.module.decoder.final_layernorm.weight":
        placement, hf_names, role = "replicated", ("model.norm.weight",), "final_norm"
        expected_global = (hidden_size,)
    else:
        match = re.fullmatch(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", info.name)
        if match is None:
            raise ValueError(f"Unsupported Qwen3MoE Megatron parameter for ModelExpress: {info.name}")
        layer, rest = match.groups()
        prefix = f"model.layers.{layer}"
        expert = re.fullmatch(r"mlp\.experts\.(linear_fc[12])\.weight(\d+)", rest)
        if expert is not None:
            projection, expert_text = expert.groups()
            expert_idx = int(expert_text)
            experts_per_rank = num_experts // ep_size
            expected_expert_range = (ep_rank * experts_per_rank, (ep_rank + 1) * experts_per_rank)
            if not expected_expert_range[0] <= expert_idx < expected_expert_range[1]:
                raise ValueError(
                    f"Qwen3MoE routed expert name {info.name!r} cannot prove global expert identity: "
                    f"EP rank {ep_rank} owns global experts {expected_expert_range}"
                )
            moe_ffn = _required_positive_int(args, "moe_ffn_hidden_size")
            if projection == "linear_fc1":
                placement = "gate_up"
                hf_names = (
                    f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight",
                    f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight",
                )
                role = "expert_gate_up"
                expected_global = (2 * moe_ffn, hidden_size)
                shard_axis, expected_stride = 0, 2
            else:
                placement = "row"
                hf_names = (f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight",)
                role = "expert_down"
                expected_global = (hidden_size, moe_ffn)
                shard_axis, expected_stride = 1, 1
            if etp_size > 1 and not reported_tp:
                raise ValueError(f"Qwen3MoE routed {projection} is missing explicit ETP " f"metadata for {info.name}")
            if etp_size > 1 and reported_tp and (reported_dim != shard_axis or reported_stride != expected_stride):
                raise ValueError(
                    f"Qwen3MoE routed {projection} reports incompatible ETP metadata for {info.name}: "
                    f"partition_dim={reported_dim}, partition_stride={reported_stride}"
                )
            local_range, _ = _validate_moe_local_shape(
                info,
                tensor,
                expected_global,
                shard_axis=shard_axis if etp_size > 1 else None,
                rank=etp_rank,
                size=etp_size,
                label=f"Qwen3MoE routed {projection} ETP",
            )
            if placement == "gate_up":
                if moe_ffn % etp_size:
                    raise ValueError(f"Qwen3MoE routed expert width {moe_ffn} must divide ETP size {etp_size}")
                alias_extent = moe_ffn // etp_size
                aliases = tuple(
                    _alias(
                        name,
                        alias_role,
                        (moe_ffn, hidden_size),
                        0 if etp_size > 1 else None,
                        (etp_rank * alias_extent, (etp_rank + 1) * alias_extent) if etp_size > 1 else None,
                    )
                    for name, alias_role in zip(hf_names, ("gate", "up"), strict=True)
                )
                conversion_metadata = {
                    "layout": "gate_then_up_per_tp_rank",
                    "split_dim": 0,
                    "split_sizes": (moe_ffn, moe_ffn),
                    "local_split_size": moe_ffn // etp_size,
                    "tp_rank": etp_rank,
                    "tp_size": etp_size,
                }
                placement_kind = "gate_up_etp"
            else:
                aliases = (_alias(hf_names[0], role, expected_global, 1 if etp_size > 1 else None, local_range),)
                conversion_metadata = {"layout": "contiguous_etp", "etp_rank": etp_rank, "etp_size": etp_size}
                placement_kind = "contiguous_etp" if etp_size > 1 else "replicated"
            return ModelExpressPublishedTensorSpec(
                native_name=info.name,
                tensor=tensor,
                hf_names=hf_names,
                role="gate_up" if placement == "gate_up" else role,
                global_shape=expected_global,
                placement_kind=placement_kind,
                shard_axis=shard_axis if etp_size > 1 else None,
                local_shard_range=local_range,
                tensor_model_parallel=etp_size > 1,
                partition_dim=shard_axis,
                partition_stride=expected_stride,
                parallel_mode=parallel_mode,
                source_rank=info.src_rank,
                aliases=aliases,
                conversion_metadata=conversion_metadata,
            )

        if rest.startswith("mlp.experts."):
            raise ValueError(
                f"Qwen3MoE routed expert name {info.name!r} cannot prove global expert identity; "
                "expected linear_fc1.weight<N> or linear_fc2.weight<N>"
            )
        shared = re.fullmatch(r"mlp\.shared_experts\.(.+)", rest)
        if shared is not None:
            shared_rest = shared.group(1)
            shared_ffn = _required_positive_int(args, "moe_shared_expert_intermediate_size")
            if shared_rest == "linear_fc1.weight":
                placement = "gate_up"
                hf_names = (
                    f"{prefix}.mlp.shared_expert.gate_proj.weight",
                    f"{prefix}.mlp.shared_expert.up_proj.weight",
                )
                role = "shared_gate_up"
                expected_global = (2 * shared_ffn, hidden_size)
            elif shared_rest == "linear_fc2.weight":
                placement = "row"
                hf_names = (f"{prefix}.mlp.shared_expert.down_proj.weight",)
                role = "shared_down"
                expected_global = (hidden_size, shared_ffn)
            elif shared_rest == "gate_weight":
                placement = "replicated"
                hf_names = (f"{prefix}.mlp.shared_expert_gate.weight",)
                role = "shared_expert_gate"
                expected_global = (1, hidden_size)
            else:
                raise ValueError(f"Unsupported Qwen3MoE shared expert parameter for ModelExpress: {info.name}")
        else:
            dense_ffn = _required_positive_int(args, "ffn_hidden_size")
            mapping = {
                "self_attention.linear_proj.weight": (
                    "row",
                    (f"{prefix}.self_attn.o_proj.weight",),
                    "o_proj",
                    (
                        hidden_size,
                        _required_positive_int(args, "num_attention_heads")
                        * int(args.kv_channels or hidden_size // _required_positive_int(args, "num_attention_heads")),
                    ),
                ),
                "mlp.linear_fc1.weight": (
                    "gate_up",
                    (f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight"),
                    "dense_gate_up",
                    (2 * dense_ffn, hidden_size),
                ),
                "mlp.linear_fc2.weight": (
                    "row",
                    (f"{prefix}.mlp.down_proj.weight",),
                    "dense_down",
                    (hidden_size, dense_ffn),
                ),
                "self_attention.linear_qkv.layer_norm_weight": (
                    "replicated",
                    (f"{prefix}.input_layernorm.weight",),
                    "input_norm",
                    (hidden_size,),
                ),
                "mlp.linear_fc1.layer_norm_weight": (
                    "replicated",
                    (f"{prefix}.post_attention_layernorm.weight",),
                    "post_attention_norm",
                    (hidden_size,),
                ),
                "pre_mlp_layernorm.weight": (
                    "replicated",
                    (f"{prefix}.post_attention_layernorm.weight",),
                    "post_attention_norm",
                    (hidden_size,),
                ),
                "self_attention.q_layernorm.weight": (
                    "replicated",
                    (f"{prefix}.self_attn.q_norm.weight",),
                    "q_norm",
                    (int(args.kv_channels or hidden_size // int(args.num_attention_heads)),),
                ),
                "self_attention.k_layernorm.weight": (
                    "replicated",
                    (f"{prefix}.self_attn.k_norm.weight",),
                    "k_norm",
                    (int(args.kv_channels or hidden_size // int(args.num_attention_heads)),),
                ),
                "mlp.router.weight": (
                    "replicated",
                    (f"{prefix}.mlp.gate.weight",),
                    "router",
                    (num_experts, hidden_size),
                ),
                "mlp.router.expert_bias": (
                    "replicated",
                    (f"{prefix}.mlp.gate.e_score_correction_bias",),
                    "expert_bias",
                    (num_experts,),
                ),
            }
            if rest in ("self_attention.linear_qkv.weight", "self_attention.linear_qkv.bias"):
                num_heads = _required_positive_int(args, "num_attention_heads")
                num_query_groups = _required_positive_int(args, "num_query_groups")
                head_dim = int(args.kv_channels or hidden_size // num_heads)
                suffix = "weight" if rest.endswith("weight") else "bias"
                tail = (hidden_size,) if suffix == "weight" else ()
                placement = "qkv"
                hf_names = tuple(f"{prefix}.self_attn.{name}_proj.{suffix}" for name in ("q", "k", "v"))
                role = "qkv"
                expected_global = ((num_heads + 2 * num_query_groups) * head_dim, *tail)
            else:
                try:
                    placement, hf_names, role, expected_global = mapping[rest]
                except KeyError as exc:
                    raise ValueError(f"Unsupported Qwen3MoE Megatron parameter for ModelExpress: {info.name}") from exc

    if placement == "vocab":
        if tensor.ndim != 2 or int(tensor.shape[1]) != hidden_size:
            raise ValueError(
                f"Qwen3MoE vocabulary geometry mismatch for {info.name}: "
                f"got {tuple(tensor.shape)}, expected hidden size {hidden_size}"
            )
        if reported_tp and (reported_dim != 0 or reported_stride != 1):
            raise ValueError(f"Qwen3MoE vocabulary tensor must be TP-sharded on axis 0: {info.name}")
        if tp_size > 1 and not reported_tp:
            raise ValueError(f"Qwen3MoE vocabulary tensor is missing TP metadata: {info.name}")
        padded_rows = int(tensor.shape[0]) * (tp_size if reported_tp else 1)
        configured_padded_rows = getattr(args, "padded_vocab_size", None)
        vocab_size = int(args.vocab_size)
        # Megatron Bridge may materialize the exact HF vocabulary even when
        # argument validation computed a larger TP padding target. The live
        # tensor is authoritative when it contains exactly the real vocabulary.
        if (
            configured_padded_rows is not None
            and int(configured_padded_rows) != padded_rows
            and padded_rows != vocab_size
        ):
            raise ValueError(
                f"Qwen3MoE padded vocabulary mismatch for {info.name}: "
                f"source has {padded_rows}, args.padded_vocab_size={configured_padded_rows}"
            )
        if not 0 < vocab_size <= padded_rows:
            raise ValueError(
                f"Qwen3MoE vocabulary size {vocab_size} exceeds source rows {padded_rows} for {info.name}"
            )
        native_range = None
        canonical_range = None
        shard_axis = None
        if reported_tp:
            shard_axis = 0
            local_rows = int(tensor.shape[0])
            native_range = (tp_rank * local_rows, (tp_rank + 1) * local_rows)
            canonical_range = (min(native_range[0], vocab_size), min(native_range[1], vocab_size))
        native_global_shape = (padded_rows, hidden_size)
        aliases = (_alias(hf_names[0], role, (vocab_size, hidden_size), shard_axis, canonical_range),)
        return ModelExpressPublishedTensorSpec(
            native_name=info.name,
            tensor=tensor,
            hf_names=hf_names,
            role=role,
            global_shape=native_global_shape,
            placement_kind="contiguous_tp" if reported_tp else "replicated",
            shard_axis=shard_axis,
            local_shard_range=native_range,
            tensor_model_parallel=reported_tp,
            partition_dim=reported_dim,
            partition_stride=reported_stride,
            parallel_mode=parallel_mode,
            source_rank=info.src_rank,
            aliases=aliases,
            conversion_metadata={
                "layout": "padded_vocab",
                "padded_vocab_size": padded_rows,
                "vocab_size": vocab_size,
            },
        )

    shard_axis = None
    rank = tp_rank
    size = tp_size
    expected_stride = 1
    if placement in ("qkv", "vocab", "gate_up"):
        shard_axis = 0
        expected_stride = 2 if placement == "gate_up" else 1
    elif placement == "row":
        shard_axis = 1
    if placement == "replicated":
        if reported_tp:
            raise ValueError(f"Qwen3MoE replicated tensor unexpectedly reports TP sharding: {info.name}")
        active_axis = None
    else:
        if size > 1 and (not reported_tp or reported_dim != shard_axis or reported_stride != expected_stride):
            raise ValueError(
                f"Qwen3MoE {placement} tensor has incompatible TP metadata for {info.name}: "
                f"partition_dim={reported_dim}, partition_stride={reported_stride}"
            )
        active_axis = shard_axis if size > 1 else None
    local_range, _ = _validate_moe_local_shape(
        info, tensor, expected_global, shard_axis=active_axis, rank=rank, size=size, label=f"Qwen3MoE {placement} TP"
    )

    conversion_metadata: dict[str, object] = {}
    if placement == "qkv":
        num_heads = int(args.num_attention_heads)
        num_query_groups = int(args.num_query_groups)
        head_dim = int(args.kv_channels or hidden_size // num_heads)
        if num_heads % num_query_groups:
            raise ValueError("Qwen3MoE attention heads must divide evenly into query groups")
        if num_heads % tp_size or num_query_groups % tp_size:
            raise ValueError("Qwen3MoE QKV heads/query groups must divide the trainer TP size")
        tail = (hidden_size,) if tensor.ndim == 2 else ()
        alias_shapes = (
            (num_heads * head_dim, *tail),
            (num_query_groups * head_dim, *tail),
            (num_query_groups * head_dim, *tail),
        )
        aliases = tuple(
            _alias(
                name,
                alias_role,
                shape,
                0,
                (
                    tp_rank * (shape[0] // tp_size),
                    (tp_rank + 1) * (shape[0] // tp_size),
                ),
            )
            for name, alias_role, shape in zip(hf_names, ("q", "k", "v"), alias_shapes, strict=True)
        )
        conversion_metadata = {
            "layout": "grouped_qkv",
            "head_dim": head_dim,
            "local_query_groups": num_query_groups // tp_size,
            "query_heads_per_group": num_heads // num_query_groups,
        }
        placement_kind = "qkv_grouped_tp"
    elif placement == "gate_up":
        shared_ffn = expected_global[0] // 2
        if shared_ffn % tp_size:
            raise ValueError(f"Qwen3MoE dense/shared width {shared_ffn} must divide TP size {tp_size}")
        alias_extent = shared_ffn // tp_size
        aliases = tuple(
            _alias(
                name,
                alias_role,
                (shared_ffn, hidden_size),
                0 if tp_size > 1 else None,
                (tp_rank * alias_extent, (tp_rank + 1) * alias_extent) if tp_size > 1 else None,
            )
            for name, alias_role in zip(hf_names, ("gate", "up"), strict=True)
        )
        conversion_metadata = {
            "layout": "gate_then_up_per_tp_rank",
            "split_dim": 0,
            "split_sizes": (shared_ffn, shared_ffn),
            "local_split_size": shared_ffn // tp_size,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
        }
        placement_kind = "gate_up_tp"
    elif placement == "vocab":
        aliases = (_alias(hf_names[0], role, expected_global, active_axis, local_range),)
        placement_kind = "contiguous_tp" if active_axis is not None else "replicated"
    else:
        aliases = (_alias(hf_names[0], role, expected_global, active_axis, local_range),)
        placement_kind = "contiguous_tp" if active_axis is not None else "replicated"

    return ModelExpressPublishedTensorSpec(
        native_name=info.name,
        tensor=tensor,
        hf_names=hf_names,
        role="gate_up" if placement == "gate_up" else role,
        global_shape=expected_global,
        placement_kind=placement_kind,
        shard_axis=active_axis,
        local_shard_range=local_range,
        tensor_model_parallel=active_axis is not None,
        partition_dim=reported_dim,
        partition_stride=reported_stride,
        parallel_mode=parallel_mode,
        source_rank=info.src_rank,
        aliases=aliases,
        conversion_metadata=conversion_metadata,
    )


def _actor_identity(engine: ActorHandle, index: int) -> str:
    actor_id = getattr(engine, "_actor_id", None)
    if actor_id is not None:
        value = actor_id.hex() if callable(getattr(actor_id, "hex", None)) else str(actor_id)
    else:
        value = getattr(engine, "worker_id", None)
        if value is None:
            raise RuntimeError(
                f"Cannot derive a stable ModelExpress identity for rollout engine {index}; "
                "the engine handle has neither a Ray actor id nor worker_id."
            )
    return str(value)


def build_rollout_cohort(
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None,
    engine_gpu_offsets: Sequence[int] | None,
) -> tuple[str, tuple[Mapping[str, object], ...]]:
    counts = list(engine_gpu_counts or [1] * len(rollout_engines))
    offsets = list(engine_gpu_offsets or range(len(rollout_engines)))
    if len(counts) != len(rollout_engines) or len(offsets) != len(rollout_engines):
        raise ValueError("ModelExpress rollout engine geometry does not match the exact engine set")
    workers = tuple(
        {
            "worker_id": _actor_identity(engine, index),
            "engine_index": index,
            "gpu_count": counts[index],
            "gpu_offset": offsets[index],
        }
        for index, engine in enumerate(rollout_engines)
    )
    encoded = json.dumps(workers, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), workers


def _source_geometry() -> dict[str, int | None]:
    ps = get_parallel_state()
    effective_dp = ps.effective_dp
    expert_dp_rank = None
    expert_dp_size = None
    try:
        from megatron.core import parallel_state as megatron_parallel_state

        rank_getter = getattr(megatron_parallel_state, "get_expert_data_parallel_rank", None)
        size_getter = getattr(megatron_parallel_state, "get_expert_data_parallel_world_size", None)
        if rank_getter is not None:
            expert_dp_rank = rank_getter()
        if size_getter is not None:
            expert_dp_size = size_getter()
    except (ImportError, RuntimeError, AssertionError):
        # Older Megatron versions do not expose expert-DP coordinates. Keep
        # them explicit and unknown instead of guessing a rank mapping.
        pass
    geometry = {
        "global_rank": dist.get_rank(group=get_gloo_group()),
        "tp_rank": ps.tp.rank,
        "tp_size": ps.tp.size,
        "pp_rank": ps.pp.rank,
        "pp_size": ps.pp.size,
        "ep_rank": ps.ep.rank,
        "ep_size": ps.ep.size,
        "etp_rank": ps.etp.rank,
        "etp_size": ps.etp.size,
        "dp_rank": effective_dp.rank,
        "dp_size": effective_dp.size,
        "expert_dp_rank": expert_dp_rank,
        "expert_dp_size": expert_dp_size,
    }
    return geometry


def _trainer_worker_id(geometry: Mapping[str, int | None]) -> str:
    del geometry
    # Worker IDs cross the gRPC/Kubernetes backend boundary and must be valid
    # label values. Full parallel coordinates remain in source_geometry.
    return f"megatron-rank-{dist.get_rank(group=get_gloo_group())}"


class UpdateWeightFromModelExpress:
    """Full-model BF16 Qwen3/Qwen3MoE refit through per-worker ModelExpress clients."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: Mapping | None,
        is_lora: bool = False,
        publisher_factory: PublisherFactory | None = None,
        param_infos: Sequence[ParamInfo] | None = None,
    ) -> None:
        del weights_getter
        validate_modelexpress_configuration(
            args, model_name=model_name, quantization_config=quantization_config, is_lora=is_lora
        )
        self.args = args
        self.model = model
        self.model_name = model_name
        self.model_kind = _qwen3_model_kind(args, model_name)
        self.weight_version = 0
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale = False
        self.source_geometry = _source_geometry()
        if self.model_kind == "moe" and self.source_geometry["expert_dp_size"] != 1:
            raise ValueError(
                "Qwen3MoE ModelExpress publication requires expert-DP size 1; "
                f"resolved expert_dp_size={self.source_geometry['expert_dp_size']!r}"
            )
        self.worker_id = _trainer_worker_id(self.source_geometry)
        # Dense data-parallel replicas expose identical shards. Publishing all
        # replicas would produce overlapping destination writes and ambiguous
        # source membership, so only ordinary DP rank zero participates.
        if self.model_kind == "moe":
            # Ordinary DP includes the EP dimension in Megatron. Every EP rank
            # owns distinct routed experts, so deduplicate only across expert-DP
            # replicas (expert-DP > 1 is rejected above until version ownership
            # is implemented).
            self._publishes = int(self.source_geometry["expert_dp_rank"]) == 0
        else:
            self._publishes = int(self.source_geometry["dp_rank"]) == 0
        self.publisher: ModelExpressPublisher | None = None
        if self._publishes:
            factory = publisher_factory or load_modelexpress_publisher_factory(
                getattr(args, "modelexpress_publisher_adapter", None)
            )
            self.publisher = factory()
            if not isinstance(self.publisher, ModelExpressPublisher):
                raise TypeError(
                    "ModelExpress publisher adapter must implement configure() " "and publish_and_execute()"
                )
        if param_infos is None:
            from .hf_weight_iterator_direct import _get_megatron_local_param_infos

            param_infos = _get_megatron_local_param_infos(args, model)
        self._param_info_by_name = {}
        for info in param_infos:
            normalized_name = info.name.replace(".to_wrap.", ".")
            if normalized_name in self._param_info_by_name:
                raise ValueError(f"Duplicate ModelExpress ParamInfo after name normalization: {normalized_name}")
            self._param_info_by_name[normalized_name] = dataclasses.replace(info, name=normalized_name)

    def is_rollout_engines_fresh(self) -> bool:
        return self.rollout_engines is not None and not self._connection_stale

    def mark_engine_connection_stale(self) -> None:
        self._connection_stale = True

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        del rollout_engine_lock
        self.rollout_engines = tuple(rollout_engines)
        self.cohort_id, self.rollout_workers = build_rollout_cohort(
            self.rollout_engines, engine_gpu_counts, engine_gpu_offsets
        )
        if self._publishes:
            assert self.publisher is not None
            self.publisher.configure(
                ModelExpressTrainerRegistration(
                    model_name=self.args.modelexpress_model_name,
                    worker_id=self.worker_id,
                    cohort_id=self.cohort_id,
                    source_geometry=self.source_geometry,
                    rollout_workers=self.rollout_workers,
                )
            )
        self._connection_stale = False

    def _receiver_payload(self, version_id: str) -> dict[str, object]:
        """Build only fields accepted by the upstream MX SGLang worker.

        Cohort identity and Miles' string weight version remain trainer-side
        metadata; the current receiver contract cannot verify them.
        """
        return {
            "target_training_step": self.weight_version,
            "version_id": version_id,
            "logical_group": MODELEXPRESS_LOGICAL_GROUP,
        }

    def _published_tensors_and_units(
        self,
    ) -> tuple[list[ModelExpressPublishedTensorSpec], tuple[tuple[str, ...], ...]]:
        named_tensors = [
            (name.replace(".to_wrap.", "."), tensor)
            for name, tensor in named_params_and_buffers(self.args, self.model)
        ]
        # EP replicas remain advertised so MX can select an owner already serving
        # this receiver's experts. The receiver deduplicates each exact shard.
        missing = [name for name, _tensor in named_tensors if name not in self._param_info_by_name]
        if missing:
            raise RuntimeError(f"ModelExpress is missing ParamInfo metadata for trainer tensors: {missing}")
        units = get_named_update_units(
            [name for name, _tensor in named_tensors],
            get_atomic_update_groups(self.args, self.model_name),
        )
        if self.model_kind == "moe":
            tensors = [
                build_qwen3moe_published_tensor_spec(
                    self.args,
                    self._param_info_by_name[name],
                    tensor,
                    tp_rank=int(self.source_geometry["tp_rank"]),
                    tp_size=int(self.source_geometry["tp_size"]),
                    ep_rank=int(self.source_geometry["ep_rank"]),
                    ep_size=int(self.source_geometry["ep_size"]),
                    etp_rank=int(self.source_geometry["etp_rank"]),
                    etp_size=int(self.source_geometry["etp_size"]),
                )
                for name, tensor in named_tensors
            ]
        else:
            tensors = [
                build_qwen3_published_tensor_spec(
                    self.args,
                    self._param_info_by_name[name],
                    tensor,
                    tp_rank=int(self.source_geometry["tp_rank"]),
                    tp_size=int(self.source_geometry["tp_size"]),
                )
                for name, tensor in named_tensors
            ]
        return tensors, tuple(unit.names for unit in units)

    @torch.no_grad()
    def update_weights(self) -> None:
        from miles.backends.megatron_utils.update_weight.modelexpress_round import run_update

        run_update(self)

    def pop_metrics(self) -> dict[str, float]:
        return {}
