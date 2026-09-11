import argparse
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.update_weight import modelexpress as mx
from miles.backends.megatron_utils.update_weight import modelexpress_round as mxround
from miles.utils.arguments import get_miles_extra_args_provider
from miles.utils.types import ParamInfo


def _args(**overrides):
    values = dict(
        update_weight_transfer_mode="modelexpress",
        colocate=False,
        lora_rank=0,
        multi_lora_n_adapters=0,
        fp8=False,
        fp16=False,
        bf16=True,
        update_weight_disk_dir=None,
        update_weight_local_checkpoint_dir=None,
        modelexpress_publisher_adapter=None,
        modelexpress_model_name="Qwen/Qwen3-0.6B",
        modelexpress_update_timeout=17.0,
        pause_generation_mode="retract",
        q_lora_rank=None,
        num_experts=None,
        megatron_to_hf_mode="raw",
        sglang_speculative_algorithm=None,
        mtp_num_layers=None,
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        ffn_hidden_size=16,
        moe_ffn_hidden_size=12,
        moe_shared_expert_intermediate_size=10,
        vocab_size=20,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_modelexpress_is_a_distinct_transfer_mode():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args, _unknown = parser.parse_known_args(
        ["--rollout-batch-size", "1", "--update-weight-transfer-mode", "modelexpress"]
    )
    assert args.update_weight_transfer_mode == "modelexpress"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"colocate": True}, "colocated"),
        ({"lora_rank": 8}, "LoRA"),
        ({"bf16": False}, "BF16"),
        ({"fp8": True}, "FP8"),
        ({"update_weight_disk_dir": "/tmp/fallback"}, "disk fallback"),
    ],
)
def test_modelexpress_validation_rejects_out_of_scope_modes(override, message):
    with pytest.raises(ValueError, match=message):
        mx.validate_modelexpress_configuration(
            _args(**override),
            model_name="qwen3config",
            quantization_config=None,
            is_lora=False,
        )


def test_modelexpress_validation_rejects_unsupported_model():
    with pytest.raises(ValueError, match="only Qwen3"):
        mx.validate_modelexpress_configuration(
            _args(),
            model_name="llamaconfig",
            quantization_config=None,
            is_lora=False,
        )


def test_modelexpress_validation_allows_qwen3moe_but_rejects_later_variants():
    mx.validate_modelexpress_configuration(
        _args(num_experts=8),
        model_name="qwen3moeconfig",
        quantization_config=None,
        is_lora=False,
    )
    for model_name in ("Qwen3NextConfig", "Qwen3_5Config", "Qwen3.6Config"):
        with pytest.raises(ValueError, match="only Qwen3"):
            mx.validate_modelexpress_configuration(
                _args(),
                model_name=model_name,
                quantization_config=None,
                is_lora=False,
            )


def test_runtime_import_has_no_fallback(monkeypatch):
    def fail_import(name):
        raise ImportError(name)

    monkeypatch.setattr(mx.importlib, "import_module", fail_import)
    with pytest.raises(RuntimeError, match="not installed"):
        mx.load_modelexpress_publisher_factory(None)


def test_concrete_modelexpress_adapter_is_loaded_by_default(monkeypatch):
    def factory():
        return object()

    module = SimpleNamespace(create_miles_publisher=factory)
    monkeypatch.setattr(mx.importlib, "import_module", lambda name: module)

    assert mx.load_modelexpress_publisher_factory(None) is factory


class _RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _Engine:
    def __init__(self, worker_id, events):
        self.worker_id = worker_id
        self.pause_generation = _RemoteMethod(lambda **kw: events.append(("pause", kw)))
        self.flush_cache = _RemoteMethod(lambda **kw: events.append(("flush", kw)))
        self.begin_weight_update = _RemoteMethod(lambda **kw: events.append(("begin", kw)))
        self.update_weights_from_modelexpress = _RemoteMethod(
            lambda **kw: events.append(("receiver", kw))
            or {
                "success": True,
                "workers": [
                    {
                        "rank": 0,
                        "response": {
                            "success": True,
                            "version_id": kw["payload"]["version_id"],
                            "installed_training_step": kw["payload"]["target_training_step"],
                        },
                    }
                ],
                "target_training_step": kw["payload"]["target_training_step"],
                "installed_training_step": kw["payload"]["target_training_step"],
                "layout_signature": "test-layout",
                "metrics": {},
                "timing": {},
                "error": None,
                "receiver_poisoned": False,
            }
        )
        self.end_weight_update = _RemoteMethod(lambda **kw: events.append(("end", kw)))
        self.update_weight_version = _RemoteMethod(lambda **kw: events.append(("version", kw)))
        self.continue_generation = _RemoteMethod(lambda **kw: events.append(("continue", kw)))


class _Publisher:
    def __init__(self, events):
        self.events = events
        self.registrations = []
        self.requests = []

    def configure(self, registration):
        self.registrations.append(registration)

    def prepare(self, tensors):
        return "slot-0"

    def create_version(self, *, source_slots, step, update_id):
        return f"mx-{step}"

    def mark_ready(self, version_id):
        self.events.append(("ready", version_id))

    def retire(self, version_id):
        self.events.append(("retire", version_id))

    def release(self, version_id):
        self.events.append(("release", version_id))

    def publish_and_execute(self, request):
        self.events.append(("publisher", request))
        self.requests.append(request)


def _info(name, shape, *, tp=False, dim=-1, stride=1, mode=None, source_rank=7):
    return ParamInfo(
        name=name,
        dtype=torch.bfloat16,
        shape=torch.Size(shape),
        attrs={
            "tensor_model_parallel": tp,
            "partition_dim": dim,
            "partition_stride": stride,
            "parallel_mode": mode,
        },
        size=torch.empty(shape, dtype=torch.bfloat16).numel() * 2,
        src_rank=source_rank,
    )


def test_qwen3_qkv_alias_geometry_is_explicit():
    name = "module.module.decoder.layers.3.self_attention.linear_qkv.weight"
    spec = mx.build_qwen3_published_tensor_spec(
        _args(), _info(name, (8, 8), tp=True, dim=0), torch.empty((8, 8), dtype=torch.bfloat16), tp_rank=1, tp_size=2
    )

    assert spec.hf_names == (
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
    )
    assert spec.global_shape == (16, 8)
    assert spec.placement_kind == "qkv_grouped_tp"
    assert [alias.global_shape for alias in spec.aliases] == [(8, 8), (4, 8), (4, 8)]
    assert [alias.local_shard_range for alias in spec.aliases] == [(4, 8), (2, 4), (2, 4)]
    assert spec.conversion_metadata["num_query_groups"] == 2
    assert spec.conversion_metadata["head_dim"] == 2


def test_qwen3_gate_up_alias_geometry_is_explicit():
    name = "module.module.decoder.layers.3.mlp.linear_fc1.weight"
    spec = mx.build_qwen3_published_tensor_spec(
        _args(),
        _info(name, (16, 8), tp=True, dim=0, stride=2),
        torch.empty((16, 8), dtype=torch.bfloat16),
        tp_rank=1,
        tp_size=2,
    )

    assert spec.hf_names == (
        "model.layers.3.mlp.gate_proj.weight",
        "model.layers.3.mlp.up_proj.weight",
    )
    assert spec.global_shape == (32, 8)
    assert spec.placement_kind == "gate_up_tp"
    assert [alias.global_shape for alias in spec.aliases] == [(16, 8), (16, 8)]
    assert [alias.local_shard_range for alias in spec.aliases] == [(8, 16), (8, 16)]
    assert spec.partition_stride == 2
    assert spec.conversion_metadata["layout"] == "gate_then_up_per_tp_rank"


def test_qwen3_row_replicated_and_tp_geometry():
    args = _args()
    row_name = "module.module.decoder.layers.0.self_attention.linear_proj.weight"
    row = mx.build_qwen3_published_tensor_spec(
        args,
        _info(row_name, (8, 4), tp=True, dim=1),
        torch.empty((8, 4), dtype=torch.bfloat16),
        tp_rank=1,
        tp_size=2,
    )
    norm_name = "module.module.decoder.final_layernorm.weight"
    replicated = mx.build_qwen3_published_tensor_spec(
        args, _info(norm_name, (8,)), torch.empty(8, dtype=torch.bfloat16), tp_rank=1, tp_size=2
    )
    vocab_name = "module.module.embedding.word_embeddings.weight"
    vocab = mx.build_qwen3_published_tensor_spec(
        args,
        _info(vocab_name, (10, 8), tp=True, dim=0),
        torch.empty((10, 8), dtype=torch.bfloat16),
        tp_rank=1,
        tp_size=2,
    )

    assert (row.global_shape, row.shard_axis, row.local_shard_range, row.role) == ((8, 8), 1, (4, 8), "o_proj")
    assert replicated.placement_kind == "replicated"
    assert replicated.shard_axis is None and replicated.local_shard_range is None
    assert (vocab.global_shape, vocab.shard_axis, vocab.local_shard_range) == ((20, 8), 0, (10, 20))


def _moe_spec(
    name, shape, *, tp=False, dim=-1, stride=1, tp_rank=0, tp_size=2, ep_rank=1, ep_size=2, etp_rank=1, etp_size=2
):
    return mx.build_qwen3moe_published_tensor_spec(
        _args(num_experts=8),
        _info(name, shape, tp=tp, dim=dim, stride=stride),
        torch.empty(shape, dtype=torch.bfloat16),
        tp_rank=tp_rank,
        tp_size=tp_size,
        ep_rank=ep_rank,
        ep_size=ep_size,
        etp_rank=etp_rank,
        etp_size=etp_size,
    )


def test_qwen3moe_routed_gate_up_uses_global_expert_and_etp_geometry():
    spec = _moe_spec(
        "module.module.decoder.layers.2.mlp.experts.linear_fc1.weight5",
        (12, 8),
        tp=True,
        dim=0,
        stride=2,
    )

    assert spec.hf_names == (
        "model.layers.2.mlp.experts.5.gate_proj.weight",
        "model.layers.2.mlp.experts.5.up_proj.weight",
    )
    assert spec.global_shape == (24, 8)
    assert spec.shard_axis == 0
    assert spec.local_shard_range == (12, 24)
    assert [alias.global_shape for alias in spec.aliases] == [(12, 8), (12, 8)]
    assert [alias.local_shard_range for alias in spec.aliases] == [(6, 12), (6, 12)]
    assert spec.conversion_metadata["tp_rank"] == 1
    assert spec.conversion_metadata["tp_size"] == 2


def test_qwen3moe_routed_down_uses_etp_row_geometry():
    spec = _moe_spec(
        "module.module.decoder.layers.2.mlp.experts.linear_fc2.weight6",
        (8, 6),
        tp=True,
        dim=1,
    )

    assert spec.hf_names == ("model.layers.2.mlp.experts.6.down_proj.weight",)
    assert spec.global_shape == (8, 12)
    assert spec.shard_axis == 1
    assert spec.local_shard_range == (6, 12)
    assert spec.aliases[0].local_shard_range == (6, 12)


def test_qwen3moe_etp1_ignores_inert_expert_partition_attributes():
    spec = _moe_spec(
        "module.module.decoder.layers.2.mlp.experts.linear_fc1.weight5",
        (24, 8),
        tp=True,
        dim=0,
        stride=1,
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
        etp_rank=0,
        etp_size=1,
    )

    assert spec.global_shape == (24, 8)
    assert spec.shard_axis is None
    assert spec.placement_kind == "gate_up_etp"


def test_qwen3moe_output_projection_uses_query_head_width():
    name = "module.module.decoder.layers.0.self_attention.linear_proj.weight"
    spec = mx.build_qwen3moe_published_tensor_spec(
        _args(num_experts=8, num_attention_heads=8, num_query_groups=2, kv_channels=2),
        _info(name, (8, 16), tp=True, dim=1),
        torch.empty((8, 16), dtype=torch.bfloat16),
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
        etp_rank=0,
        etp_size=1,
    )

    assert spec.global_shape == (8, 16)
    assert spec.aliases[0].global_shape == (8, 16)


def test_qwen3moe_tp1_qkv_aliases_expose_full_fragmented_ranges():
    name = "module.module.decoder.layers.0.self_attention.linear_qkv.weight"
    spec = mx.build_qwen3moe_published_tensor_spec(
        _args(num_experts=8),
        _info(name, (16, 8), tp=True, dim=0),
        torch.empty((16, 8), dtype=torch.bfloat16),
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
        etp_rank=0,
        etp_size=1,
    )

    assert [alias.shard_axis for alias in spec.aliases] == [0, 0, 0]
    assert [alias.local_shard_range for alias in spec.aliases] == [
        (0, 8),
        (0, 4),
        (0, 4),
    ]


def test_qwen3moe_shared_expert_uses_ordinary_tp_geometry():
    fc1 = _moe_spec(
        "module.module.decoder.layers.1.mlp.shared_experts.linear_fc1.weight",
        (10, 8),
        tp=True,
        dim=0,
        stride=2,
        tp_rank=1,
        etp_rank=0,
    )
    fc2 = _moe_spec(
        "module.module.decoder.layers.1.mlp.shared_experts.linear_fc2.weight",
        (8, 5),
        tp=True,
        dim=1,
        tp_rank=1,
        etp_rank=0,
    )
    gate = _moe_spec(
        "module.module.decoder.layers.1.mlp.shared_experts.gate_weight",
        (1, 8),
        tp_rank=1,
        etp_rank=0,
    )

    assert [alias.local_shard_range for alias in fc1.aliases] == [(5, 10), (5, 10)]
    assert fc2.local_shard_range == (5, 10)
    assert gate.hf_names == ("model.layers.1.mlp.shared_expert_gate.weight",)
    assert gate.global_shape == (1, 8)


def test_qwen3moe_router_and_expert_bias_are_validated_and_replicated():
    router = _moe_spec(
        "module.module.decoder.layers.0.mlp.router.weight",
        (8, 8),
        tp_size=1,
        ep_rank=0,
        etp_rank=0,
        etp_size=1,
    )
    bias = _moe_spec(
        "module.module.decoder.layers.0.mlp.router.expert_bias",
        (8,),
        tp_size=1,
        ep_rank=0,
        etp_rank=0,
        etp_size=1,
    )

    assert router.hf_names == ("model.layers.0.mlp.gate.weight",)
    assert bias.hf_names == ("model.layers.0.mlp.gate.e_score_correction_bias",)
    assert router.shard_axis is None and bias.shard_axis is None


def test_qwen3moe_vocab_accepts_bridge_tensor_without_argument_padding():
    name = "module.module.embedding.word_embeddings.weight"
    spec = mx.build_qwen3moe_published_tensor_spec(
        _args(num_experts=8, vocab_size=20, padded_vocab_size=22),
        _info(name, (10, 8), tp=True, dim=0),
        torch.empty((10, 8), dtype=torch.bfloat16),
        tp_rank=1,
        tp_size=2,
        ep_rank=0,
        ep_size=1,
        etp_rank=1,
        etp_size=2,
    )

    assert spec.global_shape == (20, 8)
    assert spec.local_shard_range == (10, 20)
    assert spec.aliases[0].global_shape == (20, 8)


def test_qwen3moe_vocab_rejects_rows_matching_neither_vocab_nor_padding():
    name = "module.module.embedding.word_embeddings.weight"
    with pytest.raises(ValueError, match="padded vocabulary mismatch"):
        mx.build_qwen3moe_published_tensor_spec(
            _args(num_experts=8, vocab_size=20, padded_vocab_size=22),
            _info(name, (12, 8), tp=True, dim=0),
            torch.empty((12, 8), dtype=torch.bfloat16),
            tp_rank=0,
            tp_size=2,
            ep_rank=0,
            ep_size=1,
            etp_rank=0,
            etp_size=2,
        )


@pytest.mark.parametrize(
    "name",
    [
        "module.module.decoder.layers.2.mlp.experts.linear_fc1.weight1",
        "module.module.decoder.layers.2.mlp.experts.linear_fc1.weight",
    ],
)
def test_qwen3moe_rejects_unprovable_global_expert_name(name):
    with pytest.raises(ValueError, match="cannot prove global expert identity"):
        _moe_spec(name, (12, 8))


def test_qwen3moe_rejects_wrong_etp_geometry():
    with pytest.raises(ValueError, match="local shape mismatch"):
        _moe_spec(
            "module.module.decoder.layers.2.mlp.experts.linear_fc1.weight5",
            (24, 8),
            tp=True,
            dim=0,
            stride=2,
        )


def test_qwen3moe_requires_explicit_etp_metadata():
    with pytest.raises(ValueError, match="missing explicit ETP metadata"):
        _moe_spec(
            "module.module.decoder.layers.2.mlp.experts.linear_fc1.weight5",
            (12, 8),
        )


@pytest.fixture
def runtime(monkeypatch):
    def group(rank=0, size=1):
        return SimpleNamespace(rank=rank, size=size, group=None)

    parallel_state = SimpleNamespace(
        tp=group(1, 2),
        pp=group(0, 2),
        ep=group(3, 4),
        etp=group(0, 1),
        effective_dp=group(0, 8),
    )
    monkeypatch.setattr(mx, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(mx, "named_params_and_buffers", lambda args, model: [])
    monkeypatch.setattr(mx, "get_gloo_group", lambda: None)
    monkeypatch.setattr(mxround, "get_gloo_group", lambda: None)
    monkeypatch.setattr(mx.dist, "get_rank", lambda *args, **kwargs: 0)
    monkeypatch.setattr(mx.dist, "get_world_size", lambda *args, **kwargs: 1)
    monkeypatch.setattr(mx.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mx.dist,
        "all_gather_object",
        lambda output, value, **kwargs: output.__setitem__(0, value),
    )
    monkeypatch.setattr(mxround.ray, "get", lambda value, **kwargs: value)


def test_qwen3moe_expert_data_parallel_fails_closed(runtime, monkeypatch):
    monkeypatch.setattr(
        mx,
        "_source_geometry",
        lambda: {
            "global_rank": 0,
            "tp_rank": 0,
            "tp_size": 1,
            "pp_rank": 0,
            "pp_size": 1,
            "ep_rank": 0,
            "ep_size": 1,
            "etp_rank": 0,
            "etp_size": 1,
            "dp_rank": 0,
            "dp_size": 2,
            "expert_dp_rank": 0,
            "expert_dp_size": 2,
        },
    )

    with pytest.raises(ValueError, match="expert-DP size 1"):
        mx.UpdateWeightFromModelExpress(
            _args(num_experts=8),
            model=[],
            weights_getter=lambda: {},
            model_name="qwen3moeconfig",
            quantization_config=None,
            publisher_factory=lambda: (_ for _ in ()).throw(AssertionError("must fail before publisher creation")),
            param_infos=[],
        )


def test_qwen3moe_ep_rank_advertises_experts_and_shared_replicas(runtime, monkeypatch):
    geometry = {
        "global_rank": 1,
        "tp_rank": 0,
        "tp_size": 1,
        "pp_rank": 0,
        "pp_size": 1,
        "ep_rank": 1,
        "ep_size": 2,
        "etp_rank": 0,
        "etp_size": 1,
        "dp_rank": 0,
        "dp_size": 1,
        "expert_dp_rank": 0,
        "expert_dp_size": 1,
    }
    expert_name = "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight4"
    router_name = "module.module.decoder.layers.0.mlp.router.weight"
    expert = torch.empty((24, 8), dtype=torch.bfloat16)
    router = torch.empty((8, 8), dtype=torch.bfloat16)
    monkeypatch.setattr(mx, "_source_geometry", lambda: geometry)
    monkeypatch.setattr(
        mx, "named_params_and_buffers", lambda args, model: [(expert_name, expert), (router_name, router)]
    )
    monkeypatch.setattr(mx, "get_atomic_update_groups", lambda args, model_name: ())
    monkeypatch.setattr(
        mx,
        "get_named_update_units",
        lambda names, groups: [SimpleNamespace(names=(name,)) for name in names],
    )
    publisher = _Publisher([])
    updater = mx.UpdateWeightFromModelExpress(
        _args(num_experts=8),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3moeconfig",
        quantization_config=None,
        publisher_factory=lambda: publisher,
        param_infos=[
            _info(expert_name, (24, 8)),
            _info(router_name, (8, 8)),
        ],
    )

    tensors, units = updater._published_tensors_and_units()

    assert [spec.native_name for spec in tensors] == [expert_name, router_name]
    assert units == ((expert_name,), (router_name,))


def test_qwen3moe_ep_rank_publishes_when_ordinary_dp_rank_is_nonzero(runtime, monkeypatch):
    geometry = {
        "global_rank": 1,
        "tp_rank": 0,
        "tp_size": 1,
        "pp_rank": 0,
        "pp_size": 1,
        "ep_rank": 1,
        "ep_size": 2,
        "etp_rank": 0,
        "etp_size": 1,
        "dp_rank": 1,
        "dp_size": 2,
        "expert_dp_rank": 0,
        "expert_dp_size": 1,
    }
    monkeypatch.setattr(mx, "_source_geometry", lambda: geometry)
    publisher = _Publisher([])

    updater = mx.UpdateWeightFromModelExpress(
        _args(num_experts=8),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3moeconfig",
        quantization_config=None,
        publisher_factory=lambda: publisher,
        param_infos=[],
    )

    assert updater._publishes is True
    assert updater.publisher is publisher


def test_exact_version_is_ready_before_receive_and_retired_before_resume(runtime):
    events = []
    publisher = _Publisher(events)
    updater = mx.UpdateWeightFromModelExpress(
        _args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3config",
        quantization_config=None,
        publisher_factory=lambda: publisher,
        param_infos=[],
    )
    updater.connect_rollout_engines([_Engine("rollout-a", events)], object(), [1], [12])

    updater.update_weights()
    updater.update_weights()

    names = [name for name, _value in events]
    assert names.index("publisher") < names.index("ready") < names.index("receiver")
    assert names.index("receiver") < names.index("retire") < names.index("release") < names.index("continue")
    assert [request.version for request in publisher.requests] == ["mx-1", "mx-2"]
    assert [request.training_step for request in publisher.requests] == [1, 2]
    receiver_payloads = [value["payload"] for name, value in events if name == "receiver"]
    assert [payload["target_training_step"] for payload in receiver_payloads] == [1, 2]
    assert receiver_payloads[0] == {"version_id": "mx-1", "target_training_step": 1, "logical_group": "model"}
    assert publisher.requests[0].source_geometry["tp_rank"] == 1


def test_worker_set_change_reconfigures_cohort(runtime):
    events = []
    publisher = _Publisher(events)
    updater = mx.UpdateWeightFromModelExpress(
        _args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3config",
        quantization_config=None,
        publisher_factory=lambda: publisher,
        param_infos=[],
    )
    first = [_Engine("rollout-a", events)]
    updater.connect_rollout_engines(first, object(), [2], [0])
    first_cohort = updater.cohort_id
    updater.connect_rollout_engines(first + [_Engine("rollout-b", events)], object(), [2, 2], [0, 2])

    assert updater.cohort_id != first_cohort
    assert len(publisher.registrations) == 2
    assert publisher.registrations[-1].cohort_id == updater.cohort_id
    assert len(publisher.registrations[-1].rollout_workers) == 2


def test_nonzero_data_parallel_replica_does_not_publish(runtime, monkeypatch):
    def group(rank=0, size=1):
        return SimpleNamespace(rank=rank, size=size, group=None)

    monkeypatch.setattr(
        mx,
        "get_parallel_state",
        lambda: SimpleNamespace(
            tp=group(1, 2),
            pp=group(0, 2),
            ep=group(0, 1),
            etp=group(0, 1),
            effective_dp=group(1, 2),
        ),
    )
    events = []
    updater = mx.UpdateWeightFromModelExpress(
        _args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3config",
        quantization_config=None,
        publisher_factory=lambda: (_ for _ in ()).throw(
            AssertionError("nonzero DP replica must not create a publisher")
        ),
        param_infos=[],
    )
    updater.connect_rollout_engines([_Engine("rollout-a", events)], object())

    monkeypatch.setattr(mx.dist, "get_rank", lambda: 1)
    updater.update_weights()

    assert "publisher" not in [name for name, _value in events]
    assert "continue" not in [name for name, _value in events]


def test_publisher_failure_does_not_finalize_or_resume(runtime):
    events = []

    class FailingPublisher(_Publisher):
        def publish_and_execute(self, request):
            self.events.append(("publisher", request))
            raise RuntimeError("publisher failed")

    publisher = FailingPublisher(events)
    updater = mx.UpdateWeightFromModelExpress(
        _args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3config",
        quantization_config=None,
        publisher_factory=lambda: publisher,
        param_infos=[],
    )
    updater.connect_rollout_engines([_Engine("rollout-a", events)], object(), [1], [0])

    with pytest.raises(RuntimeError, match="publisher failed"):
        updater.update_weights()

    names = [name for name, _value in events]
    assert "receiver" not in names
    assert "retire" in names and "release" in names
    assert not {"end", "version", "continue"} & set(names)


def test_receiver_failure_response_fails_closed(runtime):
    events = []
    publisher = _Publisher(events)
    engine = _Engine("rollout-a", events)
    engine.update_weights_from_modelexpress = _RemoteMethod(
        lambda **kw: events.append(("receiver", kw))
        or {
            "success": False,
            "target_training_step": kw["payload"]["target_training_step"],
            "installed_training_step": None,
            "layout_signature": "test-layout",
            "metrics": None,
            "timing": {},
            "error": "receiver poisoned",
            "receiver_poisoned": True,
        }
    )
    updater = mx.UpdateWeightFromModelExpress(
        _args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3config",
        quantization_config=None,
        publisher_factory=lambda: publisher,
        param_infos=[],
    )
    updater.connect_rollout_engines([engine], object(), [1], [0])

    with pytest.raises(RuntimeError, match="receiver poisoned"):
        updater.update_weights()

    names = [name for name, _value in events]
    assert "publisher" in names
    assert not {"end", "version", "continue"} & set(names)


@pytest.mark.parametrize("failure_phase", ["pause", "end", "version"])
def test_root_lifecycle_failure_fences_later_rounds(runtime, failure_phase):
    events = []
    publisher = _Publisher(events)
    engine = _Engine("rollout-a", events)
    attr = {"pause": "pause_generation", "end": "end_weight_update", "version": "update_weight_version"}[failure_phase]
    setattr(engine, attr, _RemoteMethod(lambda **kwargs: (_ for _ in ()).throw(RuntimeError("injected failure"))))
    updater = mx.UpdateWeightFromModelExpress(
        _args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3config",
        quantization_config=None,
        publisher_factory=lambda: publisher,
        param_infos=[],
    )
    updater.connect_rollout_engines([engine], object(), [1], [0])
    with pytest.raises(RuntimeError, match="injected failure"):
        updater.update_weights()
    names = [name for name, _ in events]
    assert "retire" in names and "release" in names
    assert "continue" not in names
    with pytest.raises(RuntimeError, match="prior ModelExpress round failed"):
        updater.update_weights()


@pytest.mark.parametrize("bad_ack", ["version", "step", "missing_rank", "duplicate_rank", "poisoned"])
def test_invalid_rank_acknowledgement_prevents_fleet_activation(runtime, bad_ack):
    events = []
    publisher = _Publisher(events)
    engine = _Engine("rollout-a", events)

    def receive(**kwargs):
        payload = kwargs["payload"]
        response = {
            "success": True,
            "version_id": payload["version_id"],
            "installed_training_step": payload["target_training_step"],
        }
        workers = [{"rank": rank, "response": dict(response)} for rank in (0, 1)]
        if bad_ack == "version":
            workers[1]["response"]["version_id"] = "other-version"
        elif bad_ack == "step":
            workers[1]["response"]["installed_training_step"] -= 1
        elif bad_ack == "missing_rank":
            workers.pop()
        elif bad_ack == "duplicate_rank":
            workers[1]["rank"] = 0
        else:
            workers[1]["response"]["receiver_poisoned"] = True
        return {"success": True, "workers": workers}

    engine.update_weights_from_modelexpress = _RemoteMethod(receive)
    updater = mx.UpdateWeightFromModelExpress(
        _args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3config",
        quantization_config=None,
        publisher_factory=lambda: publisher,
        param_infos=[],
    )
    updater.connect_rollout_engines([engine], object(), [2], [0])
    with pytest.raises(RuntimeError, match="ModelExpress receiver"):
        updater.update_weights()
    names = [name for name, _ in events]
    assert "retire" in names and "release" in names
    assert not {"end", "version", "continue"} & set(names)
    with pytest.raises(RuntimeError, match="prior ModelExpress round failed"):
        updater.update_weights()
