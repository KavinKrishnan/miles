import argparse
import logging
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.backends.sglang_utils.arguments import add_sglang_arguments, collect_eval_sglang_overrides
from miles.backends.sglang_utils.arguments import validate_args as validate_sglang_args
from miles.utils.arguments import (
    CHECKPOINT_SOURCE_DEFAULTS,
    _compute_custom_inference_engine_provider_path,
    _compute_rollout_external,
    _maybe_apply_dumper_overrides,
    _resolve_api_server_port,
    _resolve_ft_components,
    _resolve_mini_ft_controller_enable,
    _resolve_rollout_functions,
    _validate_rematerialize_param_from_master_weight,
    capture_requested_checkpoint_source,
    get_miles_extra_args_provider,
    miles_validate_args,
    resolve_checkpoint_source,
    resolve_rollout_function_paths,
    validate_async_off_policy_correction,
    validate_deploy_component,
)
from miles.utils.env_report.redaction import _SECRET_ARG_NAMES, _SECRET_ENV_VAR_PATTERN
from miles.utils.ft_utils.health_checker import SimpleHealthCheckerConfig
from miles.utils.function_registry import function_registry
from miles.utils.run_uuid import RUN_UUID_LENGTH, validate_run_uuid

PATH_ARGS = ["--rollout-function-path", "--custom-generate-function-path", "--custom-inference-engine-provider-path"]
REQUIRED_ARGS = ["--rollout-batch-size", "64"]

# These name a dataset column, a metric or a prompt field, not a credential.
_NOT_ACTUALLY_SECRET_ARG_NAMES = frozenset(
    {
        "ci_metric_checker_key",
        "eval_input_key",
        "eval_label_key",
        "eval_reward_key",
        "eval_tool_key",
        "input_key",
        "label_key",
        "metadata_key",
        "opd_teacher_key",
        "reward_key",
        "tool_key",
    }
)
_SGLANG_ARG_PREFIXES = ("sglang_", "eval_sglang_")
_INHERITED_CREDENTIAL_PATTERN = re.compile(r"^(eval_)?(sglang|router)_(.*_)?(api_keys?|password)$")


def make_class_with_add_arguments():
    class MyFn:
        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument("--my-custom-arg", type=int, default=42)

    return MyFn


def make_function_with_add_arguments():
    def my_fn():
        pass

    my_fn.add_arguments = lambda parser: parser.add_argument("--my-custom-arg", type=int, default=42)
    return my_fn


def make_function_without_add_arguments():
    def my_fn():
        pass

    return my_fn


@pytest.mark.parametrize("path_arg", PATH_ARGS)
class TestAddArgumentsSupport:

    @pytest.mark.parametrize("fn_factory", [make_class_with_add_arguments, make_function_with_add_arguments])
    def test_add_arguments_is_called_and_arg_is_parsed(self, path_arg, fn_factory):
        fn = fn_factory()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn", "--my-custom-arg", "100"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()
            assert args.my_custom_arg == 100

    def test_skips_function_without_add_arguments(self, path_arg):
        fn = make_function_without_add_arguments()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)


class TestAddArgumentsWithoutTheExperimentalRolloutFlag:
    def test_an_engine_provider_registers_its_own_flags_in_the_default_environment(self, monkeypatch):
        """External rollout does not need MILES_EXPERIMENTAL_ROLLOUT_REFACTOR, so a provider's
        add_arguments hook must run when that env var is off, as the docs promise."""
        monkeypatch.delenv("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", raising=False)
        fn = make_function_with_add_arguments()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys,
            "argv",
            ["test", "--custom-inference-engine-provider-path", "test:fn", "--my-custom-arg", "100"] + REQUIRED_ARGS,
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()

        assert args.my_custom_arg == 100


class TestRolloutExternalDerivation:
    def test_static_addrs_imply_external_rollout(self):
        """Giving engine addresses is the whole point of external mode, so no separate flag is needed."""
        args = SimpleNamespace(
            rollout_external_engine_addrs=["host1:8000"], custom_inference_engine_provider_path=None
        )

        assert _compute_rollout_external(args) is True

    def test_a_custom_provider_path_implies_external_rollout(self):
        """A user-supplied provider means miles must not launch engines of its own."""
        args = SimpleNamespace(
            rollout_external_engine_addrs=None, custom_inference_engine_provider_path="my_pkg.my_provider"
        )

        assert _compute_rollout_external(args) is True

    def test_without_either_arg_rollout_stays_internal(self):
        """The default run keeps launching its own engines."""
        args = SimpleNamespace(rollout_external_engine_addrs=None, custom_inference_engine_provider_path=None)

        assert _compute_rollout_external(args) is False

    def test_the_standalone_external_flag_no_longer_exists(self):
        """--rollout-external was replaced by derivation, so the parser must not define it anymore."""
        with patch.object(sys, "argv", ["test"] + REQUIRED_ARGS):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)

        option_strings = {s for action in parser._actions for s in action.option_strings}
        assert "--rollout-external" not in option_strings
        assert "--rollout-external-engine-addrs" in option_strings
        assert "--custom-inference-engine-provider-path" in option_strings


class TestEngineProviderPathAutofill:
    def test_a_user_given_path_is_never_overwritten(self):
        """The custom hook is the escape hatch, so validation must not replace it with a builtin."""
        args = SimpleNamespace(
            rollout_external_engine_addrs=["host1:8000"],
            custom_inference_engine_provider_path="my_pkg.my_provider",
        )

        assert _compute_custom_inference_engine_provider_path(args) == "my_pkg.my_provider"

    def test_static_addrs_fill_in_the_discovery_provider(self):
        """Static addresses mean the built-in discovery provider, chosen once in arg validation."""
        args = SimpleNamespace(
            rollout_external_engine_addrs=["host1:8000"], custom_inference_engine_provider_path=None
        )

        assert _compute_custom_inference_engine_provider_path(args) == (
            "miles.ray.rollout.external_engine_provider.static_inference_engine_provider"
        )

    def test_an_internal_run_fills_in_the_backend_provider(self):
        """Without external args the backend keeps announcing the engines it launches itself."""
        args = SimpleNamespace(rollout_external_engine_addrs=None, custom_inference_engine_provider_path=None)

        assert _compute_custom_inference_engine_provider_path(args) == (
            "miles.ray.specs.inference.backend_inference_engine_provider"
        )


EXTERNAL_ARGS = [
    "--rollout-external-engine-addrs",
    "host1:8000",
    "--rollout-num-gpus",
    "1",
    "--rollout-num-gpus-per-engine",
    "1",
    "--num-rollout",
    "1",
]


class TestExternalRolloutValidation:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + REQUIRED_ARGS)

    def test_static_addrs_derive_both_external_and_the_discovery_provider(self):
        """The helpers are unit tested in isolation, so only the real chain proves the order they run in."""
        args = self._parse(EXTERNAL_ARGS)

        miles_validate_args(args)

        assert args.rollout_external is True
        assert args.custom_inference_engine_provider_path == (
            "miles.ray.rollout.external_engine_provider.static_inference_engine_provider"
        )

    def test_an_internal_run_derives_the_backend_provider(self):
        """Every existing run takes this path, and it must reach the provider the backend announces."""
        args = self._parse(["--num-rollout", "1"])

        miles_validate_args(args)

        assert args.rollout_external is False
        assert args.custom_inference_engine_provider_path == (
            "miles.ray.specs.inference.backend_inference_engine_provider"
        )

    def test_a_custom_provider_path_alone_is_external_and_is_kept(self):
        """A user-supplied provider means miles launches no engines, and its path must survive autofill."""
        args = self._parse(
            ["--custom-inference-engine-provider-path", "my_pkg.my_provider", "--num-rollout", "1"]
        )

        miles_validate_args(args)

        assert args.rollout_external is True
        assert args.custom_inference_engine_provider_path == "my_pkg.my_provider"

    def test_static_addrs_do_not_overrule_a_custom_provider_path(self):
        """Both args together are how a custom provider reads the address book miles parsed."""
        args = self._parse(EXTERNAL_ARGS + ["--custom-inference-engine-provider-path", "my_pkg.my_provider"])

        miles_validate_args(args)

        assert args.custom_inference_engine_provider_path == "my_pkg.my_provider"

    @pytest.mark.parametrize(
        "extra, message",
        [
            (["--prefill-num-servers", "1"], "prefill_num_servers cannot be set"),
            (["--eval-num-gpus", "1"], "eval_num_gpus cannot be set"),
        ],
    )
    def test_an_arg_that_declares_a_second_topology_is_rejected(self, extra, message):
        """Two topologies would size the placement group, the router and the weight-update group
        against different fleets."""
        args = self._parse(EXTERNAL_ARGS + extra)

        with pytest.raises(AssertionError, match=message):
            miles_validate_args(args)

    def test_an_sglang_config_is_rejected_with_external_engines(self, tmp_path):
        """The external topology comes from discovery, so a declared one could only disagree with it."""
        config = tmp_path / "sglang.yaml"
        config.write_text("sglang:\n  - name: default\n    server_groups:\n      - num_gpus: 1\n")
        args = self._parse(EXTERNAL_ARGS + ["--sglang-config", str(config)])

        with pytest.raises(AssertionError, match="sglang_config cannot be set"):
            miles_validate_args(args)

    def test_the_external_pd_router_flag_is_rejected_on_an_internal_run(self):
        """An internal run reads PD off its own config, so the flag could only contradict it."""
        args = self._parse(["--rollout-external-router-pd", "--num-rollout", "1"])

        with pytest.raises(AssertionError, match="rollout-external-router-pd"):
            miles_validate_args(args)

    def test_the_external_pd_router_flag_is_accepted_with_external_engines(self):
        """This is the only channel external PD has, so the guard must not close it."""
        args = self._parse(EXTERNAL_ARGS + ["--rollout-external-router-pd"])

        miles_validate_args(args)

        assert args.rollout_external_router_pd is True

    def test_the_same_args_stay_legal_on_an_internal_run(self):
        """The guards are about the combination, so each half alone must keep working."""
        args = self._parse(["--prefill-num-servers", "1", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.rollout_external is False


class TestMaybeApplyDumperOverrides:
    def _make_args(
        self,
        *,
        dumper_enable: bool = False,
        use_fault_tolerance: bool = False,
        ft_components: list[str] | None = None,
        router_disable_health_check: bool = False,
        rollout_health_check_interval: float = 30.0,
        miles_router_health_check_failure_threshold: int = 3,
        miles_router_max_connections: int | None = 64,
        miles_router_timeout: float | None = None,
        start_rollout_id: int | None = None,
        num_rollout: int = 10,
        eval_interval: int | None = 5,
        save: str | None = "/tmp/checkpoint",
        save_interval: int | None = 5,
        save_retain_interval: int | None = 10,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            dumper_enable=dumper_enable,
            use_fault_tolerance=use_fault_tolerance,
            ft_components=ft_components if ft_components is not None else [],
            mini_ft_controller_enable=None,
            router_disable_health_check=router_disable_health_check,
            rollout_health_check_interval=rollout_health_check_interval,
            miles_router_health_check_failure_threshold=miles_router_health_check_failure_threshold,
            miles_router_max_connections=miles_router_max_connections,
            miles_router_timeout=miles_router_timeout,
            start_rollout_id=start_rollout_id,
            num_rollout=num_rollout,
            eval_interval=eval_interval,
            save=save,
            save_interval=save_interval,
            save_retain_interval=save_retain_interval,
        )

    def test_noop_when_dumper_disabled(self) -> None:
        args = self._make_args(
            dumper_enable=False,
            use_fault_tolerance=True,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is True
        assert args.router_disable_health_check is False
        assert args.num_rollout == 10
        assert args.eval_interval == 5
        assert args.save == "/tmp/checkpoint"
        assert args.save_interval == 5
        assert args.save_retain_interval == 10

    def test_disables_fault_tolerance_and_sglang_router_heartbeats(self) -> None:
        """Dumper mode turns off fault tolerance and the SGLang router health check."""
        args = self._make_args(
            dumper_enable=True,
            use_fault_tolerance=True,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is False
        assert args.router_disable_health_check is True

    def test_no_healing_loop_survives_dumper_mode(self) -> None:
        """It is resolved from ft_components, which dumper mode clears, so resolving it first
        would leave the loop polling a registry with nothing in it for the whole run."""
        args = self._make_args(dumper_enable=True, use_fault_tolerance=True, ft_components=["rollout"])

        _maybe_apply_dumper_overrides(args)

        assert _resolve_mini_ft_controller_enable(args) is False

    def test_the_selected_ft_components_go_with_the_flag(self) -> None:
        """ft_components is resolved from the flag long before this runs, so clearing the flag
        alone would leave every component selected and its probes still firing."""
        args = self._make_args(dumper_enable=True, use_fault_tolerance=True, ft_components=["rollout", "train"])

        _maybe_apply_dumper_overrides(args)

        assert args.ft_components == []

    def test_forces_single_rollout(self) -> None:
        args = self._make_args(dumper_enable=True, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.start_rollout_id == 0
        assert args.num_rollout == 1
        assert args.eval_interval is None
        assert args.save is None
        assert args.save_interval is None
        assert args.save_retain_interval is None

    def test_respects_start_rollout_id(self) -> None:
        args = self._make_args(dumper_enable=True, start_rollout_id=5, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.num_rollout == 6


def test_fully_async_eval_resolves_to_the_producer_itself():
    """Only the producer's own instance pauses on eval, and RolloutManager reuses one
    instance only when both paths match."""
    path = "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn"
    default = SimpleNamespace(rollout_function_path=None, eval_function_path=None, fully_async=True)
    assert resolve_rollout_function_paths(default) == (path, path)

    override = SimpleNamespace(rollout_function_path=None, eval_function_path="pkg.CustomEval", fully_async=True)
    assert resolve_rollout_function_paths(override) == (path, "pkg.CustomEval")


def test_fully_async_rejects_abort_pause_mode(monkeypatch):
    """Generation is always in flight, so aborting on every weight update would kill it."""
    monkeypatch.setenv("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "1")
    args = SimpleNamespace(
        fully_async=True,
        multi_lora=False,
        rollout_function_path=None,
        eval_function_path=None,
        colocate=False,
        partial_rollout=False,
        pause_generation_mode="abort",
        recompute_logprobs_via_prefill=False,
        rollout_all_samples_process_path=None,
        eval_num_gpus=0,
    )

    with pytest.raises(AssertionError, match="pause-generation-mode abort"):
        _resolve_rollout_functions(args)

    args.pause_generation_mode = "retract"
    _resolve_rollout_functions(args)


class TestClusterBackend:

    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + REQUIRED_ARGS)

    def test_defaults_to_ray(self):
        """Runs that do not mention the flag keep the ray-launched worker behaviour."""
        assert self._parse([]).cluster_backend == "ray"

    @pytest.mark.parametrize("backend", ["ray", "kubernetes"])
    def test_accepts_supported_backends(self, backend):
        """Both supported backends parse into the raw string."""
        assert self._parse(["--cluster-backend", backend]).cluster_backend == backend

    def test_rejects_unknown_backend(self):
        """An unsupported backend name fails at parse time instead of later."""
        with pytest.raises(SystemExit):
            self._parse(["--cluster-backend", "slurm"])

    def test_validation_accepts_kubernetes_now_that_it_provisions_workers(self):
        """The kubernetes backend observes platform-created workers, so validation must let a run reach it."""
        args = self._parse(["--cluster-backend", "kubernetes", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.cluster_backend == "kubernetes"

    def test_the_custom_config_file_still_decides_the_backend(self, tmp_path):
        """The config file overwrites args after the flags are parsed, so its backend must be the one that survives."""
        config = tmp_path / "override.yaml"
        config.write_text("cluster_backend: kubernetes\n")
        args = self._parse(["--custom-config-path", str(config), "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.cluster_backend == "kubernetes"

    def test_a_kubernetes_run_is_moved_onto_the_mooncake_object_store(self):
        """A ray store reference can only be redeemed by a ray driver, and this run has none."""
        args = self._parse(["--cluster-backend", "kubernetes", "--object-store-backend", "ray", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.object_store_backend == "mooncake"

    def test_the_override_outlives_the_custom_config_file(self, tmp_path):
        """That file is applied late, so a ray store named there would otherwise survive the override."""
        config = tmp_path / "override.yaml"
        config.write_text("cluster_backend: kubernetes\nobject_store_backend: ray\n")
        args = self._parse(["--custom-config-path", str(config), "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.object_store_backend == "mooncake"

    def test_a_ray_run_may_keep_the_ray_object_store(self):
        """Every existing run takes this path, and nothing about it changed."""
        args = self._parse(["--cluster-backend", "ray", "--object-store-backend", "ray", "--num-rollout", "1"])

        miles_validate_args(args)

        assert args.object_store_backend == "ray"


def test_recompute_logprobs_via_prefill_flag_is_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(["--recompute-logprobs-via-prefill"] + REQUIRED_ARGS)

    assert args.recompute_logprobs_via_prefill is True


def test_sglang_parallel_sizes_keep_server_args_destinations():
    parser = add_sglang_arguments(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--sglang-tp-size",
            "6",
            "--sglang-data-parallel-size",
            "2",
            "--sglang-pipeline-parallel-size",
            "3",
            "--sglang-expert-parallel-size",
            "4",
            "--sglang-attention-context-parallel-size",
            "5",
        ]
    )
    args.rollout_num_gpus_per_engine = 8
    args.true_on_policy_mode = False
    args.sglang_enable_dp_attention = True
    args.use_session_server = False

    validate_sglang_args(args)

    assert args.sglang_tp_size == 8
    assert args.sglang_dp_size == 2
    assert args.sglang_pp_size == 3
    assert args.sglang_ep_size == 4
    assert args.sglang_attn_cp_size == 5


_SHARED_STORE_ARGS = [
    "--object-store-backend",
    "mooncake",
    "--mooncake-store-init-kwargs",
    '{"master_server_address": "the-master:50051"}',
]


def _write_megatron_config(tmp_path, model_ids: list[str]) -> str:
    import yaml

    path = tmp_path / "megatron.yaml"
    path.write_text(yaml.dump({"megatron": [{"name": model_id} for model_id in model_ids]}))
    return str(path)


class TestDeployComponent:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args = parser.parse_args(extra + REQUIRED_ARGS + ["--num-rollout", "1"])
        args.ft_components = []
        args.mini_ft_controller_enable = False
        return args

    def _parse_validated(self, extra):
        args = self._parse(extra)
        args.ft_components = _resolve_ft_components(args)
        args.api_server_port = _resolve_api_server_port(args)
        args.mini_ft_controller_enable = _resolve_mini_ft_controller_enable(args)
        return args

    def test_defaults_to_deploying_the_whole_run(self):
        """A run that does not mention the flag is one deployment, exactly as before the flag existed."""
        assert self._parse([]).deploy_component == "all"

    def test_rejects_a_component_that_is_not_one_of_the_four(self):
        """The four values partition the run, so a fifth name would deploy an undefined subset."""
        with pytest.raises(ValueError):
            validate_deploy_component(self._parse(["--deploy-component", "router"]))

    def test_an_instance_of_a_component_that_comes_in_instances_validates(self):
        """A multi policy run installs one trainer deployment per policy, each naming its own model id."""
        validate_deploy_component(self._parse(["--deploy-component", "trainer:policy_a", *_SHARED_STORE_ARGS]))

    def test_rejects_an_instance_of_a_component_a_run_deploys_exactly_one_of(self):
        """Naming an instance of the orchestration script would deploy a second copy of the run itself."""
        with pytest.raises(AssertionError, match="names an instance of primary"):
            validate_deploy_component(self._parse(["--deploy-component", "primary:one", *_SHARED_STORE_ARGS]))

    @pytest.mark.parametrize("component", ["trainer", "inference"])
    def test_a_deployment_without_the_orchestration_script_needs_no_addresses(self, component):
        """It calls nobody: the orchestration script calls it, so it has nothing to be told."""
        validate_deploy_component(self._parse(["--deploy-component", component, *_SHARED_STORE_ARGS]))

    @pytest.mark.parametrize("component", ["trainer", "inference"])
    def test_a_deployment_without_the_orchestration_script_has_to_share_an_object_store(self, component):
        """A ray reference is redeemable only inside the deployment that made it, and the data crosses deployments."""
        with pytest.raises(AssertionError, match="--object-store-backend"):
            validate_deploy_component(self._parse(["--deploy-component", component, "--object-store-backend", "ray"]))

    def test_a_deployment_without_the_orchestration_script_has_to_be_told_where_the_store_master_is(self):
        """It runs no master of its own, so an unnamed one leaves it writing into a store nobody else reads."""
        with pytest.raises(AssertionError, match="master_server_address"):
            validate_deploy_component(
                self._parse(["--deploy-component", "trainer", "--object-store-backend", "mooncake"])
            )

    def test_the_store_master_address_has_to_carry_a_port(self):
        """A host without a port cannot be dialed, and the failure would surface as a hang much later."""
        with pytest.raises(AssertionError, match="master_server_address"):
            validate_deploy_component(
                self._parse(
                    [
                        "--deploy-component",
                        "trainer",
                        "--object-store-backend",
                        "mooncake",
                        "--mooncake-store-init-kwargs",
                        '{"master_server_address": "the-master"}',
                    ]
                )
            )

    def test_a_deployment_told_where_the_store_master_is_validates(self):
        """This is what makes the trainer read the rollout data the orchestration script's deployment wrote."""
        validate_deploy_component(
            self._parse(
                [
                    "--deploy-component",
                    "trainer",
                    "--object-store-backend",
                    "mooncake",
                    "--mooncake-store-init-kwargs",
                    '{"master_server_address": "the-master:50051"}',
                ]
            )
        )

    def test_a_primary_deployment_has_to_be_told_where_the_trainer_is(self):
        """Nothing derives another release's pod names, so an unnamed trainer is unreachable."""
        with pytest.raises(AssertionError, match="--trainer-controller-addrs"):
            validate_deploy_component(self._parse(["--deploy-component", "primary"]))

    def test_a_primary_deployment_has_to_be_told_where_the_inference_side_is(self):
        """Same for the inference controller, which the weight update window is opened on."""
        with pytest.raises(AssertionError, match="--inference-controller-addrs"):
            validate_deploy_component(
                self._parse(["--deploy-component", "primary", "--trainer-controller-addrs", "10.0.0.1:8000"])
            )

    def test_a_primary_deployment_has_to_be_told_where_the_routers_are(self):
        """The rollout executor generates through the router, which lives with the engines it fronts."""
        with pytest.raises(AssertionError, match="--inference-router-addrs"):
            validate_deploy_component(
                self._parse(
                    [
                        "--deploy-component",
                        "primary",
                        "--trainer-controller-addrs",
                        "10.0.0.1:8000",
                        "--inference-controller-addrs",
                        "10.0.0.2:8000",
                    ]
                )
            )

    def test_a_fully_addressed_primary_deployment_validates(self):
        """The three flags together are what makes an orchestration script able to run without its workers."""
        validate_deploy_component(
            self._parse(
                [
                    "--deploy-component",
                    "primary",
                    "--trainer-controller-addrs",
                    "10.0.0.1:8000",
                    "--inference-controller-addrs",
                    "10.0.0.2:8000",
                    "--inference-router-addrs",
                    "10.0.0.3:8000",
                    *_SHARED_STORE_ARGS,
                ]
            )
        )

    def test_a_primary_deployment_shares_an_object_store_too(self):
        """It writes the rollout data the trainer deployment reads, which its own store alone cannot carry."""
        with pytest.raises(AssertionError, match="--object-store-backend"):
            validate_deploy_component(
                self._parse(
                    [
                        "--deploy-component",
                        "primary",
                        "--trainer-controller-addrs",
                        "10.0.0.1:8000",
                        "--inference-controller-addrs",
                        "10.0.0.2:8000",
                        "--inference-router-addrs",
                        "10.0.0.3:8000",
                        "--object-store-backend",
                        "ray",
                    ]
                )
            )

    def test_a_train_only_run_still_has_to_be_told_where_the_inference_controller_is(self):
        """The orchestration script builds its handle and calls init on it whether or not engines are deployed."""
        with pytest.raises(AssertionError, match="--inference-controller-addrs"):
            validate_deploy_component(
                self._parse(
                    [
                        "--deploy-component",
                        "primary",
                        "--trainer-controller-addrs",
                        "10.0.0.1:8000",
                        "--debug-train-only",
                        *_SHARED_STORE_ARGS,
                    ]
                )
            )

    def test_a_train_only_run_needs_no_router_addresses(self):
        """--debug-train-only deploys no engines and no routers, so no router is ever resolved."""
        validate_deploy_component(
            self._parse(
                [
                    "--deploy-component",
                    "primary",
                    "--trainer-controller-addrs",
                    "10.0.0.1:8000",
                    "--inference-controller-addrs",
                    "10.0.0.2:8000",
                    "--debug-train-only",
                    *_SHARED_STORE_ARGS,
                ]
            )
        )

    @pytest.mark.parametrize(
        ("component", "flag"),
        [
            ("trainer", "--trainer-controller-addrs"),
            ("inference", "--inference-controller-addrs"),
            ("inference", "--inference-router-addrs"),
            ("all", "--trainer-controller-addrs"),
            ("all", "--inference-controller-addrs"),
            ("all", "--inference-router-addrs"),
        ],
    )
    def test_refuses_a_static_address_for_a_component_this_launch_deploys_itself(self, component, flag):
        """A static address describes what another launch deploys, so one for our own release is a contradiction."""
        with pytest.raises(AssertionError, match=flag):
            validate_deploy_component(
                self._parse(["--deploy-component", component, flag, "10.0.0.1:8000", *_SHARED_STORE_ARGS])
            )

    @pytest.mark.parametrize(
        ("component", "flag"),
        [
            ("trainer", "--inference-controller-addrs"),
            ("inference", "--trainer-controller-addrs"),
        ],
    )
    def test_allows_a_static_address_for_a_component_another_launch_deploys(self, component, flag):
        """That component is outside this release either way, so naming it is at worst unused, never wrong."""
        validate_deploy_component(
            self._parse(["--deploy-component", component, flag, "10.0.0.1:8000", *_SHARED_STORE_ARGS])
        )

    def test_refuses_an_api_server_that_answers_for_cells_another_deployment_owns(self):
        """It reads cells of its own release, so a split run's would come back missing instead of unreachable."""
        with pytest.raises(AssertionError, match="--api-server-port"):
            validate_deploy_component(
                self._parse_validated(
                    [
                        "--deploy-component",
                        "primary",
                        "--trainer-controller-addrs",
                        "10.0.0.1:8000",
                        "--inference-controller-addrs",
                        "10.0.0.2:8000",
                        "--inference-router-addrs",
                        "10.0.0.3:8000",
                        "--use-fault-tolerance",
                        *_SHARED_STORE_ARGS,
                    ]
                )
            )

    def test_a_trainer_deployment_keeps_the_fault_tolerance_of_its_own_cells(self):
        """Its controller watches its own ranks, and it runs no api server to be blind with."""
        validate_deploy_component(
            self._parse_validated(
                [
                    "--deploy-component",
                    "trainer",
                    "--use-fault-tolerance",
                    "--ft-components",
                    "train",
                    *_SHARED_STORE_ARGS,
                ]
            )
        )

    def test_refuses_to_split_a_colocated_run(self):
        """Colocated trainers and engines share gpus, so they can only be installed as one unit."""
        with pytest.raises(AssertionError, match="--colocate"):
            validate_deploy_component(self._parse(["--deploy-component", "trainer", "--colocate"]))

    def test_a_colocated_single_policy_run_stays_allowed(self):
        """One policy has one trainer, so its engines have exactly one set of nodes to sit on."""
        validate_deploy_component(self._parse(["--colocate"]))

    def test_refuses_to_colocate_a_run_that_trains_several_policies(self, tmp_path):
        """With two trainers, which of them an engine shares gpus with would be undefined."""
        with pytest.raises(AssertionError, match="which trainer an engine belongs beside"):
            validate_deploy_component(
                self._parse(["--colocate", "--megatron-config", _write_megatron_config(tmp_path, ["a", "b"])])
            )

    def test_refuses_to_colocate_a_single_named_instance(self, tmp_path):
        """The engines pinned onto a trainer's nodes are installed with it, never as a deployment of their own."""
        with pytest.raises(AssertionError, match="install one instance of them on its own"):
            validate_deploy_component(self._parse(["--deploy-component", "trainer:a", "--colocate"]))

    def test_an_engine_only_deployment_needs_the_controller_it_registers_into(self):
        """It deploys no controller of its own, so without an address its engines join nothing."""
        with pytest.raises(AssertionError, match="--inference-controller-addrs"):
            validate_deploy_component(self._parse_validated(["--deploy-component", "inference:east"]))

    def test_an_engine_only_deployment_may_name_the_controller_of_another_release(self):
        """The controller lives in the unnamed inference release, which this launch does not deploy."""
        validate_deploy_component(
            self._parse_validated(
                [
                    "--deploy-component",
                    "inference:east",
                    "--inference-controller-addrs",
                    "http://10.0.0.4:8000",
                    "--registration-token",
                    "secret",
                    *_SHARED_STORE_ARGS,
                ]
            )
        )

    def test_an_engine_only_deployment_without_a_shared_token_is_refused(self):
        """A token defaulting to None leaves a cross-datacenter endpoint that takes membership from anyone."""
        with pytest.raises(AssertionError, match="--registration-token"):
            validate_deploy_component(
                self._parse_validated(
                    [
                        "--deploy-component",
                        "inference:east",
                        "--inference-controller-addrs",
                        "http://10.0.0.4:8000",
                        *_SHARED_STORE_ARGS,
                    ]
                )
            )

    def test_a_run_that_waits_for_reporters_without_a_shared_token_is_refused(self):
        """The receiving side has to authenticate too, or the token on the reporting side proves nothing."""
        with pytest.raises(AssertionError, match="--registration-token"):
            validate_deploy_component(
                self._parse_validated(
                    ["--deploy-component", "inference", "--expected-registration-reporters", "1", *_SHARED_STORE_ARGS]
                )
            )

    def test_refuses_to_colocate_a_run_that_other_deployments_register_engines_into(self):
        """A colocated weight update writes to gpus of its own nodes, which a registered engine has none of."""
        with pytest.raises(AssertionError, match="hold gpus of their own"):
            validate_deploy_component(
                self._parse(["--colocate", "--expected-registration-reporters", "1", "--registration-token", "secret"])
            )

    def test_the_inference_release_may_not_name_the_controller_it_deploys(self):
        """A launch reaches what it deploys by the names of its own release, never by a static address."""
        with pytest.raises(AssertionError, match="deploys it here"):
            validate_deploy_component(
                self._parse_validated(
                    [
                        "--deploy-component",
                        "inference",
                        "--inference-controller-addrs",
                        "http://10.0.0.4:8000",
                        *_SHARED_STORE_ARGS,
                    ]
                )
            )

    def test_an_engine_only_deployment_does_not_wait_for_reporters_itself(self):
        """It is the one that reports; waiting for reporters belongs to the release holding the controller."""
        with pytest.raises(AssertionError, match="--expected-registration-reporters"):
            validate_deploy_component(
                self._parse_validated(
                    [
                        "--deploy-component",
                        "inference:east",
                        "--inference-controller-addrs",
                        "http://10.0.0.4:8000",
                        "--registration-token",
                        "secret",
                        "--expected-registration-reporters",
                        "2",
                    ]
                )
            )

    def test_a_primary_deployment_still_names_the_one_inference_controller(self):
        """There is exactly one controller in a run, and the orchestration script drives it directly."""
        with pytest.raises(AssertionError, match="--inference-controller-addrs"):
            validate_deploy_component(
                self._parse_validated(
                    [
                        "--deploy-component",
                        "primary",
                        "--trainer-controller-addrs",
                        "10.0.0.1:8000",
                        *_SHARED_STORE_ARGS,
                    ]
                )
            )


class TestEvalSglangOverrides:
    """Unset means "inherit --sglang-*", so an unset flag must leave no attribute at all."""

    def _parse(self, argv):
        return add_sglang_arguments(argparse.ArgumentParser()).parse_args(argv)

    def test_unset_flags_produce_no_overrides(self):
        args = self._parse(["--sglang-mem-fraction-static", "0.7"])

        assert collect_eval_sglang_overrides(args) == {}
        assert not hasattr(args, "eval_sglang_mem_fraction_static")

    def test_set_flag_becomes_an_override_without_touching_the_base_family(self):
        args = self._parse(["--sglang-mem-fraction-static", "0.7", "--eval-sglang-mem-fraction-static", "0.9"])

        assert collect_eval_sglang_overrides(args) == {"mem_fraction_static": 0.9}
        assert args.sglang_mem_fraction_static == 0.7

    def test_boolean_can_be_turned_back_off(self):
        args = self._parse(["--sglang-enable-dp-attention", "--no-eval-sglang-enable-dp-attention"])

        assert args.sglang_enable_dp_attention is True
        assert collect_eval_sglang_overrides(args) == {"enable_dp_attention": False}

    def test_parallel_sizes_keep_server_args_destinations(self):
        args = self._parse(["--eval-sglang-data-parallel-size", "2", "--eval-sglang-expert-parallel-size", "4"])

        assert collect_eval_sglang_overrides(args) == {"dp_size": 2, "ep_size": 4}

    def test_tp_size_is_not_exposed(self):
        """A second TP knob could move tp_size off the bundles --eval-num-gpus-per-engine placed."""
        with pytest.raises(SystemExit):
            self._parse(["--eval-sglang-tp-size", "2"])


def test_custom_megatron_post_save_hook_path_is_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(["--custom-megatron-post-save-hook-path", "pkg.module.hook"] + REQUIRED_ARGS)

    assert args.custom_megatron_post_save_hook_path == "pkg.module.hook"


def test_custom_megatron_post_save_hook_path_requires_save():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        ["--custom-megatron-post-save-hook-path", "pkg.module.hook", "--num-rollout", "1"] + REQUIRED_ARGS
    )

    with pytest.raises(
        AssertionError,
        match="'--save' is required when custom_megatron_post_save_hook_path is set.",
    ):
        miles_validate_args(args)


def test_dynamic_global_batch_size_requires_dynamic_batch_size():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(["--use-dynamic-global-batch-size", "--num-rollout", "1"] + REQUIRED_ARGS)

    with pytest.raises(AssertionError, match="requires --use-dynamic-batch-size"):
        miles_validate_args(args)


class TestCriticSaveDerivation:
    def _validate(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        args = parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)
        miles_validate_args(args)
        return args

    def test_derives_sibling_dir_from_save(self):
        args = self._validate(["--advantage-estimator", "ppo", "--save", "/ckpts/run1"])
        assert args.critic_save == "/ckpts/run1_critic"

    def test_trailing_slash_is_stripped(self):
        args = self._validate(["--advantage-estimator", "ppo", "--save", "/ckpts/run1/"])
        assert args.critic_save == "/ckpts/run1_critic"

    def test_explicit_critic_save_is_respected(self):
        args = self._validate(
            ["--advantage-estimator", "ppo", "--save", "/ckpts/run1", "--critic-save", "/elsewhere/critic"]
        )
        assert args.critic_save == "/elsewhere/critic"

    def test_stays_none_without_save(self):
        args = self._validate(["--advantage-estimator", "ppo"])
        assert args.critic_save is None


class TestSessionServerV2Validation:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    @pytest.mark.parametrize(
        ("extra", "flag"),
        [
            (["--group-rm"], "--group-rm"),
            (["--partial-rollout"], "--partial-rollout"),
            (
                ["--true-on-policy-mode", "--recompute-logprobs-via-prefill"],
                "--recompute-logprobs-via-prefill",
            ),
        ],
    )
    def test_rejects_unsupported_list_consumers(self, extra, flag):
        args = self._parse(["--use-session-server", "v2", *extra])

        with pytest.raises(ValueError) as exc_info:
            miles_validate_args(args)

        assert str(exc_info.value) == (f"--use-session-server v2 does not support {flag}; v2 returns list[Sample]")


class TestTitoFixedTemplateConfiguration:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    def test_removed_role_flag_is_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--tito-allowed-append-roles", "tool"])

    @pytest.mark.parametrize(
        ("extra", "expect_warning"),
        [
            (["--use-session-server"], True),
            ([], False),
            (["--use-session-server", "--tito-model", "qwen3"], False),
        ],
    )
    def test_warns_only_for_default_model_session(self, caplog, extra, expect_warning):
        args = self._parse(extra)

        with caplog.at_level(logging.WARNING, logger="miles.utils.arguments"):
            miles_validate_args(args)

        target_records = [
            record
            for record in caplog.records
            if record.getMessage().startswith("--tito-model=default uses a best-effort four-role append surface.")
        ]
        assert len(target_records) == int(expect_warning)

    def test_named_family_requires_session_server(self):
        args = self._parse(["--tito-model", "qwen3"])
        with pytest.raises(ValueError, match="--tito-model=qwen3 requires --use-session-server"):
            miles_validate_args(args)

    def test_named_family_resolves_registered_template_and_kwargs(self):
        args = self._parse(["--use-session-server", "--tito-model", "qwen3"])
        miles_validate_args(args)
        assert args.chat_template_path.endswith("/qwen3_fixed.jinja")
        assert args.apply_chat_template_kwargs == {"clear_thinking": False}

    def test_named_family_rejects_custom_template(self):
        args = self._parse(
            [
                "--use-session-server",
                "--tito-model",
                "qwen3",
                "--chat-template-path",
                "/tmp/custom.jinja",
            ]
        )
        with pytest.raises(ValueError, match="cannot override the template registered"):
            miles_validate_args(args)

    def test_named_family_rejects_conflicting_registered_kwarg(self):
        args = self._parse(
            [
                "--use-session-server",
                "--tito-model",
                "qwen3",
                "--apply-chat-template-kwargs",
                '{"clear_thinking": true}',
            ]
        )
        with pytest.raises(ValueError, match="clear_thinking=True conflicts"):
            miles_validate_args(args)

    def test_named_family_accepts_same_registered_and_additional_kwargs(self):
        args = self._parse(
            [
                "--use-session-server",
                "--tito-model",
                "qwen3",
                "--apply-chat-template-kwargs",
                '{"clear_thinking": false, "enable_thinking": true}',
            ]
        )
        miles_validate_args(args)
        assert args.apply_chat_template_kwargs == {
            "clear_thinking": False,
            "enable_thinking": True,
        }


def test_bridge_mode_rejects_critic(tmp_path):
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        [
            "--advantage-estimator",
            "ppo",
            "--megatron-to-hf-mode",
            "bridge",
            "--hf-checkpoint",
            str(tmp_path),
            "--num-rollout",
            "1",
        ]
        + REQUIRED_ARGS
    )

    with pytest.raises(
        AssertionError,
        match="Critic models are not supported with --megatron-to-hf-mode bridge",
    ):
        miles_validate_args(args)


def test_critic_is_accepted_on_the_only_trainer(tmp_path):
    """Shared actor/critic PPO used to be rejected on the cell based trainer, which is now the only one."""
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        ["--advantage-estimator", "ppo", "--hf-checkpoint", str(tmp_path), "--num-rollout", "1"] + REQUIRED_ARGS
    )

    miles_validate_args(args)

    assert args.use_critic is True


def test_critic_rejects_reward_level_kl(tmp_path):
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        [
            "--advantage-estimator",
            "ppo",
            "--kl-coef",
            "0.05",
            "--ref-load",
            str(tmp_path),
            "--hf-checkpoint",
            str(tmp_path),
            "--num-rollout",
            "1",
        ]
        + REQUIRED_ARGS
    )

    with pytest.raises(AssertionError, match="does not support reward-level KL"):
        miles_validate_args(args)


class TestMultiLoRAValidation:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(
            [
                "--multi-lora-n-adapters",
                "2",
                "--lora-rank",
                "8",
                "--target-modules",
                "linear_qkv",
                "--num-rollout",
                "1",
            ]
            + extra
            + REQUIRED_ARGS
        )

    def test_rejects_multiple_tokenizer_workers(self):
        # Each sglang tokenizer worker holds its own LoRA registry, so per-step
        # upserts fail non-deterministically; fail at launch, not first push.
        args = self._parse(["--sglang-tokenizer-worker-num", "2"])

        with pytest.raises(AssertionError, match="sglang-tokenizer-worker-num 1"):
            miles_validate_args(args)

    def test_accepts_default_single_tokenizer_worker(self):
        args = self._parse([])

        miles_validate_args(args)

        assert args.multi_lora is True

    def test_defaults_rollout_fn_and_data_source_to_multi_lora(self):
        args = self._parse([])

        miles_validate_args(args)

        assert args.rollout_function_path == "miles.rollout.multi_lora.async_rollout.generate_rollout_multi_lora"
        assert args.data_source_path == "miles.rollout.multi_lora.data_source.MultiLoRAAsyncDataSource"
        assert args.rollout_global_dataset is True

    def test_keeps_user_supplied_rollout_fn_and_data_source(self):
        args = self._parse(
            ["--rollout-function-path", "my.custom.rollout_fn", "--data-source-path", "my.custom.DataSource"]
        )

        miles_validate_args(args)

        assert args.rollout_function_path == "my.custom.rollout_fn"
        assert args.data_source_path == "my.custom.DataSource"

    def test_empty_wait_is_a_registered_argument(self):
        assert self._parse([]).multi_lora_max_empty_wait_s == 30.0
        assert self._parse(["--multi-lora-max-empty-wait-s", "5"]).multi_lora_max_empty_wait_s == 5.0

    def test_rejects_non_adam_optimizer(self):
        # Per-slot optimizer isolation (state init, retirement cleanup, step
        # clocks) only implements Adam semantics. Muon has its own dedicated
        # rejection; anything else non-Adam trips the generic guard.
        args = self._parse([])
        args.optimizer = "muon"
        with pytest.raises(AssertionError, match="does not support Muon"):
            miles_validate_args(args)

        args = self._parse([])
        args.optimizer = "sgd"
        with pytest.raises(AssertionError, match="requires --optimizer adam"):
            miles_validate_args(args)

    def test_is_accepted_on_the_only_trainer(self):
        """Multi-LoRA used to be rejected on the cell based trainer, which is now the only one."""
        args = self._parse([])

        miles_validate_args(args)

    def test_rejects_pipeline_parallelism(self):
        # Adapter routing is not recompute-safe under a pipelined schedule.
        args = self._parse([])
        args.pipeline_model_parallel_size = 2
        with pytest.raises(AssertionError, match="pipeline-model-parallel-size 1"):
            miles_validate_args(args)

    def test_rejects_bshd_qkv_format(self):
        # bshd interleaves samples in the sequence-major flattening the spans assume.
        args = self._parse([])
        args.qkv_format = "bshd"
        with pytest.raises(AssertionError, match="qkv-format thd"):
            miles_validate_args(args)

    def test_rejects_shared_outer_expert_loras(self):
        # Per-expert layout only; the flag would switch sglang to a layout training never produces.
        args = self._parse([])
        args.experts_shared_outer_loras = True
        with pytest.raises(AssertionError, match="experts-shared-outer-loras"):
            miles_validate_args(args)

    def test_accepts_expert_leaf_targets_without_expert_tp_flag(self):
        # --expert-tensor-parallel-size stays None until Megatron's own validate_args;
        # comparing the raw value here rejected every run that omitted the flag.
        args = self._parse(["--target-modules", "gate_proj,up_proj,down_proj"])
        args.expert_tensor_parallel_size = None

        miles_validate_args(args)

        assert args.multi_lora is True


class TestResolveFtComponents:
    def test_disabled_with_no_components_returns_empty_without_warning(self, caplog) -> None:
        """use_fault_tolerance off and no ft_components yields an empty list and no warning."""
        args = SimpleNamespace(use_fault_tolerance=False, ft_components=None)
        with caplog.at_level(logging.WARNING, logger="miles.utils.arguments"):
            result = _resolve_ft_components(args)

        assert result == []
        assert not any("--ft-components is ignored" in record.message for record in caplog.records)

    def test_disabled_with_components_returns_empty_and_warns(self, caplog) -> None:
        """use_fault_tolerance off but ft_components set returns empty list and logs an ignore warning."""
        args = SimpleNamespace(use_fault_tolerance=False, ft_components=["train"])
        with caplog.at_level(logging.WARNING, logger="miles.utils.arguments"):
            result = _resolve_ft_components(args)

        assert result == []
        assert any(
            "--ft-components is ignored without --use-fault-tolerance" in record.message for record in caplog.records
        )

    def test_enabled_with_no_components_returns_default(self) -> None:
        """use_fault_tolerance on with no ft_components falls back to the default ['rollout']."""
        args = SimpleNamespace(use_fault_tolerance=True, ft_components=None)
        result = _resolve_ft_components(args)

        assert result == ["rollout"]

    def test_enabled_with_components_returns_distinct_copy(self) -> None:
        """use_fault_tolerance on with ft_components returns an equal but distinct list copy."""
        components = ["train", "rollout"]
        args = SimpleNamespace(use_fault_tolerance=True, ft_components=components)
        result = _resolve_ft_components(args)

        assert result == ["train", "rollout"]
        assert result is not components


@pytest.mark.parametrize(
    ("parallel_args", "expected"),
    [
        ([], (1, 1, 1, 1)),
        (
            [
                "--sglang-tensor-parallel-size",
                "2",
                "--sglang-data-parallel-size",
                "3",
                "--sglang-pipeline-parallel-size",
                "4",
                "--sglang-expert-parallel-size",
                "5",
                "--sglang-enable-dp-attention",
            ],
            (2, 3, 4, 5),
        ),
        (
            [
                "--sglang-tp-size",
                "2",
                "--sglang-dp-size",
                "3",
                "--sglang-pp-size",
                "4",
                "--sglang-ep-size",
                "5",
                "--sglang-enable-dp-attention",
            ],
            (2, 3, 4, 5),
        ),
    ],
)
def test_sglang_parallel_sizes_use_short_namespace_fields(parallel_args, expected):
    parser = argparse.ArgumentParser()
    add_sglang_arguments(parser)
    args = parser.parse_args(parallel_args)

    assert (args.sglang_tp_size, args.sglang_dp_size, args.sglang_pp_size, args.sglang_ep_size) == expected
    assert not hasattr(args, "sglang_tensor_parallel_size")
    assert not hasattr(args, "sglang_data_parallel_size")
    assert not hasattr(args, "sglang_pipeline_parallel_size")
    assert not hasattr(args, "sglang_expert_parallel_size")

    args.rollout_num_gpus_per_engine = 8
    args.true_on_policy_mode = False
    args.recompute_logprobs_via_prefill = False
    args.sglang_router_policy = None
    args.use_session_server = False

    validate_sglang_args(args)

    assert args.sglang_tp_size == 8
    assert (args.sglang_dp_size, args.sglang_pp_size, args.sglang_ep_size) == expected[1:]


def test_sglang_parallel_size_aliases_keep_last_value():
    parser = argparse.ArgumentParser()
    add_sglang_arguments(parser)

    args = parser.parse_args(["--sglang-data-parallel-size", "2", "--sglang-dp-size", "3"])

    assert args.sglang_dp_size == 3


def _make_async_ppo_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        use_critic=True,
        use_rollout_logprobs=False,
        use_tis=False,
        keep_old_actor=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestValidateAsyncOffPolicyCorrection:
    def test_ppo_without_correction_is_rejected(self):
        with pytest.raises(AssertionError, match="behavior-policy correction"):
            validate_async_off_policy_correction(_make_async_ppo_args())

    @pytest.mark.parametrize("flag", ["use_rollout_logprobs", "use_tis", "keep_old_actor"])
    def test_ppo_with_any_correction_passes(self, flag):
        validate_async_off_policy_correction(_make_async_ppo_args(**{flag: True}))

    def test_non_ppo_estimators_are_unaffected(self):
        validate_async_off_policy_correction(_make_async_ppo_args(use_critic=False))


class TestValidateRematerializeParamFromMasterWeight:
    def _make_args(self, **overrides) -> SimpleNamespace:
        args = SimpleNamespace(
            rematerialize_param_from_master_weight=True,
            train_backend="megatron",
            lora_rank=0,
            lora_adapter_path=None,
            debug_disable_optimizer=False,
            indep_dp=False,
            colocate=True,
            offload_train=True,
            offload_train_target="cpu",
            use_distributed_optimizer=True,
            keep_old_actor=False,
            kl_coef=0,
            use_kl_loss=False,
            opd_teacher_load=None,
            use_precision_aware_optimizer=False,
            optimizer_cpu_offload=False,
            overlap_param_gather=False,
            compute_advantages_and_returns=True,
            num_critic_only_steps=0,
            debug_train_only=False,
            ci_test=False,
            check_rematerialize_param_from_master_weight=False,
            disable_param_buffers_cpu_backup=False,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_valid_config_forces_no_param_buffer_cpu_backup(self):
        args = self._make_args()
        _validate_rematerialize_param_from_master_weight(args)
        assert args.disable_param_buffers_cpu_backup is True

    def test_accepts_precision_aware_with_cpu_offload(self):
        args = self._make_args(use_precision_aware_optimizer=True, optimizer_cpu_offload=True)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.disable_param_buffers_cpu_backup is True

    def test_ci_test_auto_enables_the_check(self):
        args = self._make_args(ci_test=True)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.check_rematerialize_param_from_master_weight is True

    def test_check_stays_off_outside_ci(self):
        args = self._make_args()
        _validate_rematerialize_param_from_master_weight(args)
        assert args.check_rematerialize_param_from_master_weight is False

    def test_accepts_ref_and_teacher_tags(self):
        for overrides in ({"use_kl_loss": True}, {"kl_coef": 0.1}, {"opd_teacher_load": "/path/to/teacher"}):
            _validate_rematerialize_param_from_master_weight(self._make_args(**overrides))

    def test_debug_train_only_silently_disables(self):
        args = self._make_args(debug_train_only=True, colocate=False)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.rematerialize_param_from_master_weight is False
        assert args.disable_param_buffers_cpu_backup is False

    def test_noop_when_disabled(self):
        args = self._make_args(rematerialize_param_from_master_weight=False, colocate=False)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.disable_param_buffers_cpu_backup is False

    @pytest.mark.parametrize(
        "overrides",
        [
            {"train_backend": "fsdp"},
            {"lora_rank": 8},
            {"lora_adapter_path": "/path/to/adapter"},
            {"debug_disable_optimizer": True},
            {"indep_dp": True},
            {"colocate": False},
            {"offload_train": False},
            {"offload_train_target": "disk"},
            {"use_distributed_optimizer": False},
            {"keep_old_actor": True},
            {"use_precision_aware_optimizer": True},
            {"overlap_param_gather": True},
            {"compute_advantages_and_returns": False},
            {"num_critic_only_steps": 2},
        ],
    )
    def test_rejects_unsupported_config(self, overrides):
        with pytest.raises(AssertionError):
            _validate_rematerialize_param_from_master_weight(self._make_args(**overrides))

    def test_backend_is_checked_before_megatron_only_args(self):
        # An fsdp Namespace has none of the megatron args the later asserts read.
        args = SimpleNamespace(
            rematerialize_param_from_master_weight=True,
            train_backend="fsdp",
            debug_train_only=False,
        )
        with pytest.raises(AssertionError, match="Megatron"):
            _validate_rematerialize_param_from_master_weight(args)


class TestRunUuidResolution:
    def _parse(self, extra: list[str]):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(["--num-rollout", "1"] + extra + REQUIRED_ARGS)

    def test_unset_run_uuid_is_generated(self):
        """Every launch gets an identifier, so nothing has to cope with it being absent."""
        args = self._parse([])
        miles_validate_args(args)

        assert validate_run_uuid(args.run_uuid)

    def test_two_launches_do_not_share_a_run_uuid(self):
        """A colliding identifier would attribute one run's artifacts to another."""
        first, second = self._parse([]), self._parse([])
        miles_validate_args(first)
        miles_validate_args(second)

        assert first.run_uuid != second.run_uuid

    def test_an_explicit_run_uuid_is_kept(self):
        """Reproducing a run means being able to pin its identifier."""
        pinned = ("ab12cd34ef5678ab" * 4)[:RUN_UUID_LENGTH]
        args = self._parse(["--run-uuid", pinned])
        miles_validate_args(args)

        assert args.run_uuid == pinned

    def test_a_run_uuid_from_the_custom_config_file_is_validated_too(self, tmp_path):
        """The config file overwrites args after the flags are parsed, so it must not skip the check."""
        config = tmp_path / "override.yaml"
        config.write_text("run_uuid: my-experiment\n")
        args = self._parse(["--custom-config-path", str(config)])

        with pytest.raises(ValueError, match="invalid run uuid"):
            miles_validate_args(args)

    def test_a_run_uuid_blanked_by_the_custom_config_file_is_regenerated(self, tmp_path):
        """A null in the config file must not leave the identifier unset for the whole run."""
        config = tmp_path / "override.yaml"
        config.write_text("run_uuid: null\n")
        args = self._parse(["--custom-config-path", str(config)])
        miles_validate_args(args)

        assert validate_run_uuid(args.run_uuid)

    def test_a_malformed_explicit_run_uuid_fails_at_launch(self):
        """Rejecting it here beats corrupting every string that embeds it hours into a run."""
        args = self._parse(["--run-uuid", "my-experiment"])

        with pytest.raises(ValueError, match="invalid run uuid"):
            miles_validate_args(args)


class TestRolloutHealthCheckArguments:
    def _parse(self, extra: list[str]):
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        return parser.parse_args(extra + REQUIRED_ARGS)

    def test_the_rollout_defaults_survive_the_move_onto_the_shared_config(self):
        """The shared config carries the trainer's defaults, which are not the rollout ones."""
        args = self._parse([])

        assert args.rollout_health_check_interval == 30.0
        assert args.rollout_health_check_timeout == 30.0
        assert args.rollout_health_check_first_wait == 0.0

    def test_the_first_wait_grace_period_is_still_tunable(self):
        """A first launch compiling deepgemm kernels needs a grace period, or it is killed while warming up."""
        assert self._parse(["--rollout-health-check-first-wait", "600"]).rollout_health_check_first_wait == 600.0

    def test_the_resolved_rollout_config_matches_the_parsed_arguments(self):
        """The config is what the checker actually runs on, so it must not diverge from the flags."""
        config = SimpleHealthCheckerConfig.from_args(
            self._parse(["--rollout-health-check-first-wait", "600"]), prefix="rollout_health_check"
        )

        assert (config.interval, config.timeout, config.first_wait) == (30.0, 30.0, 600.0)

    def test_the_trainer_heartbeat_keeps_its_own_debounce(self):
        """The rollout default must not be pushed down into the shared config: a trainer heartbeat
        shares an RPC channel with the train step, so one slow reply is a blip, not a dead cell."""
        assert self._parse([]).trainer_heartbeat_checker_failure_threshold == 3


class TestSecretArgumentsAreClassified:
    def _declared_names(self) -> set[str]:
        parser = argparse.ArgumentParser()
        get_miles_extra_args_provider()(parser)
        # The eval sglang flags default to SUPPRESS, so parsing alone would not materialise them.
        return {action.dest for action in parser._actions}

    def test_every_secret_looking_miles_flag_is_either_redacted_or_declared_harmless(self):
        """The env report hashes args by an explicit list, so a new credential flag would leak until listed."""
        suspicious = {
            name
            for name in self._declared_names()
            if _SECRET_ENV_VAR_PATTERN.search(name) and not name.startswith(_SGLANG_ARG_PREFIXES)
        }

        assert suspicious - _SECRET_ARG_NAMES == _NOT_ACTUALLY_SECRET_ARG_NAMES, (
            "an argument's name looks like a credential; add it to _SECRET_ARG_NAMES in env_report/redaction.py so the env "
            "report hashes it, or to _NOT_ACTUALLY_SECRET_ARG_NAMES here to say it names something else"
        )

    def test_every_credential_inherited_from_sglang_and_the_router_is_redacted(self):
        """sglang and the router contribute api keys and key passwords that land in the args dump verbatim."""
        credentials = {name for name in self._declared_names() if _INHERITED_CREDENTIAL_PATTERN.search(name)}

        assert credentials >= {"sglang_api_key", "eval_sglang_api_key", "router_api_key"}
        assert credentials <= _SECRET_ARG_NAMES


class TestValidateMultiPolicyArgs:
    @staticmethod
    def _write(tmp_path, name: str, data: dict) -> str:
        import yaml

        path = tmp_path / name
        path.write_text(yaml.dump(data))
        return str(path)

    def _args(self, tmp_path, *, model_ids: list[str], fully_async: bool = True, **overrides):
        megatron_config = self._write(tmp_path, "megatron.yaml", {"megatron": [{"name": n} for n in model_ids]})
        sglang_config = self._write(
            tmp_path,
            "sglang.yaml",
            {
                "sglang": [
                    {"name": n, "update_weights": True, "server_groups": [{"worker_type": "regular", "num_gpus": 1}]}
                    for n in model_ids
                ]
            },
        )
        defaults = dict(
            megatron_config=megatron_config,
            sglang_config=sglang_config,
            fully_async=fully_async,
            use_critic=False,
            save=None,
            load=None,
            hf_checkpoint="/models/base",
            advantage_estimator="grpo",
            num_steps_per_rollout=None,
            global_batch_size=None,
            rollout_batch_size=8,
            n_samples_per_prompt=4,
            rollout_num_gpus=len(model_ids),
            rollout_num_gpus_per_engine=1,
            eval_num_gpus=0,
            prefill_num_servers=None,
            offload_rollout=False,
            debug_train_only=False,
            debug_rollout_only=False,
            colocate=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=1,
            critic_num_nodes=0,
            critic_num_gpus_per_node=0,
            critic_train_only=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_a_single_policy_run_is_always_allowed(self, tmp_path):
        """Every existing run is a single policy run and must not be gated on anything new."""
        from miles.utils.arguments import validate_multi_policy_args

        validate_multi_policy_args(SimpleNamespace(megatron_config=None, fully_async=False, use_critic=True))

    def test_a_single_policy_config_carrying_per_policy_args_is_refused(self, tmp_path):
        """Nothing applies an overlay to a single policy run, so those arguments would vanish."""
        from miles.utils.arguments import validate_multi_policy_args

        args = self._args(tmp_path, model_ids=["a"])
        args.megatron_config = self._write(tmp_path, "solo.yaml", {"megatron": [{"name": "a", "args": "--lr 5e-7"}]})

        with pytest.raises(AssertionError, match="silently ignored"):
            validate_multi_policy_args(args)

    def test_a_per_policy_argument_outside_the_whitelist_is_refused_at_startup(self, tmp_path):
        """A rhythm argument is read from the base command line, so failing late wastes a job."""
        from miles.utils.arguments import validate_multi_policy_args

        args = self._args(tmp_path, model_ids=["a", "b"])
        args.megatron_config = self._write(
            tmp_path, "rhythm.yaml", {"megatron": [{"name": "a", "args": "--num-rollout 3"}, {"name": "b"}]}
        )

        with pytest.raises(AssertionError, match="--num-rollout"):
            validate_multi_policy_args(args)

    def test_a_policy_whose_engines_are_frozen_by_inference_is_refused(self, tmp_path):
        """update_weights defaults to False for a model_path of its own, and the run would die mid-training."""
        from miles.utils.arguments import validate_multi_policy_args

        args = self._args(tmp_path, model_ids=["a", "b"])
        args.sglang_config = self._write(
            tmp_path,
            "sglang_paths.yaml",
            {
                "sglang": [
                    {"name": "a", "server_groups": [{"worker_type": "regular", "num_gpus": 1}]},
                    {
                        "name": "b",
                        "model_path": "/models/other",
                        "server_groups": [{"worker_type": "regular", "num_gpus": 1}],
                    },
                ]
            },
        )

        with pytest.raises(AssertionError, match="no matching --sglang-config model"):
            validate_multi_policy_args(args)

    def test_a_run_with_a_save_directory_passes_validation(self, tmp_path):
        """Deriving one checkpoint directory per policy must not collide with itself on the happy path."""
        from miles.utils.arguments import validate_multi_policy_args

        validate_multi_policy_args(self._args(tmp_path, model_ids=["a", "b"], save="/ckpt/run"))

    def test_multi_policy_requires_fully_async(self, tmp_path):
        """The other rollout modes drive one policy per rollout round; failing late wastes a job."""
        from miles.utils.arguments import validate_multi_policy_args

        with pytest.raises(AssertionError, match="only supported for --fully-async"):
            validate_multi_policy_args(self._args(tmp_path, model_ids=["a", "b"], fully_async=False))

    def test_multi_policy_accepts_a_matching_sglang_config(self, tmp_path):
        """The names are one model id shared by the trainer and the inference side."""
        from miles.utils.arguments import validate_multi_policy_args

        validate_multi_policy_args(self._args(tmp_path, model_ids=["a", "b"]))

    def test_a_policy_without_an_inference_model_is_refused(self, tmp_path):
        """Weights of a policy with no engines of its own would have nowhere to land."""
        from miles.utils.arguments import validate_multi_policy_args

        args = self._args(tmp_path, model_ids=["a", "b"])
        args.sglang_config = self._write(
            tmp_path,
            "sglang_partial.yaml",
            {"sglang": [{"name": "a", "server_groups": [{"worker_type": "regular", "num_gpus": 2}]}]},
        )

        with pytest.raises(AssertionError, match="no matching --sglang-config model"):
            validate_multi_policy_args(args)

    def test_multi_policy_without_an_sglang_config_is_refused(self, tmp_path):
        """One inference model per policy is what makes per-model weight updates possible."""
        from miles.utils.arguments import validate_multi_policy_args

        args = self._args(tmp_path, model_ids=["a", "b"])
        args.sglang_config = None

        with pytest.raises(AssertionError, match="needs --sglang-config"):
            validate_multi_policy_args(args)

    def _checkpoint_args(self, tmp_path, checkpoints: dict[str, str], **overrides):
        args = self._args(tmp_path, model_ids=sorted(checkpoints), **overrides)
        args.megatron_config = self._write(
            tmp_path,
            "megatron_checkpoints.yaml",
            {
                "megatron": [
                    {"name": model_id, "args": f"--hf-checkpoint {path}"}
                    for model_id, path in sorted(checkpoints.items())
                ]
            },
        )
        return args

    @staticmethod
    def _stub_tokenizers(monkeypatch, *, vocab_sizes: dict[str, int], fingerprints: dict[str, str]) -> None:
        monkeypatch.setattr(
            "miles.utils.arguments.load_hf_config", lambda path: SimpleNamespace(vocab_size=vocab_sizes[path])
        )
        monkeypatch.setattr("miles.utils.arguments.compute_tokenizer_fingerprint", lambda path: fingerprints[path])

    def test_policies_sharing_one_tokenizer_pass_the_vocabulary_check(self, tmp_path, monkeypatch):
        """Two different model sizes of the same family is the use case multi policy exists for."""
        from miles.utils.arguments import validate_multi_policy_args

        args = self._checkpoint_args(tmp_path, {"a": "/models/small", "b": "/models/large"})
        self._stub_tokenizers(
            monkeypatch,
            vocab_sizes={"/models/small": 100, "/models/large": 100, "/models/base": 100},
            fingerprints={"/models/small": "x", "/models/large": "x", "/models/base": "x"},
        )

        validate_multi_policy_args(args)

    def test_policies_with_different_vocabulary_sizes_are_refused(self, tmp_path, monkeypatch):
        """One tokenizer encodes every prompt, so a policy on another vocabulary trains on noise."""
        from miles.utils.arguments import validate_multi_policy_args

        args = self._checkpoint_args(tmp_path, {"a": "/models/small", "b": "/models/large"})
        self._stub_tokenizers(
            monkeypatch,
            vocab_sizes={"/models/small": 100, "/models/large": 200, "/models/base": 100},
            fingerprints={"/models/small": "x", "/models/large": "x", "/models/base": "x"},
        )

        with pytest.raises(AssertionError, match="different vocabularies"):
            validate_multi_policy_args(args)

    def test_policies_whose_tokenizers_differ_below_the_vocabulary_size_are_refused(self, tmp_path, monkeypatch):
        """Two tokenizers of the same size can still map the same text to different ids, and nothing errors."""
        from miles.utils.arguments import validate_multi_policy_args

        args = self._checkpoint_args(tmp_path, {"a": "/models/small", "b": "/models/large"})
        self._stub_tokenizers(
            monkeypatch,
            vocab_sizes={"/models/small": 100, "/models/large": 100, "/models/base": 100},
            fingerprints={"/models/small": "x", "/models/large": "y", "/models/base": "x"},
        )

        with pytest.raises(AssertionError, match="tokenizers disagree"):
            validate_multi_policy_args(args)

    def test_the_checkpoint_that_builds_the_rollout_tokenizer_is_compared_too(self, tmp_path, monkeypatch):
        """Every policy overriding --hf-checkpoint used to leave the tokenizer's own checkpoint out of the set."""
        from miles.utils.arguments import validate_multi_policy_args

        args = self._checkpoint_args(tmp_path, {"a": "/models/other", "b": "/models/other"})
        self._stub_tokenizers(
            monkeypatch,
            vocab_sizes={"/models/other": 200, "/models/base": 100},
            fingerprints={"/models/other": "x", "/models/base": "x"},
        )

        with pytest.raises(AssertionError, match="different vocabularies"):
            validate_multi_policy_args(args)


class TestTheCheckpointSourceDerivation:
    def _args(self, tmp_path, *, load: str, **overrides):
        args = argparse.Namespace(
            megatron_to_hf_mode="raw",
            load=load,
            ref_load=str(tmp_path / "ref"),
            hf_checkpoint=None,
            ref_ckpt_step=None,
            ckpt_step=None,
            finetune=False,
            no_load_optim=False,
            no_load_rng=False,
            start_rollout_id=9,
            **overrides,
        )
        args.requested_checkpoint_source = {name: getattr(args, name) for name in CHECKPOINT_SOURCE_DEFAULTS}
        return args

    def _write_checkpoint(self, tmp_path):
        (tmp_path / "save").mkdir(exist_ok=True)
        (tmp_path / "save" / "latest_checkpointed_iteration.txt").write_text("12")

    def test_a_run_whose_load_dir_is_not_there_yet_starts_from_the_reference_weights(self, tmp_path):
        """This is every first launch: the checkpoint dir the command names is created by the run itself."""
        args = self._args(tmp_path, load=str(tmp_path / "save"))

        resolve_checkpoint_source(args)

        assert args.load == args.ref_load
        assert (args.finetune, args.no_load_optim, args.no_load_rng) == (True, True, True)
        assert args.start_rollout_id == 0

    def test_the_same_command_derives_a_different_source_once_the_run_has_written_a_checkpoint(self, tmp_path):
        """The derivation reads the filesystem, so the answer it gave at launch expires as soon as the run saves."""
        args = self._args(tmp_path, load=str(tmp_path / "save"))
        resolve_checkpoint_source(args)

        self._write_checkpoint(tmp_path)
        resolve_checkpoint_source(args)

        assert args.load == str(tmp_path / "save")
        assert (args.finetune, args.no_load_optim, args.no_load_rng) == (False, False, False)

    def test_a_user_asking_for_finetune_keeps_it_when_the_checkpoint_appears(self, tmp_path):
        """Only the derived values are re-derived; what the command line asked for is restored as it was."""
        args = self._args(tmp_path, load=str(tmp_path / "save"), finetune=True)
        args.requested_checkpoint_source["finetune"] = True
        resolve_checkpoint_source(args)

        self._write_checkpoint(tmp_path)
        resolve_checkpoint_source(args)

        assert args.finetune is True

    def test_the_reference_checkpoint_step_is_used_only_while_the_reference_is(self, tmp_path):
        """--ref-ckpt-step names a step of the reference checkpoint, which means nothing in the run's own one."""
        args = self._args(tmp_path, load=str(tmp_path / "save"), ref_ckpt_step=3)
        resolve_checkpoint_source(args)
        assert args.ckpt_step == 3

        self._write_checkpoint(tmp_path)
        resolve_checkpoint_source(args)

        assert args.ckpt_step is None

    def test_bridge_mode_falls_back_to_the_reference_only_until_a_checkpoint_exists(self, tmp_path):
        """The bridge path derives its load dir from the filesystem too, and goes as stale as the other one."""
        args = self._args(tmp_path, load=str(tmp_path / "save"), megatron_to_hf_mode="bridge")
        resolve_checkpoint_source(args)
        assert args.load == args.ref_load

        self._write_checkpoint(tmp_path)
        resolve_checkpoint_source(args)

        assert args.load == str(tmp_path / "save")

    def test_the_derivation_is_idempotent(self, tmp_path):
        """Every reload re-runs it against the same filesystem, so a second run must not read its own output."""
        args = self._args(tmp_path, load=str(tmp_path / "save"), ref_ckpt_step=3)

        resolve_checkpoint_source(args)
        once = {name: getattr(args, name) for name in CHECKPOINT_SOURCE_DEFAULTS}
        resolve_checkpoint_source(args)

        assert {name: getattr(args, name) for name in CHECKPOINT_SOURCE_DEFAULTS} == once


class TestTheCheckpointSourceCapture:
    def _args(self, tmp_path, **overrides):
        return argparse.Namespace(
            megatron_to_hf_mode="raw",
            load=str(tmp_path / "save"),
            ref_load=str(tmp_path / "ref"),
            hf_checkpoint=None,
            ref_ckpt_step=None,
            start_rollout_id=9,
            **overrides,
        )

    def test_a_backend_whose_parser_omits_a_field_still_records_it(self, tmp_path):
        """The fsdp parser defines neither --finetune nor --ckpt-step, and an unrecorded field is never restored."""
        args = self._args(tmp_path)

        capture_requested_checkpoint_source(args)

        assert args.requested_checkpoint_source == dict(
            load=str(tmp_path / "save"), ckpt_step=None, finetune=False, no_load_optim=False, no_load_rng=False
        )

    def test_a_field_the_derivation_created_is_restored_by_the_next_one(self, tmp_path):
        """This is the hot-restart bug in miniature: a fresh run turns finetune on and nothing ever turns it off."""
        args = self._args(tmp_path)
        capture_requested_checkpoint_source(args)
        resolve_checkpoint_source(args)
        assert args.finetune is True

        (tmp_path / "save").mkdir(exist_ok=True)
        (tmp_path / "save" / "latest_checkpointed_iteration.txt").write_text("12")
        resolve_checkpoint_source(args)

        assert args.finetune is False
        assert args.load == str(tmp_path / "save")

    def test_what_the_command_line_asked_for_is_recorded_as_it_was(self, tmp_path):
        """A user who passed --finetune has to keep it, so the capture cannot substitute its own default."""
        args = self._args(tmp_path, finetune=True, ckpt_step=5, no_load_optim=True, no_load_rng=False)

        capture_requested_checkpoint_source(args)

        assert args.requested_checkpoint_source["finetune"] is True
        assert args.requested_checkpoint_source["ckpt_step"] == 5
        assert args.requested_checkpoint_source["no_load_optim"] is True
