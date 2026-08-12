from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import base64
from argparse import Namespace

import pytest
import yaml

from miles.utils.megatron_config import (
    DEFAULT_TRAINER_MODEL_ID,
    compute_model_args,
    compute_policy_checkpoint_dir,
    resolve_megatron_config,
)


def _write_yaml(data: dict, tmp_path) -> str:
    path = tmp_path / "megatron.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


def _make_args(megatron_config: str | None = None, **overrides) -> Namespace:
    defaults = dict(
        megatron_config=megatron_config,
        lr=1e-6,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        hf_checkpoint="/models/base",
        global_batch_size=None,
        eps_clip_high=None,
        save=None,
        load=None,
        advantage_estimator="grpo",
        num_steps_per_rollout=None,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestResolveMegatronConfig:
    def test_a_run_without_the_flag_is_a_single_element_config_list(self):
        """Legacy single policy runs must keep working, normalized to one model named 'default'."""
        config = resolve_megatron_config(_make_args())

        assert config.model_ids == [DEFAULT_TRAINER_MODEL_ID]
        assert config.primary_model_id == DEFAULT_TRAINER_MODEL_ID
        assert not config.is_multi_policy

    def test_the_yaml_names_become_the_trainer_model_ids(self, tmp_path):
        """The `name` field is the source of truth for trainer_model_id and spec names."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "--lr 1e-5"}, {"name": "b"}]}, tmp_path)

        config = resolve_megatron_config(_make_args(path))

        assert config.model_ids == ["a", "b"]
        assert config.primary_model_id == "a"
        assert config.is_multi_policy

    def test_the_first_model_is_the_primary_policy(self, tmp_path):
        """The primary owns the global checkpoint index, so its identity must be positional and stable."""
        path = _write_yaml({"megatron": [{"name": "second"}, {"name": "first"}]}, tmp_path)

        assert resolve_megatron_config(_make_args(path)).primary_model_id == "second"

    def test_an_inline_base64_payload_is_accepted(self, tmp_path):
        """Launchers that cannot ship a file still need to pass the config."""
        payload = base64.b64encode(yaml.dump({"megatron": [{"name": "solo"}]}).encode()).decode()

        config = resolve_megatron_config(_make_args(f"base64:{payload}"))

        assert config.model_ids == ["solo"]

    def test_duplicate_names_are_refused(self, tmp_path):
        """Two policies sharing an id would silently interleave into one queue and one checkpoint."""
        path = _write_yaml({"megatron": [{"name": "a"}, {"name": "a"}]}, tmp_path)

        with pytest.raises(AssertionError, match="unique"):
            resolve_megatron_config(_make_args(path))

    def test_an_unknown_yaml_key_is_refused(self, tmp_path):
        """A strict model turns a typo into a startup error instead of a silently ignored setting."""
        path = _write_yaml({"megatron": [{"name": "a", "arg": "--lr 1e-5"}]}, tmp_path)

        with pytest.raises(Exception, match="arg"):
            resolve_megatron_config(_make_args(path))

    def test_getting_an_unknown_model_id_fails_loudly(self, tmp_path):
        """Callers routing by model id must not silently fall back to another policy."""
        path = _write_yaml({"megatron": [{"name": "a"}]}, tmp_path)

        with pytest.raises(KeyError, match="Unknown trainer model id"):
            resolve_megatron_config(_make_args(path)).get("b")


class TestComputeModelArgs:
    def test_each_policy_overlays_its_own_args_on_the_base_arguments(self, tmp_path):
        """Per-policy megatron args are the whole point of the flag; the base args stay untouched."""
        path = _write_yaml(
            {"megatron": [{"name": "a", "args": "--lr 5e-7 --tensor-model-parallel-size 2"}, {"name": "b"}]},
            tmp_path,
        )
        args = _make_args(path)

        model_a = compute_model_args(args, "a")
        model_b = compute_model_args(args, "b")

        assert (model_a.lr, model_a.tensor_model_parallel_size) == (5e-7, 2)
        assert (model_b.lr, model_b.tensor_model_parallel_size) == (1e-6, 1)
        assert (args.lr, args.tensor_model_parallel_size) == (1e-6, 1)

    def test_a_valueless_flag_becomes_true(self, tmp_path):
        """store_true arguments have no value on the command line."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "--sequence-parallel"}, {"name": "b"}]}, tmp_path)

        assert compute_model_args(_make_args(path), "a").sequence_parallel is True

    def test_the_equals_form_is_accepted(self, tmp_path):
        """`--lr=5e-7` is as common as `--lr 5e-7` in launch scripts."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "--lr=5e-7"}, {"name": "b"}]}, tmp_path)

        assert compute_model_args(_make_args(path), "a").lr == 5e-7

    def test_a_value_is_typed_by_the_whitelist_not_by_the_base_default(self, tmp_path):
        """Arguments whose base default is None used to keep the raw string, so `if args.x` read '0' as true."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "--global-batch-size 128"}, {"name": "b"}]}, tmp_path)

        assert compute_model_args(_make_args(path), "a").global_batch_size == 128

    def test_a_valueless_non_flag_argument_is_refused(self, tmp_path):
        """Without a declared type, `--eps-clip-high` with no value used to silently become True."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "--eps-clip-high"}, {"name": "b"}]}, tmp_path)

        with pytest.raises(AssertionError, match="without a value"):
            compute_model_args(_make_args(path), "a")

    def test_an_argument_outside_the_per_policy_whitelist_is_refused(self, tmp_path):
        """Rhythm arguments are read from the base command line, so accepting them here would do nothing."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "--num-rollout 3"}, {"name": "b"}]}, tmp_path)

        with pytest.raises(AssertionError, match="--num-rollout"):
            compute_model_args(_make_args(path), "a")

    def test_an_unknown_argument_is_refused(self, tmp_path):
        """A per-policy typo would otherwise be dropped and the policy would train with base settings."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "--no-such-flag 3"}, {"name": "b"}]}, tmp_path)

        with pytest.raises(AssertionError, match="--no-such-flag"):
            compute_model_args(_make_args(path), "a")

    def test_a_whitelisted_argument_this_run_does_not_declare_is_refused(self, tmp_path):
        """A whitelist entry is not a promise that every backend's parser declares it."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "--min-lr 1e-8"}, {"name": "b"}]}, tmp_path)

        with pytest.raises(AssertionError, match="does not know"):
            compute_model_args(_make_args(path), "a")

    def test_repeated_values_are_refused(self, tmp_path):
        """Every per-policy argument takes one value; a second one means the args string was mis-typed."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "--lr 1e-5 2e-5"}, {"name": "b"}]}, tmp_path)

        with pytest.raises(AssertionError, match="several values"):
            compute_model_args(_make_args(path), "a")

    def test_a_value_without_a_flag_is_refused(self, tmp_path):
        """A stray token means the args string was mis-typed; silently ignoring it hides a wrong config."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "1e-5"}, {"name": "b"}]}, tmp_path)

        with pytest.raises(AssertionError, match="must start with a flag"):
            compute_model_args(_make_args(path), "a")


class TestPolicyCheckpointDirs:
    def test_a_multi_policy_run_gives_every_policy_its_own_checkpoint_dir(self, tmp_path):
        """A shared --save makes two policies write the same iter_* directory and overwrite each other."""
        path = _write_yaml({"megatron": [{"name": "a"}, {"name": "b"}]}, tmp_path)
        args = _make_args(path, save="/ckpt/run", load="/ckpt/old")

        model_a = compute_model_args(args, "a")
        model_b = compute_model_args(args, "b")

        assert (model_a.save, model_a.load) == ("/ckpt/run/policies/a", "/ckpt/old/policies/a")
        assert (model_b.save, model_b.load) == ("/ckpt/run/policies/b", "/ckpt/old/policies/b")

    def test_a_single_policy_run_keeps_the_paths_it_was_given(self, tmp_path):
        """Existing checkpoints and existing resume commands must keep working byte for byte."""
        args = _make_args(save="/ckpt/run", load="/ckpt/old")

        model = compute_model_args(args, DEFAULT_TRAINER_MODEL_ID)

        assert (model.save, model.load) == ("/ckpt/run", "/ckpt/old")

    def test_an_unset_checkpoint_dir_stays_unset(self, tmp_path):
        """A run without --save must not grow a derived path out of None."""
        path = _write_yaml({"megatron": [{"name": "a"}, {"name": "b"}]}, tmp_path)

        assert compute_model_args(_make_args(path), "a").save is None

    def test_the_derived_dir_is_the_model_id_under_a_policies_directory(self):
        """The layout is a user visible contract: it is where a resume looks for a policy's checkpoints."""
        assert compute_policy_checkpoint_dir("/ckpt/run", "policy_b") == "/ckpt/run/policies/policy_b"


class TestDerivedPerPolicyArgs:
    def test_an_overlaid_advantage_estimator_redrives_use_critic(self, tmp_path):
        """The overlay is applied after the base derivations ran, so use_critic used to keep the base value."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "--advantage-estimator ppo"}, {"name": "b"}]}, tmp_path)
        args = _make_args(path, advantage_estimator="grpo")

        assert compute_model_args(args, "a").use_critic is True
        assert compute_model_args(args, "b").use_critic is False

    def test_an_overlaid_global_batch_size_contradicting_num_steps_per_rollout_is_refused(self, tmp_path):
        """--num-steps-per-rollout claims a number of gradient steps a policy would silently not take."""
        path = _write_yaml({"megatron": [{"name": "a", "args": "--global-batch-size 128"}, {"name": "b"}]}, tmp_path)
        args = _make_args(path, num_steps_per_rollout=4, rollout_batch_size=8, n_samples_per_prompt=4)

        with pytest.raises(AssertionError, match="num_steps_per_rollout"):
            compute_model_args(args, "a")

    def test_a_policy_without_an_overlay_derives_the_same_global_batch_size(self, tmp_path):
        """The re-derivation must reproduce what the base validation already computed."""
        path = _write_yaml({"megatron": [{"name": "a"}, {"name": "b"}]}, tmp_path)
        args = _make_args(path, num_steps_per_rollout=4, rollout_batch_size=8, n_samples_per_prompt=4)

        assert compute_model_args(args, "b").global_batch_size == 8


class TestModelIdNames:
    def test_a_model_id_that_escapes_its_checkpoint_directory_is_refused(self, tmp_path):
        """A model id is pasted into --save and --load, so it must stay one path component."""
        path = _write_yaml({"megatron": [{"name": "../evil"}, {"name": "b"}]}, tmp_path)

        with pytest.raises(AssertionError, match="not usable as path components"):
            resolve_megatron_config(_make_args(path))
