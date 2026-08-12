from types import SimpleNamespace

import pytest
import yaml

from miles.ray.specs.trainer_identity import (
    CRITIC_TRAINER_ROLE,
    DEFAULT_TRAINER_ROLE,
    compute_policy_trainer_roles,
    compute_trainer_controller_pool_id,
    compute_trainer_model_id,
    compute_trainer_pool_id,
    compute_trainer_role,
    compute_trainer_roles,
)
from miles.utils.megatron_config import resolve_megatron_config


def _write_megatron_config(tmp_path, *model_ids: str) -> str:
    path = tmp_path / "megatron.yaml"
    path.write_text(yaml.dump({"megatron": [{"name": model_id} for model_id in model_ids]}))
    return str(path)


def _make_args(megatron_config: str | None = None, **overrides) -> SimpleNamespace:
    args = SimpleNamespace(megatron_config=megatron_config, use_critic=False)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestComputeTrainerRole:
    def test_a_single_policy_run_keeps_calling_its_trainer_the_actor(self):
        """Every legacy name, pool and checkpoint path is built from this role, so it must not move."""
        config = resolve_megatron_config(_make_args())

        assert compute_trainer_role(config, config.primary_model_id) == DEFAULT_TRAINER_ROLE

    def test_a_multi_policy_run_names_each_trainer_after_its_model(self, tmp_path):
        """The role is what tells two policies' pools, cells and controllers apart."""
        config = resolve_megatron_config(_make_args(_write_megatron_config(tmp_path, "alpha", "beta")))

        assert [compute_trainer_role(config, model_id) for model_id in config.model_ids] == ["alpha", "beta"]


class TestComputePolicyTrainerRoles:
    def test_a_single_policy_run_has_exactly_one_policy_role(self):
        """A run without --megatron-config trains one model, whatever the config normalizes it to."""
        assert compute_policy_trainer_roles(_make_args()) == [DEFAULT_TRAINER_ROLE]

    def test_the_roles_follow_the_order_the_config_declares(self, tmp_path):
        """The first model is the primary policy, so the order is part of the run's contract."""
        args = _make_args(_write_megatron_config(tmp_path, "alpha", "beta", "gamma"))

        assert compute_policy_trainer_roles(args) == ["alpha", "beta", "gamma"]

    def test_a_critic_is_not_a_policy(self, tmp_path):
        """Policy roles drive weight updates and rollout, which the critic takes no part in."""
        args = _make_args(_write_megatron_config(tmp_path, "alpha"), use_critic=True)

        assert compute_policy_trainer_roles(args) == ["alpha"]


class TestComputeTrainerRoles:
    def test_a_plain_run_trains_only_the_actor(self):
        """A critic role nobody asked for would be waiting for cells that are never scheduled."""
        assert compute_trainer_roles(_make_args()) == [DEFAULT_TRAINER_ROLE]

    def test_the_critic_is_appended_after_every_policy(self, tmp_path):
        """The critic is a trainer too, and it comes last so the policies keep their slots."""
        args = _make_args(_write_megatron_config(tmp_path, "alpha", "beta"), use_critic=True)

        assert compute_trainer_roles(args) == ["alpha", "beta", CRITIC_TRAINER_ROLE]


class TestComputeTrainerModelId:
    def test_a_single_policy_trainer_has_no_model_id(self):
        """Stamping a model id on a single policy run would rename every artifact it writes."""
        assert compute_trainer_model_id(_make_args(), role=DEFAULT_TRAINER_ROLE) is None

    def test_a_multi_policy_trainer_is_identified_by_its_role(self, tmp_path):
        """The role is the model id, which is what rollout data and checkpoints are keyed by."""
        args = _make_args(_write_megatron_config(tmp_path, "alpha", "beta"))

        assert compute_trainer_model_id(args, role="beta") == "beta"

    def test_the_critic_has_no_model_id_even_under_multi_policy(self, tmp_path):
        """The critic serves the whole run, so it is not one of the policies rollout selects between."""
        args = _make_args(_write_megatron_config(tmp_path, "alpha", "beta"), use_critic=True)

        assert compute_trainer_model_id(args, role=CRITIC_TRAINER_ROLE) is None


class TestComputeTrainerPoolIds:
    @pytest.mark.parametrize("role", ["actor", "critic", "alpha"])
    def test_the_engine_pool_name_encodes_the_role(self, role):
        """Pool ids are the address book's keys, so two roles may never collide in it."""
        assert compute_trainer_pool_id(role) == f"trainer-engine-{role}"

    @pytest.mark.parametrize("role", ["actor", "critic", "alpha"])
    def test_the_controller_pool_name_encodes_the_role(self, role):
        """Each role's controller is its own worker, addressed apart from the ranks it drives."""
        assert compute_trainer_controller_pool_id(role) == f"trainer-controller-{role}"

    def test_a_controller_and_its_engines_never_share_a_pool_id(self):
        """One pool id for both would make the controller heal itself as if it were a rank."""
        assert compute_trainer_controller_pool_id("actor") != compute_trainer_pool_id("actor")
