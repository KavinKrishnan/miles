from __future__ import annotations

import textwrap

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.ray.specs.inference import compute_inference_controller_provider, compute_router_providers
from miles.ray.specs.static_addrs import inference_controller_urls, static_router_addrs, trainer_controller_urls
from miles.ray.specs.train import compute_trainer_controller_provider
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider


def _args(tmp_path, **overrides):
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        make_sglang_config_yaml(server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 4}])
    )
    return make_args(sglang_config=str(config_path), rollout_num_gpus=8, **overrides)


def _two_model_args(tmp_path, **overrides):
    config_path = tmp_path / "sglang-two-models.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            sglang:
              - name: a
                server_groups:
                  - worker_type: regular
                    num_gpus: 8
                    num_gpus_per_engine: 4
              - name: b
                server_groups:
                  - worker_type: regular
                    num_gpus: 8
                    num_gpus_per_engine: 4
            """
        )
    )
    return make_args(sglang_config=str(config_path), rollout_num_gpus=16, **overrides)


class TestTrainerControllerUrls:
    def test_a_run_without_the_flag_names_no_controller(self, tmp_path):
        """An all-in-one run finds its own controller, so nothing may be invented for it."""
        assert trainer_controller_urls(_args(tmp_path), role="actor") is None

    def test_a_bare_address_belongs_to_the_actor(self, tmp_path):
        """Most runs train one model, and writing 'actor=' on every launch is noise."""
        args = _args(tmp_path, trainer_controller_addrs=["10.0.0.1:8000"])

        assert trainer_controller_urls(args, role="actor") == ["10.0.0.1:8000"]

    def test_each_role_is_addressed_separately(self, tmp_path):
        """A critic is its own controller in its own pod, and calling the actor's would train the wrong model."""
        args = _args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000", "critic=10.0.0.2:8000"])

        assert trainer_controller_urls(args, role="critic") == ["10.0.0.2:8000"]

    def test_refuses_a_role_that_is_not_one_of_the_run_s(self, tmp_path):
        """A typo would otherwise leave the role it meant to name silently unaddressed."""
        args = _args(tmp_path, trainer_controller_addrs=["actro=10.0.0.1:8000"])

        with pytest.raises(AssertionError, match="not one of"):
            trainer_controller_urls(args, role="actor")

    def test_refuses_a_run_whose_critic_was_left_unaddressed(self, tmp_path):
        """A critic named by nothing would be reached at the actor's controller and train the wrong model."""
        args = _args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000"])

        with pytest.raises(AssertionError, match="exactly one"):
            trainer_controller_urls(args, role="critic")

    def test_refuses_two_controllers_for_one_role(self, tmp_path):
        """Driving several trainers is a composite controller's job, and silently using one would drop the other."""
        args = _args(tmp_path, trainer_controller_addrs=["actor=10.0.0.1:8000", "actor=10.0.0.2:8000"])

        with pytest.raises(AssertionError, match="exactly one"):
            trainer_controller_urls(args, role="actor")


class TestInferenceControllerUrls:
    def test_a_run_without_the_flag_names_no_controller(self, tmp_path):
        """An all-in-one run finds its own controller, so nothing may be invented for it."""
        assert inference_controller_urls(_args(tmp_path)) is None

    def test_refuses_more_than_one_until_a_composite_controller_can_fan_out(self, tmp_path):
        """The flag takes a list to keep that door open, but nothing today knows how to use two."""
        args = _args(tmp_path, inference_controller_addrs=["10.0.0.1:8000", "10.0.0.2:8000"])

        with pytest.raises(AssertionError, match="exactly one"):
            inference_controller_urls(args)


class TestStaticRouterAddrs:
    def test_a_single_model_run_may_write_a_bare_address(self, tmp_path):
        """One model has one router, so naming it is redundant."""
        addrs = static_router_addrs(_args(tmp_path, inference_router_addrs=["10.0.0.3:8000"]))

        assert [addr.addr for addr in addrs.values()] == ["http://10.0.0.3:8000"]

    def test_keys_the_routers_in_the_order_the_models_are_configured(self, tmp_path):
        """The driver records the first entry as the run's default router, so it has to be the first model's."""
        args = _two_model_args(tmp_path, inference_router_addrs=["b=10.0.0.4:8000", "a=10.0.0.3:8000"])

        addrs = static_router_addrs(args)

        assert list(addrs) == ["a", "b"]
        assert next(iter(addrs.values())).addr == "http://10.0.0.3:8000"

    def test_every_model_has_to_be_given_its_own_router(self, tmp_path):
        """A model whose router is unknown would send its requests nowhere at all."""
        args = _two_model_args(tmp_path, inference_router_addrs=["a=10.0.0.3:8000"])

        with pytest.raises(AssertionError, match="needs an entry"):
            static_router_addrs(args)


class TestProviderSelection:
    def test_a_given_trainer_controller_address_is_used_instead_of_the_backend_s(self, tmp_path):
        """The trainer lives in another deployment, whose names this one's backend cannot resolve."""
        args = _args(tmp_path, trainer_controller_addrs=["10.0.0.1:8000"])
        capability = FakeBackendCapability(static_provider=object())

        provider = compute_trainer_controller_provider(args, capability=capability, role="actor")

        assert isinstance(provider, SimpleWorkerProvider)
        assert capability.requested_static_pool_ids == []

    def test_an_all_in_one_run_still_asks_its_own_backend_for_the_trainer_controller(self, tmp_path):
        """Nothing addresses a ray actor statically, so the all-in-one path must be untouched."""
        capability = FakeBackendCapability(static_provider=object())

        provider = compute_trainer_controller_provider(_args(tmp_path), capability=capability, role="actor")

        assert provider is capability.static_provider
        assert capability.requested_static_pool_ids == ["trainer-controller-actor"]

    def test_a_given_inference_controller_address_is_used_instead_of_the_backend_s(self, tmp_path):
        """Same for the inference side: its pod names belong to the release that installed it."""
        args = _args(tmp_path, inference_controller_addrs=["10.0.0.2:8000"])
        capability = FakeBackendCapability(static_provider=object())

        provider = compute_inference_controller_provider(args, capability=capability)

        assert isinstance(provider, SimpleWorkerProvider)
        assert capability.requested_static_pool_ids == []

    def test_given_router_addresses_leave_the_router_pools_unasked_for(self, tmp_path):
        """The routers are installed by the inference deployment, so this release holds no pool to observe."""
        args = _args(tmp_path, inference_router_addrs=["10.0.0.3:8000"])
        capability = FakeBackendCapability(static_provider=object())

        assert compute_router_providers(args, capability=capability) == []
        assert capability.requested_static_pool_ids == []
