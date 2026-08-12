from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

from argparse import Namespace

import pytest

from miles.rollout.router_addressing import compute_router_url, compute_sample_router_url
from miles.utils.types import Sample

_ROUTERS = {"actor": ("10.0.0.1", 3000), "ref": ("10.0.0.2", 3001)}


_ONE_ROUTER = {"actor": ("10.0.0.1", 3000)}


class TestComputeRouterUrl:
    def test_without_a_model_id_it_addresses_the_one_router_of_a_single_model_run(self):
        """Callers that predate multi-policy pass no model id, and such a run has exactly one router to mean."""
        args = Namespace(sglang_model_routers=_ONE_ROUTER)

        assert compute_router_url(args) == "http://10.0.0.1:3000"

    def test_without_a_model_id_it_is_refused_when_several_routers_serve_the_run(self):
        """Picking the first router would silently generate one policy's samples on another policy's engines."""
        args = Namespace(sglang_model_routers=_ROUTERS)

        with pytest.raises(AssertionError, match="no one router to go to"):
            compute_router_url(args)

    def test_a_named_model_reaches_its_own_router(self):
        """Each model is served by its own router, so its requests must not land on the primary one."""
        args = Namespace(sglang_model_routers=_ROUTERS)

        assert compute_router_url(args, model_id="ref", endpoint="/generate") == "http://10.0.0.2:3001/generate"

    def test_an_unknown_model_id_falls_back_only_when_one_router_serves_the_run(self):
        """A single-model run has nowhere else to go, so the fallback stays for it."""
        args = Namespace(sglang_model_routers=_ONE_ROUTER)

        assert compute_router_url(args, model_id="nonexistent") == "http://10.0.0.1:3000"

    def test_an_unknown_model_id_is_refused_when_several_routers_serve_the_run(self):
        """Falling back would send one policy's generations to another policy's engines, and nothing would say so."""
        args = Namespace(sglang_model_routers=_ROUTERS)

        with pytest.raises(AssertionError, match="no router to fall back on"):
            compute_router_url(args, model_id="nonexistent")

    def test_an_unresolved_router_map_is_a_wiring_bug(self):
        """The routers are resolved before any rollout runs, so a missing map must fail loud."""
        args = Namespace(sglang_model_routers=None)

        with pytest.raises(AssertionError, match="resolved before any rollout runs"):
            compute_router_url(args)


class TestComputeSampleRouterUrl:
    def test_a_sample_is_routed_by_the_policy_it_trains(self):
        """This is what makes a multi-policy run send each policy's requests to that policy's router."""
        args = Namespace(sglang_model_routers=_ROUTERS)
        sample = Sample(index=0, trainer_model_id="ref")

        assert compute_sample_router_url(args, sample, endpoint="/generate") == "http://10.0.0.2:3001/generate"

    def test_a_sample_without_a_policy_reaches_the_one_router_of_a_single_model_run(self):
        """Single-policy runs never stamp a trainer model id, so they must keep using their one router."""
        args = Namespace(sglang_model_routers=_ONE_ROUTER)
        sample = Sample(index=0)

        assert compute_sample_router_url(args, sample, endpoint="/generate") == "http://10.0.0.1:3000/generate"

    def test_a_sample_without_a_policy_is_refused_in_a_multi_policy_run(self):
        """A sample that lost its trainer model id must not quietly generate on the first policy's engines."""
        args = Namespace(sglang_model_routers=_ROUTERS)
        sample = Sample(index=0)

        with pytest.raises(AssertionError, match="no one router to go to"):
            compute_sample_router_url(args, sample, endpoint="/generate")
