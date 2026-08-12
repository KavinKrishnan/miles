from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])

import pytest

from miles.rollout.multi_policy import TrainerModelRouter
from miles.utils.types import Sample


def _samples(*trainer_model_ids: str | None) -> list[Sample]:
    return [Sample(index=i, trainer_model_id=model_id) for i, model_id in enumerate(trainer_model_ids)]


class TestTrainerModelRouterSinglePolicy:
    @staticmethod
    def _router() -> TrainerModelRouter:
        return TrainerModelRouter(["default"])

    def test_none_resolves_to_the_only_policy(self):
        """Existing single policy runs must keep working without touching their rollout function."""
        assert self._router().resolve_model_id(None) == "default"

    def test_an_explicit_matching_id_is_accepted(self):
        """A rollout function that already sets the id must not be punished for it."""
        assert self._router().resolve_model_id("default") == "default"

    def test_an_unknown_id_fails_loudly(self):
        """A typo would otherwise train nothing and stall the run on an empty queue."""
        with pytest.raises(AssertionError, match="unknown trainer_model_id"):
            self._router().resolve_model_id("typo")

    def test_a_group_that_names_no_policy_routes_to_the_only_one(self):
        """A single policy run never sets the field, so an unset group must still find its queue."""
        assert self._router().resolve_group_model_id(_samples(None, None)) == "default"


class TestTrainerModelRouterMultiPolicy:
    @staticmethod
    def _router() -> TrainerModelRouter:
        return TrainerModelRouter(["a", "b"])

    def test_none_fails_loudly_before_enqueueing(self):
        """Defaulting to the primary would silently train one policy on another's data."""
        with pytest.raises(AssertionError, match="trainer_model_id is required"):
            self._router().resolve_model_id(None)

    def test_an_unknown_id_fails_loudly(self):
        """The known ids come from --megatron-config; anything else is a user bug, not a fallback."""
        with pytest.raises(AssertionError, match="unknown trainer_model_id"):
            self._router().resolve_model_id("c")

    def test_a_group_routes_to_the_policy_all_its_samples_name(self):
        """Grouping happens per prompt group, which is the unit the buffer stores."""
        assert self._router().resolve_group_model_id(_samples("b", "b")) == "b"

    def test_a_group_split_across_policies_fails_loudly(self):
        """Group-relative advantages are meaningless once a group is split across policies."""
        with pytest.raises(AssertionError, match="exactly one policy model"):
            self._router().resolve_group_model_id(_samples("a", "b"))
