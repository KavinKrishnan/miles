"""Tests for process_identity module."""

import pytest
from pydantic import ValidationError

from miles.utils.audit_utils.process_identity import (
    SimpleProcessIdentity,
    TrainerControllerProcessIdentity,
    TrainProcessIdentity,
)


class TestProcessIdentityToName:
    def test_main(self) -> None:
        assert SimpleProcessIdentity(component="main").to_name() == "main"

    def test_rollout_executor(self) -> None:
        assert SimpleProcessIdentity(component="rollout_executor").to_name() == "rollout_executor"

    def test_actor(self) -> None:
        source = TrainProcessIdentity(component="actor", cell_index=1, rank_within_cell=3)
        assert source.to_name() == "actor_cell1_rank3"

    def test_critic(self) -> None:
        source = TrainProcessIdentity(component="critic", cell_index=0, rank_within_cell=2)
        assert source.to_name() == "critic_cell0_rank2"

    def test_actor_of_a_named_policy(self) -> None:
        """Two policies write side by side, so a rank's name has to carry the model it trains."""
        source = TrainProcessIdentity(component="actor", model_id="policy_a", cell_index=1, rank_within_cell=3)
        assert source.to_name() == "actor_policy_a_cell1_rank3"

    def test_two_policies_never_share_a_name(self) -> None:
        """Same cell and rank of two policies would otherwise overwrite each other's audit output."""
        first = TrainProcessIdentity(component="actor", model_id="policy_a", cell_index=0, rank_within_cell=0)
        second = TrainProcessIdentity(component="actor", model_id="policy_b", cell_index=0, rank_within_cell=0)
        assert first.to_name() != second.to_name()

    def test_trainer_controller(self) -> None:
        assert TrainerControllerProcessIdentity(role="actor").to_name() == "trainer_controller_actor"

    def test_a_policy_model_id_is_a_trainer_controller_role(self) -> None:
        """In a multi policy run a trainer's role is its model id, so the name has to accept any of them."""
        assert TrainerControllerProcessIdentity(role="policy_a").to_name() == "trainer_controller_policy_a"

    def test_inference_controller(self) -> None:
        assert SimpleProcessIdentity(component="inference_controller").to_name() == "inference_controller"

    def test_multi_lora_controller(self) -> None:
        assert SimpleProcessIdentity(component="multi_lora_controller").to_name() == "multi_lora_controller"

    def test_worker_manager(self) -> None:
        assert SimpleProcessIdentity(component="worker_manager").to_name() == "worker_manager"

    def test_an_unknown_component_is_rejected(self) -> None:
        """A simple identity only names the components that exist."""
        with pytest.raises(ValidationError):
            SimpleProcessIdentity(component="nope")


class TestControllerIdentityRoundtrip:
    def test_trainer_controller_keeps_its_role(self) -> None:
        """Two trainer controllers share a component, so only the role tells their events apart."""
        source = TrainerControllerProcessIdentity(role="critic")
        assert TrainerControllerProcessIdentity.model_validate_json(source.model_dump_json()) == source


class TestTrainProcessIdentityValidation:
    def test_negative_cell_index_rejected(self) -> None:
        """A negative cell_index fails validation."""
        with pytest.raises(ValidationError):
            TrainProcessIdentity(component="actor", cell_index=-1, rank_within_cell=0)

    def test_negative_rank_within_cell_rejected(self) -> None:
        """A negative rank_within_cell fails validation."""
        with pytest.raises(ValidationError):
            TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=-1)


class TestTrainProcessIdentityRoundtrip:
    def test_serialize_deserialize(self) -> None:
        source = TrainProcessIdentity(component="actor", cell_index=2, rank_within_cell=0)
        parsed = TrainProcessIdentity.model_validate_json(source.model_dump_json())
        assert parsed.cell_index == 2
        assert parsed.rank_within_cell == 0
        assert parsed.component == "actor"
        assert parsed.model_id is None

    def test_serialize_deserialize_keeps_the_model_id(self) -> None:
        """The identity crosses a process boundary, and a dropped model id merges two policies' records."""
        source = TrainProcessIdentity(component="actor", model_id="policy_b", cell_index=2, rank_within_cell=0)
        parsed = TrainProcessIdentity.model_validate_json(source.model_dump_json())
        assert parsed.model_id == "policy_b"
