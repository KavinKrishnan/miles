from __future__ import annotations

from typing import Any

import pytest

from miles.ray.train.composite import CompositeTrainerController


class _FakeTrainer:
    def __init__(self, role: str) -> None:
        self.role = role
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.ready_timeouts: list[float] = []

    async def wait_ready(self, *, timeout: float) -> None:
        self.ready_timeouts.append(timeout)

    async def init(self, args) -> list[Any]:
        self.calls.append(("init", {"args": args}))
        return [f"{self.role}-init"]

    async def train(
        self,
        *,
        rollout_id: int,
        rollout_data_pack: dict[str, Any],
        external_data: list[Any] | None,
    ) -> list[Any]:
        self.calls.append(
            (
                "train",
                {
                    "rollout_id": rollout_id,
                    "rollout_data_pack": rollout_data_pack,
                    "external_data": external_data,
                },
            )
        )
        return [f"{self.role}-train"]

    async def save_model(self, *, rollout_id: int, force_sync: bool) -> None:
        self.calls.append(("save_model", {"rollout_id": rollout_id, "force_sync": force_sync}))

    async def onload(self) -> None:
        self.calls.append(("onload", {}))

    async def offload(self) -> None:
        self.calls.append(("offload", {}))

    async def clear_memory(self) -> None:
        self.calls.append(("clear_memory", {}))

    async def reconcile_adapters(self) -> None:
        self.calls.append(("reconcile_adapters", {}))

    async def dispose(self) -> None:
        self.calls.append(("dispose", {}))

    def method_names(self) -> list[str]:
        return [name for name, _ in self.calls]


def _make_composite(*roles: str) -> tuple[CompositeTrainerController, dict[str, _FakeTrainer]]:
    trainers = {role: _FakeTrainer(role=role) for role in roles}
    return CompositeTrainerController(trainers=trainers), trainers


class TestCompositeTrainerController:
    async def test_a_named_model_id_reaches_exactly_that_trainer(self):
        """Routing by model id must drive the named trainer and leave the others untouched."""
        composite, trainers = _make_composite("actor", "critic")

        result = await composite.train(
            rollout_id=4,
            rollout_data_pack={"payload": "x"},
            external_data=["e"],
            model_id="critic",
        )

        assert result == ["critic-train"]
        assert trainers["critic"].calls == [
            ("train", {"rollout_id": 4, "rollout_data_pack": {"payload": "x"}, "external_data": ["e"]})
        ]
        assert trainers["actor"].calls == []

    async def test_an_unknown_model_id_raises(self):
        """A typo'd or undeployed model id must fail loudly instead of silently picking a trainer."""
        composite, _ = _make_composite("actor", "critic")

        with pytest.raises(AssertionError, match="no trainer is deployed for model 'reward'"):
            await composite.init(object(), model_id="reward")

    async def test_a_one_instance_composite_accepts_calls_without_a_model_id(self):
        """A single-trainer run must keep working for every caller that never names a model."""
        composite, trainers = _make_composite("actor")
        direct = _FakeTrainer(role="actor")

        through_composite = await composite.train(rollout_id=1, rollout_data_pack={"payload": "y"}, external_data=None)
        directly = await direct.train(rollout_id=1, rollout_data_pack={"payload": "y"}, external_data=None)

        assert through_composite == directly
        assert trainers["actor"].calls == direct.calls

    async def test_a_multi_instance_composite_rejects_calls_without_a_model_id(self):
        """With several trainers there is no sane default, so an unnamed call must raise."""
        composite, _ = _make_composite("actor", "critic")

        with pytest.raises(AssertionError, match="every call has to name the model it drives"):
            await composite.save_model(rollout_id=2, force_sync=True)

    async def test_wait_ready_waits_for_every_trainer(self):
        """Training may only start once all trainers of the run are up, not just the first one."""
        composite, trainers = _make_composite("actor", "critic")

        await composite.wait_ready(timeout=12.5)

        assert trainers["actor"].ready_timeouts == [12.5]
        assert trainers["critic"].ready_timeouts == [12.5]

    @pytest.mark.parametrize("method_name", ["onload", "offload", "clear_memory", "dispose"])
    async def test_a_fan_out_method_without_a_model_id_reaches_every_trainer(self, method_name: str):
        """Memory and lifecycle calls are run-wide, so an unnamed call must hit all trainers."""
        composite, trainers = _make_composite("actor", "critic")

        await getattr(composite, method_name)()

        assert trainers["actor"].method_names() == [method_name]
        assert trainers["critic"].method_names() == [method_name]

    @pytest.mark.parametrize("method_name", ["onload", "offload", "clear_memory", "dispose"])
    async def test_a_fan_out_method_with_a_model_id_reaches_only_that_trainer(self, method_name: str):
        """Naming a model must narrow a fan-out call down to that one trainer."""
        composite, trainers = _make_composite("actor", "critic")

        await getattr(composite, method_name)(model_id="actor")

        assert trainers["actor"].method_names() == [method_name]
        assert trainers["critic"].method_names() == []

    async def test_model_ids_lists_every_deployed_trainer(self):
        """Callers pick a trainer by model id, so the composite must expose the ids it routes."""
        composite, _ = _make_composite("actor", "critic")

        assert sorted(composite.model_ids) == ["actor", "critic"]

    async def test_a_composite_without_trainers_is_rejected(self):
        """A composite that fans out over nothing would swallow every call it is given."""
        with pytest.raises(AssertionError, match="at least one trainer"):
            CompositeTrainerController(trainers={})
