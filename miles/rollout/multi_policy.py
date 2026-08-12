import logging
from collections.abc import Iterable

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class TrainerModelRouter:
    def __init__(self, model_ids: list[str]) -> None:
        assert model_ids, "a run must train at least one policy model"
        self._model_ids = list(model_ids)

    @property
    def model_ids(self) -> list[str]:
        return list(self._model_ids)

    @property
    def is_multi_policy(self) -> bool:
        return len(self._model_ids) > 1

    def resolve_model_id(self, trainer_model_id: str | None) -> str:
        if trainer_model_id is None:
            assert not self.is_multi_policy, (
                f"trainer_model_id is required when training multiple policy models {self._model_ids}; "
                f"the custom rollout function must set Sample.trainer_model_id"
            )
            return self._model_ids[0]
        assert trainer_model_id in self._model_ids, (
            f"unknown trainer_model_id {trainer_model_id!r}, known ids are {self._model_ids}; "
            f"the ids come from the --megatron-config model names"
        )
        return trainer_model_id

    def resolve_group_model_id(self, samples: Iterable[Sample]) -> str:
        found = {sample.trainer_model_id for sample in samples}
        assert len(found) == 1, (
            f"one prompt group must train exactly one policy model, got {sorted(map(str, found))}; "
            f"group-relative advantages make a group meaningless once it is split across policies"
        )
        [trainer_model_id] = found
        return self.resolve_model_id(trainer_model_id)
