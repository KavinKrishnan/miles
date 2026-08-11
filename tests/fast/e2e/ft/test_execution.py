from unittest.mock import patch

from tests.e2e.ft.conftest_ft.execution import run_training
from tests.e2e.ft.conftest_ft.modes import MODES

from miles.utils.external_utils import command_utils


class _StubBackend:
    def execute_train(self, **kwargs: object) -> None:
        pass


def test_the_training_run_uses_the_config_the_injector_was_pointed_at() -> None:
    """A second config would carry a run_id of its own, and the injector would fault a run nobody launched."""
    config = command_utils.ExecuteTrainConfig(run_id="260101-000000-000", namespace="rl")
    used: list[command_utils.ExecuteTrainConfig] = []

    with patch.object(command_utils.ExecuteTrainConfig, "create_backend", autospec=True) as create_backend:
        create_backend.side_effect = lambda self: used.append(self) or _StubBackend()
        run_training(train_args="--train-backend fsdp", mode=next(iter(MODES.values())), config=config)

    assert used == [config]
