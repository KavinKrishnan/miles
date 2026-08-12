from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from types import SimpleNamespace

from miles.backends.training_utils import log_utils
from miles.backends.training_utils.log_utils import log_train_step


def _capture(monkeypatch) -> tuple[dict, list[str]]:
    logged: dict = {}
    step_keys: list[str] = []
    monkeypatch.setattr(
        log_utils.tracking,
        "log",
        lambda _args, metrics, step_key, **_kwargs: (logged.update(metrics), step_keys.append(step_key)),
    )
    return logged, step_keys


class TestTrainerMetricNamespace:
    def test_a_single_policy_run_logs_the_train_keys_it_always_logged(self, monkeypatch):
        """Every existing dashboard and tracking query is written against these names."""
        logged, step_keys = _capture(monkeypatch)
        args = SimpleNamespace(trainer_model_id=None)

        log_train_step(
            args,
            loss_dict={"pg_loss": 0.5},
            grad_norm=1.0,
            rollout_id=3,
            step_id=1,
            num_steps_per_rollout=2,
            should_log=True,
        )

        assert step_keys == ["train/step"]
        assert logged["train/pg_loss"] == 0.5
        assert logged["train/step"] == 7

    def test_a_multi_policy_run_gives_every_policy_its_own_train_keys_and_step(self, monkeypatch):
        """Two trainers share one tracking run, so unprefixed keys interleave into one unreadable curve."""
        logged, step_keys = _capture(monkeypatch)
        args = SimpleNamespace(trainer_model_id="policy_b")

        log_train_step(
            args,
            loss_dict={"pg_loss": 0.5},
            grad_norm=1.0,
            rollout_id=3,
            step_id=1,
            num_steps_per_rollout=2,
            should_log=True,
        )

        assert step_keys == ["policy_b/train/step"]
        assert logged["policy_b/train/pg_loss"] == 0.5
        assert not any(key.startswith("train/") for key in logged)
