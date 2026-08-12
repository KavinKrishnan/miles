from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from miles.utils.arguments import CHECKPOINT_SOURCE_DEFAULTS
from miles.utils.init_once import InitOnce

_CHECKPOINT_ROLLOUT_ID = 41


class _RecordingBackuper:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def backup(self, tag: str) -> None:
        self._events.append(f"backup:{tag}")


def _checkpoint_dir(tmp_path: Path, name: str, *, written: bool = True) -> str:
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    if written:
        (path / "latest_checkpointed_iteration.txt").write_text(str(_CHECKPOINT_ROLLOUT_ID))
    return str(path)


def _worker(
    actor_module: Any,
    tmp_path: Path,
    *,
    role: str = "actor",
    with_ref: bool = False,
    with_opd_teacher: bool = False,
    asleep: bool = False,
    initialized: bool = True,
    requested_checkpoint_source: dict[str, Any] | None = None,
    **arg_overrides: Any,
) -> Any:
    worker = object.__new__(actor_module.MegatronTrainRayActor)
    arg_values = {
        "debug_rollout_only": False,
        "offload_train": False,
        "keep_old_actor": False,
        "update_weights_interval": 1,
        "colocate": False,
        "megatron_to_hf_mode": "raw",
        "load": _checkpoint_dir(tmp_path, "actor"),
        "ref_load": _checkpoint_dir(tmp_path, "ref"),
        "critic_load": _checkpoint_dir(tmp_path, "critic"),
        "opd_teacher_load": _checkpoint_dir(tmp_path, "teacher"),
        "hf_checkpoint": None,
        "ref_ckpt_step": None,
        "ckpt_step": None,
        "finetune": False,
        "no_load_optim": False,
        "no_load_rng": False,
        "start_rollout_id": 0,
        **arg_overrides,
    }
    arg_values["requested_checkpoint_source"] = (
        {name: arg_values[name] for name in CHECKPOINT_SOURCE_DEFAULTS}
        if requested_checkpoint_source is None
        else requested_checkpoint_source
    )
    worker.args = Namespace(**arg_values)
    worker.role = role
    worker.with_ref = with_ref
    worker.with_opd_teacher = with_opd_teacher
    worker._asleep = asleep
    worker._init_once = InitOnce(component="MegatronTrainRayActor")
    if initialized:
        worker._init_once.enter()
    worker.model = ["model"]
    worker.optimizer = "optimizer"
    worker.opt_param_scheduler = "scheduler"
    worker._reload_checkpointing_context = {"local_checkpoint_manager": "stashed"}

    worker.events: list[str] = []
    worker.weights_backuper = _RecordingBackuper(worker.events)
    worker.load_other_checkpoint = Mock(side_effect=lambda tag, path: worker.events.append(f"load:{tag}:{path}"))
    worker._switch_model = Mock(side_effect=lambda tag: worker.events.append(f"switch:{tag}"))
    worker.wake_up = Mock(side_effect=lambda: worker.events.append("wake_up"))
    worker.sleep = Mock(side_effect=lambda: worker.events.append("sleep"))
    return worker


@pytest.fixture
def reload_calls(actor_module: Any, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _fake_load_model_state(args: Namespace, **kwargs: Any) -> int:
        calls.append(kwargs)
        return _CHECKPOINT_ROLLOUT_ID

    monkeypatch.setattr(actor_module, "load_model_state", _fake_load_model_state)
    monkeypatch.setattr(actor_module, "clear_memory", lambda *a, **kw: None)
    return calls


class TestTheTrainerReload:
    def test_it_answers_the_rollout_id_to_resume_at(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """A restarted orchestration script never called init, so this answer is its only resume point."""
        worker = _worker(actor_module, tmp_path)

        assert worker.load_state() == _CHECKPOINT_ROLLOUT_ID + 1
        assert worker._last_rollout_id == _CHECKPOINT_ROLLOUT_ID

    def test_a_trainer_init_never_built_refuses_to_reload(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """There is no model to load into, and an attribute error would hide why."""
        worker = _worker(actor_module, tmp_path, initialized=False)

        with pytest.raises(AssertionError, match="init already built"):
            worker.load_state()

    def test_it_reloads_through_the_checkpointing_context_init_stashed(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """An in-memory checkpoint manager lives in the trainer process and cannot be rebuilt by the new script."""
        worker = _worker(actor_module, tmp_path)

        worker.load_state()

        assert reload_calls == [
            dict(
                model=["model"],
                optimizer="optimizer",
                opt_param_scheduler="scheduler",
                role="actor",
                checkpointing_context={"local_checkpoint_manager": "stashed"},
            )
        ]

    def test_a_sleeping_trainer_is_woken_before_the_reload(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """load_checkpoint writes into gpu memory that offload has taken away."""
        worker = _worker(actor_module, tmp_path, asleep=True)

        worker.load_state()

        assert worker.events[0] == "wake_up"

    def test_an_offloaded_trainer_goes_back_to_sleep_last(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """--offload-train leaves the trainer offloaded between steps, exactly as init does."""
        worker = _worker(actor_module, tmp_path, offload_train=True)

        worker.load_state()

        assert worker.events[-1] == "sleep"

    def test_the_reference_model_is_reloaded_from_its_own_checkpoint(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """--ref-update-interval refreshes the ref slot in place, so a rollback that skips it keeps a future ref."""
        worker = _worker(actor_module, tmp_path, with_ref=True)

        worker.load_state()

        assert f"load:ref:{worker.args.ref_load}" in worker.events

    def test_a_run_without_a_reference_model_reloads_none(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """There is no ref slot to restore, and reading --ref-load would load a checkpoint nobody asked for."""
        worker = _worker(actor_module, tmp_path, with_ref=False)

        worker.load_state()

        assert not [event for event in worker.events if event.startswith("load:ref")]

    def test_the_slots_are_refreshed_in_the_order_init_builds_them(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """Every load_other_checkpoint leaves the active tag on what it loaded, so the order decides the result."""
        worker = _worker(actor_module, tmp_path, with_ref=True, keep_old_actor=True)

        worker.load_state()

        assert worker.events == [
            "backup:actor",
            f"load:ref:{worker.args.ref_load}",
            f"load:old_actor:{worker.args.load}",
            "backup:rollout_actor",
            "switch:actor",
        ]

    def test_the_adapter_bookkeeping_is_reset(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """The reload replaced the base weights, so nothing the previous script pushed is still current."""
        worker = _worker(actor_module, tmp_path)
        worker.loaded_adapters = {"a": object()}
        worker._multi_lora_pending_push = {"a"}

        worker.load_state()

        assert worker.loaded_adapters == {}
        assert worker._multi_lora_pending_push == set()

    def test_a_critic_reloads_its_own_state_and_touches_no_backup_slot(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """A critic has no ref, no old actor and no rollout actor; init builds none of them either."""
        worker = _worker(actor_module, tmp_path, role="critic")

        assert worker.load_state() == _CHECKPOINT_ROLLOUT_ID + 1
        assert worker.events == []

    def test_a_debug_rollout_only_trainer_reloads_nothing(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """It never built a model, exactly as its init returned before building one."""
        worker = _worker(actor_module, tmp_path, debug_rollout_only=True)

        assert worker.load_state() == 0
        assert reload_calls == []

    def test_the_distillation_teacher_is_reloaded_too(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """init builds the teacher slot as well, and a rollback that skips it is a slot the run never restores."""
        worker = _worker(actor_module, tmp_path, with_opd_teacher=True)

        worker.load_state()

        assert f"load:teacher:{worker.args.opd_teacher_load}" in worker.events


class TestTheCheckpointSourceARestartLoadsFrom:
    def test_a_hot_restart_loads_the_checkpoint_the_run_wrote_rather_than_the_reference(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """A fresh run's parse pointed --load at --ref-load, and reloading that would undo the whole run."""
        worker = _stale_worker(actor_module, tmp_path)

        worker.load_state()

        assert worker.args.load == _checkpoint_dir(tmp_path, "actor")
        assert (worker.args.finetune, worker.args.no_load_optim, worker.args.no_load_rng) == (False, False, False)

    def test_the_checkpoint_step_the_reference_pinned_is_dropped_with_it(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """--ref-ckpt-step pinned a step of the reference checkpoint, which names nothing in the run's own one."""
        worker = _stale_worker(actor_module, tmp_path, ckpt_step=7, ref_ckpt_step=7)

        worker.load_state()

        assert worker.args.ckpt_step is None

    def test_a_run_that_has_not_written_a_checkpoint_yet_still_starts_from_the_reference(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """Nothing was trained yet, so the derivation a fresh parse makes is still the right one."""
        worker = _stale_worker(actor_module, tmp_path, checkpoint_written=False)

        worker.load_state()

        assert worker.args.load == worker.args.ref_load
        assert worker.args.finetune is True

    def test_a_critic_reloads_its_own_checkpoint_after_the_rederivation(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """init points a critic at --critic-load after the parse, and the reload has to end in the same place."""
        worker = _stale_worker(actor_module, tmp_path, role="critic")

        worker.load_state()

        assert worker.args.load == _checkpoint_dir(tmp_path, "critic")

    def test_a_bridge_run_reloads_its_own_checkpoint_and_keeps_zeroing_the_rollout_id(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """The bridge branch derives the same load dir a fresh parse of the same command would, zeroed id included."""
        worker = _stale_worker(actor_module, tmp_path, megatron_to_hf_mode="bridge", start_rollout_id=9)

        worker.load_state()

        assert worker.args.load == _checkpoint_dir(tmp_path, "actor")
        assert worker.args.start_rollout_id == 0

    def test_a_user_who_asked_for_finetune_keeps_it(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """A cold restart of the same command would honor --finetune, and a hot one has to land where it lands."""
        worker = _stale_worker(actor_module, tmp_path, requested_finetune=True)

        worker.load_state()

        assert worker.args.finetune is True

    def test_a_second_reload_derives_exactly_what_the_first_one_did(
        self, actor_module: Any, reload_calls: list[dict], tmp_path: Path
    ) -> None:
        """Nothing on disk moved between them, so a derivation reading its own output would drift on every restart."""
        worker = _stale_worker(actor_module, tmp_path)
        worker.load_state()
        once = {name: getattr(worker.args, name) for name in CHECKPOINT_SOURCE_DEFAULTS}

        worker.load_state()

        assert {name: getattr(worker.args, name) for name in CHECKPOINT_SOURCE_DEFAULTS} == once


def _stale_worker(
    actor_module: Any,
    tmp_path: Path,
    *,
    role: str = "actor",
    ckpt_step: int | None = None,
    ref_ckpt_step: int | None = None,
    checkpoint_written: bool = True,
    requested_finetune: bool = False,
    **arg_overrides: Any,
) -> Any:
    requested_load = (
        _checkpoint_dir(tmp_path, "actor") if checkpoint_written else _checkpoint_dir(tmp_path, "fresh", written=False)
    )
    return _worker(
        actor_module,
        tmp_path,
        role=role,
        load=_checkpoint_dir(tmp_path, "ref"),
        ckpt_step=ckpt_step,
        ref_ckpt_step=ref_ckpt_step,
        finetune=True,
        no_load_optim=True,
        no_load_rng=True,
        requested_checkpoint_source=dict(
            load=requested_load,
            ckpt_step=None,
            finetune=requested_finetune,
            no_load_optim=False,
            no_load_rng=False,
        ),
        **arg_overrides,
    )
