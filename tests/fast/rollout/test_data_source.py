import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.rollout import data_source
from miles.rollout.data_source import RolloutDataSource, RolloutDataSourceWithBuffer


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        rollout_global_dataset=False,
        save=None,
        load=None,
        rollout_shuffle=False,
        buffer_filter_path=None,
        hf_checkpoint=None,
        chat_template_path=None,
        dump_details=None,
        prompt_data=None,
        rollout_max_prompt_len=None,
        input_key=None,
        multimodal_keys=None,
        label_key=None,
        metadata_key=None,
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=0,
    )
    return SimpleNamespace(**{**defaults, **overrides})


def _written_names(root: Path) -> list[str]:
    return sorted(path.name for path in (root / "rollout").iterdir())


@pytest.fixture
def global_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_source, "load_tokenizer", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(data_source, "load_processor", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(data_source, "Dataset", lambda *args, **kwargs: SimpleNamespace(samples=[]))


class _PersistingBufferSource(RolloutDataSourceWithBuffer):
    def save_buffer(self, rollout_id: int) -> None:
        path = Path(self.args.save) / "rollout" / f"buffer_{rollout_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.buffer))

    def load_buffer(self, rollout_id: int | None = None) -> None:
        self.buffer = json.loads((Path(self.args.load) / "rollout" / f"buffer_{rollout_id}.json").read_text())


def test_save_writes_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The built-in source guards itself, so the executor needs no outer guard to keep it silent."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))

    source.save(rollout_id=3)

    assert list(tmp_path.iterdir()) == []


def test_load_reads_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The load side has always been called unconditionally and relies on the same internal guard."""
    source = RolloutDataSource(_make_args(load=str(tmp_path)))

    source.load(rollout_id=3)

    assert source.sample_offset == 0
    assert source.epoch_id == 0


class TestBufferPersistenceHooks:
    def test_save_calls_the_buffer_hook_with_the_rollout_id(self, tmp_path: Path) -> None:
        """The buffer belongs to this process, and a replay buffer persists it through exactly this hook."""
        source = _BufferSourceSpy(_make_args(save=str(tmp_path)))

        source.save(rollout_id=5)

        assert source.saved == [5]

    def test_load_calls_the_buffer_hook_with_the_rollout_id(self, tmp_path: Path) -> None:
        """A restarted rollout executor restores its backlog here, or accepts losing it."""
        source = _BufferSourceSpy(_make_args(load=str(tmp_path)))

        source.load(rollout_id=4)

        assert source.loaded == [4]

    def test_the_default_hooks_write_the_dataset_state_and_nothing_else(
        self, tmp_path: Path, global_dataset: None
    ) -> None:
        """Losing the backlog on a restart is the accepted behaviour, and it must be the visible default."""
        source = RolloutDataSourceWithBuffer(_make_args(save=str(tmp_path), rollout_global_dataset=True))
        source.buffer = [["a"]]

        source.save(rollout_id=1)

        assert _written_names(tmp_path) == ["global_dataset_state_dict_1.pt"]

    def test_a_restarted_source_starts_with_an_empty_buffer(self, tmp_path: Path, global_dataset: None) -> None:
        """This is what a hot restart costs: the pending samples of the previous script are gone."""
        first = RolloutDataSourceWithBuffer(_make_args(save=str(tmp_path), rollout_global_dataset=True))
        first.buffer = [["a"]]
        first.save(rollout_id=1)

        second = RolloutDataSourceWithBuffer(_make_args(load=str(tmp_path), rollout_global_dataset=True))
        second.load(rollout_id=1)

        assert second.buffer == []

    def test_a_source_that_implements_the_pair_carries_its_buffer_across_objects(
        self, tmp_path: Path, global_dataset: None
    ) -> None:
        """These two hooks are the extension point a replay buffer plugs into; nothing else has to move."""
        first = _PersistingBufferSource(_make_args(save=str(tmp_path), rollout_global_dataset=True))
        first.buffer = [["a"], ["b"]]
        first.save(rollout_id=1)

        second = _PersistingBufferSource(
            _make_args(save=str(tmp_path), load=str(tmp_path), rollout_global_dataset=True)
        )
        second.load(rollout_id=1)

        assert second.buffer == [["a"], ["b"]]
        assert sorted(_written_names(tmp_path)) == ["buffer_1.json", "global_dataset_state_dict_1.pt"]


class _BufferSourceSpy(RolloutDataSourceWithBuffer):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.saved: list[int] = []
        self.loaded: list[int | None] = []

    def save_buffer(self, rollout_id: int) -> None:
        self.saved.append(rollout_id)

    def load_buffer(self, rollout_id: int | None = None) -> None:
        self.loaded.append(rollout_id)
