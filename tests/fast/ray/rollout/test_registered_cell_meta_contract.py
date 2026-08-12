from __future__ import annotations

from functools import partial
from typing import Any

import pytest

from miles.ray.rollout.server_cell import compute_server_cell_meta_from_info, compute_server_cell_refusal_reason
from miles.utils.workers.registration.models import (
    REGISTERED_LAUNCH_GATE_TIMEOUT_SECONDS,
    RegisteredCell,
    RegisteredWorker,
    RegistrationAck,
    RegistrationSnapshot,
    compute_snapshot_digest,
)
from miles.utils.workers.registration.provider import RegistrationWorkerProvider
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import HostAndPort

_REPORTER = "west"
_POOL_ID = "west-inference-engine-0-0"

_WHOLE_META: dict[str, Any] = dict(
    model_id="default",
    worker_type="regular",
    num_gpus_per_engine=1,
    gpu_offset=0,
    sglang_api_key=None,
    needs_offload=False,
    update_weights=True,
)


class _Watcher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, CellInfo | None]] = []

    async def __call__(self, cell_id: str, observed: CellInfo | None) -> None:
        self.calls.append((cell_id, observed))

    @property
    def added(self) -> list[str]:
        return [cell_id for cell_id, observed in self.calls if observed is not None]


def _cell(cell_index: int, *, meta: dict[str, Any] | None = None) -> RegisteredCell:
    return RegisteredCell(
        cell_id=f"{_POOL_ID}-{cell_index}",
        pool_id=_POOL_ID,
        workers_hash="hash-1",
        workers=[
            RegisteredWorker(
                name=f"{_POOL_ID}-{cell_index}-0",
                addrs={"primary": HostAndPort(host="10.9.0.1", port=8000 + cell_index)},
                gpu_ids=[0],
            )
        ],
        meta=dict(_WHOLE_META) if meta is None else meta,
    )


def _snapshot(cells: list[RegisteredCell]) -> RegistrationSnapshot:
    expected = {"default": len(cells)}
    return RegistrationSnapshot(
        reporter_id=_REPORTER,
        epoch="epoch-1",
        sequence=1,
        digest=compute_snapshot_digest(cells=cells, expected_num_cells_by_model=expected),
        expected_num_cells_by_model=expected,
        cells=cells,
    )


def _provider(*, model_ids: set[str] | None = None) -> RegistrationWorkerProvider:
    return RegistrationWorkerProvider(
        expected_num_reporters=1,
        refuse_cell=partial(
            compute_server_cell_refusal_reason, model_ids={"default"} if model_ids is None else model_ids
        ),
    )


async def _apply_and_watch(
    provider: RegistrationWorkerProvider, cells: list[RegisteredCell]
) -> tuple[RegistrationAck, _Watcher]:
    watcher = _Watcher()
    await provider.watch_cells(watcher)
    ack = await provider.apply_snapshot(_snapshot(cells))
    await provider._wait_pending_dispatches()
    return ack, watcher


class TestTheMetadataContractOfARegisteredCell:
    @pytest.mark.parametrize("missing_field", sorted(set(_WHOLE_META) - {"worker_type"}))
    @pytest.mark.asyncio
    async def test_a_cell_missing_a_metadata_field_is_named_and_left_out(self, missing_field, caplog):
        """A peer running another version of miles must lose only its own cell, and be told which one."""
        meta = {name: value for name, value in _WHOLE_META.items() if name != missing_field}

        ack, watcher = await _apply_and_watch(_provider(), [_cell(0, meta=meta)])

        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]
        assert watcher.added == []
        assert f"{_POOL_ID}-0" in caplog.text and missing_field in caplog.text

    @pytest.mark.asyncio
    async def test_a_cell_whose_metadata_is_ill_typed_is_named_and_left_out(self, caplog):
        """A field this run cannot read is as unusable as a field that is not there at all."""
        meta = dict(_WHOLE_META) | dict(num_gpus_per_engine={"not": "a count"})

        ack, watcher = await _apply_and_watch(_provider(), [_cell(0, meta=meta)])

        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]
        assert watcher.added == []
        assert "num_gpus_per_engine" in caplog.text

    @pytest.mark.asyncio
    async def test_a_cell_of_a_model_this_run_does_not_serve_is_named_and_left_out(self, caplog):
        """Its engine would sit in a run whose router never sends it a request, and nothing would say why."""
        meta = dict(_WHOLE_META) | dict(model_id="some-other-policy")

        ack, watcher = await _apply_and_watch(_provider(), [_cell(0, meta=meta)])

        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]
        assert watcher.added == []
        assert "some-other-policy" in caplog.text

    @pytest.mark.asyncio
    async def test_a_refused_cell_never_becomes_an_exception(self):
        """An exception inside the reconcile would clear the digest every period and never name the offender."""
        provider = _provider()

        ack, _watcher = await _apply_and_watch(provider, [_cell(0, meta=dict(worker_type="regular"))])

        assert ack.applied_sequence == 1
        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]

    @pytest.mark.asyncio
    async def test_the_healthy_cells_a_refused_one_travelled_with_still_enter_the_run(self):
        """One cell built by an older peer must not keep a whole datacenter out of the run."""
        provider = _provider()

        ack, watcher = await _apply_and_watch(
            provider, [_cell(0, meta=dict(_WHOLE_META) | dict(model_id="unknown")), _cell(1)]
        )

        assert watcher.added == [f"{_POOL_ID}-1"]
        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]
        assert ack.applied_digest is None

    @pytest.mark.asyncio
    async def test_a_whole_snapshot_of_valid_cells_is_taken_in(self):
        """The contract must accept exactly what a peer of the same version reports."""
        provider = _provider()

        ack, watcher = await _apply_and_watch(provider, [_cell(0), _cell(1)])

        assert watcher.added == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]
        assert ack.excluded_cell_ids == []


class TestTheLaunchGateBudgetOfARegisteredCell:
    @pytest.mark.asyncio
    async def test_a_registered_cell_carries_the_shorter_gate_budget_into_its_metadata(self):
        """A cross datacenter gate that never answers must give up in a period, not in half an hour."""
        provider = _provider()

        _ack, watcher = await _apply_and_watch(provider, [_cell(0)])

        [(_cell_id, info)] = watcher.calls
        assert compute_server_cell_meta_from_info(info).launch_gate_timeout_seconds == (
            REGISTERED_LAUNCH_GATE_TIMEOUT_SECONDS
        )

    def test_a_cell_this_deployment_launched_itself_keeps_the_long_gate_budget(self):
        """Its gate is one pod away, and a local rollout may legitimately take many minutes to come up."""
        info = CellInfo(
            cell_id=f"{_POOL_ID}-0",
            pool_id=_POOL_ID,
            alive=True,
            worker_names=[f"{_POOL_ID}-0-0"],
            workers_hash="hash-1",
            meta=dict(_WHOLE_META),
        )

        assert compute_server_cell_meta_from_info(info).launch_gate_timeout_seconds > (
            REGISTERED_LAUNCH_GATE_TIMEOUT_SECONDS
        )
