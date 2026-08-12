from __future__ import annotations

import asyncio
import logging

import pytest

from miles.utils.workers.registration.models import (
    RegisteredCell,
    RegisteredWorker,
    RegistrationAck,
    RegistrationSnapshot,
    compute_snapshot_digest,
)
from miles.utils.workers.registration.provider import EPOCH_CHURN_ERROR_SECONDS, RegistrationWorkerProvider
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import HostAndPort

_REPORTER = "west"
_POOL_ID = "west-inference-engine-0-0"
_EPOCH = "epoch-1"


class _Watcher:
    def __init__(self, *, failing_cell_ids: set[str] | None = None) -> None:
        self.calls: list[tuple[str, CellInfo | None]] = []
        self._failing_cell_ids = set(failing_cell_ids or set())

    async def __call__(self, cell_id: str, observed: CellInfo | None) -> None:
        self.calls.append((cell_id, observed))
        if cell_id in self._failing_cell_ids:
            self._failing_cell_ids.discard(cell_id)
            raise RuntimeError(f"reconciling {cell_id} failed once")

    @property
    def added(self) -> list[str]:
        return [cell_id for cell_id, observed in self.calls if observed is not None]

    @property
    def removed(self) -> list[str]:
        return [cell_id for cell_id, observed in self.calls if observed is None]


class _BlockingWatcher(_Watcher):
    def __init__(self, *, blocked_cell_ids: set[str]) -> None:
        super().__init__()
        self._blocked_cell_ids = set(blocked_cell_ids)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, cell_id: str, observed: CellInfo | None) -> None:
        if cell_id in self._blocked_cell_ids:
            self._blocked_cell_ids.discard(cell_id)
            self.entered.set()
            await self.release.wait()
        await super().__call__(cell_id, observed)


def _cell(
    cell_index: int, *, workers_hash: str = "hash-1", worker_type: str = "regular", port: int | None = None
) -> RegisteredCell:
    return RegisteredCell(
        cell_id=f"{_POOL_ID}-{cell_index}",
        pool_id=_POOL_ID,
        workers_hash=workers_hash,
        workers=[
            RegisteredWorker(
                name=f"{_POOL_ID}-{cell_index}-0",
                addrs={"primary": HostAndPort(host="10.9.0.1", port=port if port is not None else 8000 + cell_index)},
                gpu_ids=[0],
            )
        ],
        meta=dict(model_id="default", worker_type=worker_type, num_gpus_per_engine=1),
    )


def _snapshot(
    cells: list[RegisteredCell] | None,
    *,
    sequence: int,
    expected: dict[str, int] | None = None,
    digest: str | None = None,
    token: str | None = None,
    reporter_id: str = _REPORTER,
    epoch: str = _EPOCH,
) -> RegistrationSnapshot:
    expected = expected if expected is not None else {"default": len(cells or [])}
    return RegistrationSnapshot(
        reporter_id=reporter_id,
        epoch=epoch,
        sequence=sequence,
        digest=(
            digest
            if digest is not None
            else compute_snapshot_digest(cells=cells or [], expected_num_cells_by_model=expected)
        ),
        expected_num_cells_by_model=expected,
        token=token,
        cells=cells,
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _provider(
    *, expected_num_reporters: int = 1, token: str | None = None, clock: _Clock | None = None
) -> RegistrationWorkerProvider:
    if clock is None:
        return RegistrationWorkerProvider(expected_num_reporters=expected_num_reporters, token=token)
    return RegistrationWorkerProvider(expected_num_reporters=expected_num_reporters, token=token, clock=clock)


async def _apply(provider: RegistrationWorkerProvider, snapshot: RegistrationSnapshot) -> RegistrationAck:
    ack = await provider.apply_snapshot(snapshot)
    await provider._wait_pending_dispatches()
    return ack


class TestApplySnapshotReplacesMembership:
    async def test_the_first_snapshot_announces_every_cell_it_carries(self):
        """A snapshot is the whole truth of one deployment, so its cells enter the run as they are."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)

        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=1))

        assert watcher.added == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]
        assert provider.cell_ids() == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]

    async def test_a_cell_missing_from_the_next_snapshot_is_dropped(self):
        """Replacement, not merging: what a reporter stops reporting has left its deployment."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=1))

        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        assert watcher.removed == [f"{_POOL_ID}-1"]
        assert provider.cell_ids() == [f"{_POOL_ID}-0"]

    async def test_a_changed_workers_hash_re_announces_the_cell(self):
        """A restarted engine is a different process behind the same cell id."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        await _apply(provider, _snapshot([_cell(0, workers_hash="hash-2")], sequence=2))

        assert watcher.added == [f"{_POOL_ID}-0", f"{_POOL_ID}-0"]
        assert watcher.calls[-1][1].workers_hash == "hash-2"

    async def test_a_changed_address_re_announces_the_cell(self):
        """A cell whose addresses moved under an unchanged hash would be dialled where it no longer is."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        await _apply(provider, _snapshot([_cell(0, port=9999)], sequence=2))

        assert (await provider.get_addrs(f"{_POOL_ID}-0-0"))["primary"].port == 9999
        assert watcher.added == [f"{_POOL_ID}-0", f"{_POOL_ID}-0"]

    async def test_an_unchanged_cell_is_not_announced_again(self):
        """Reconciling an unchanged cell would tear a healthy engine down and build it again."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        assert watcher.calls == [(f"{_POOL_ID}-0", watcher.calls[0][1])]

    async def test_cells_reported_before_the_watch_are_replayed_to_it(self):
        """The controller may start watching after a reporter already registered its cells."""
        provider = _provider()
        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        watcher = _Watcher()

        await provider.watch_cells(watcher)

        assert watcher.added == [f"{_POOL_ID}-0"]


class TestSnapshotSequence:
    async def test_a_late_snapshot_does_not_overwrite_a_newer_one(self):
        """A retried request that crosses the wan out of order would resurrect dead cells."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=1))
        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=1))

        assert provider.cell_ids() == [f"{_POOL_ID}-0"]

    async def test_the_ack_reports_the_sequence_this_run_holds(self):
        """The reporter reads the ack to tell whether its snapshot landed."""
        provider = _provider()

        ack = await _apply(provider, _snapshot([_cell(0)], sequence=7))

        assert ack.applied_sequence == 7

    async def test_a_restarted_reporter_counting_from_one_again_is_taken_in(self):
        """Its pod can be evicted at any time, and refusing it for good would freeze its datacenter."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=5))

        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=1, epoch="epoch-2"))

        assert provider.cell_ids() == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]

    async def test_a_fresh_provider_takes_in_a_snapshot_of_an_incarnation_it_never_saw(self):
        """A restarted controller holds no registry, and the reporter resends under the epoch it still has."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)

        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=9, expected={"default": 2}))

        assert provider.cell_ids() == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]
        assert provider.extra_expected_num_cells(model_id="default") == 2

    async def test_a_late_snapshot_of_a_new_incarnation_is_still_ordered_within_it(self):
        """The epoch resets the counter; inside one incarnation the wan may still reorder."""
        provider = _provider()
        await provider.watch_cells(_Watcher())
        await _apply(provider, _snapshot([_cell(0)], sequence=5))
        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=2, epoch="epoch-2"))

        await _apply(provider, _snapshot([_cell(0)], sequence=1, epoch="epoch-2"))

        assert provider.cell_ids() == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]

    async def test_a_snapshot_of_an_incarnation_that_was_replaced_is_ignored(self):
        """A snapshot of a dead pod that crossed the wan late would roll the live pod's membership back."""
        provider = _provider()
        await provider.watch_cells(_Watcher())
        await _apply(provider, _snapshot([_cell(0)], sequence=5))
        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=1, epoch="epoch-2"))

        await _apply(provider, _snapshot([_cell(0)], sequence=6))

        assert provider.cell_ids() == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]
        assert provider._reporters[_REPORTER].epoch == "epoch-2"

    async def test_a_retired_epoch_does_not_reset_the_sequence_of_the_live_one(self):
        """Taking the old epoch in would restart the counter and make every later snapshot look new."""
        provider = _provider()
        await provider.watch_cells(_Watcher())
        await _apply(provider, _snapshot([_cell(0)], sequence=5))
        await _apply(provider, _snapshot([_cell(0)], sequence=2, epoch="epoch-2"))

        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=9))

        assert provider._reporters[_REPORTER].sequence == 2
        assert provider.cell_ids() == [f"{_POOL_ID}-0"]

    async def test_a_heartbeat_of_a_new_incarnation_asks_for_the_whole_snapshot(self):
        """A restarted reporter may still hold the digest its predecessor had acknowledged."""
        provider = _provider()
        full = _snapshot([_cell(0)], sequence=1)
        await _apply(provider, full)

        ack = await _apply(provider, _snapshot(None, sequence=1, digest=full.digest, epoch="epoch-2"))

        assert ack.applied_digest is None


class TestDigestShortCircuit:
    async def test_a_heartbeat_for_the_digest_this_run_holds_changes_nothing(self):
        """An unchanged deployment sends no body, and the run must neither parse nor drop anything."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        full = _snapshot([_cell(0)], sequence=1)
        await _apply(provider, full)

        ack = await _apply(provider, _snapshot(None, sequence=2, digest=full.digest))

        assert provider.cell_ids() == [f"{_POOL_ID}-0"]
        assert watcher.calls == [(f"{_POOL_ID}-0", watcher.calls[0][1])]
        assert ack.applied_digest == full.digest

    async def test_a_heartbeat_for_an_unknown_digest_is_answered_with_the_known_one(self):
        """The reporter has to notice that its short circuit is stale and send the whole snapshot again."""
        provider = _provider()

        ack = await _apply(provider, _snapshot(None, sequence=1, digest="never-seen"))

        assert ack.applied_digest is None
        assert provider.cell_ids() == []

    async def test_a_refused_heartbeat_still_counts_as_a_sign_of_life(self):
        """Staleness is what says a datacenter went quiet, so a live reporter must never look stale."""
        provider = _provider()
        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        provider._reporters[_REPORTER].last_snapshot_at = -1000.0

        await _apply(provider, _snapshot(None, sequence=2, digest="stale"))

        assert provider.seconds_since_last_snapshot(_REPORTER) < 1000.0


class TestSnapshotRetryAfterAFailedReconcile:
    async def test_a_cell_whose_reconcile_failed_is_taken_in_by_the_next_snapshot(self):
        """A registry that never retries would leave a whole deployment out of the run for good."""
        provider = _provider()
        watcher = _Watcher(failing_cell_ids={f"{_POOL_ID}-0"})
        await provider.watch_cells(watcher)

        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        assert provider.cell_ids() == []

        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        assert provider.cell_ids() == [f"{_POOL_ID}-0"]

    async def test_a_failed_update_of_a_known_cell_keeps_addressing_it(self):
        """Dropping it outright turns every address of a live engine into a KeyError until the next snapshot."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        watcher._failing_cell_ids.add(f"{_POOL_ID}-0")

        await _apply(provider, _snapshot([_cell(0, workers_hash="hash-2")], sequence=2))

        assert (await provider.get_addrs(f"{_POOL_ID}-0-0"))["primary"].port == 8000
        assert provider._cells[f"{_POOL_ID}-0"].info.workers_hash == "hash-1"

    async def test_an_identical_snapshot_applied_twice_reconciles_once(self):
        """Retrying a snapshot must be free, or every retry would restart healthy engines."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)

        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        assert len(watcher.calls) == 1


class TestSnapshotValidation:
    async def test_a_snapshot_carrying_the_wrong_token_is_refused(self):
        """The registration endpoint is reachable across datacenters, so it authenticates its callers."""
        provider = _provider(token="secret")

        with pytest.raises(AssertionError, match="registration token"):
            await provider.apply_snapshot(_snapshot([_cell(0)], sequence=1, token="guess"))

    async def test_a_snapshot_whose_digest_does_not_match_its_cells_is_refused(self):
        """The digest is what the heartbeat short circuit trusts later, so it is checked on arrival."""
        provider = _provider()

        with pytest.raises(AssertionError, match="digest"):
            await provider.apply_snapshot(_snapshot([_cell(0)], sequence=1, digest="wrong"))

    async def test_prefill_and_decode_engines_of_another_deployment_are_refused(self):
        """Pairing a prefill engine with a decode engine across datacenters is not built yet."""
        provider = _provider()

        ack = await _apply(provider, _snapshot([_cell(0, worker_type="prefill"), _cell(1)], sequence=1))

        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]
        assert provider.cell_ids() == [f"{_POOL_ID}-1"]

    async def test_the_cells_a_refused_one_travelled_with_still_enter_the_run(self):
        """One mislabelled engine must not keep a whole datacenter out of the run."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)

        ack = await _apply(provider, _snapshot([_cell(0, worker_type="decode"), _cell(1), _cell(2)], sequence=1))

        assert watcher.added == [f"{_POOL_ID}-1", f"{_POOL_ID}-2"]
        assert ack.applied_digest is None
        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]

    async def test_two_reporters_claiming_one_cell_id_keep_the_first_one(self):
        """Colliding pool ids would let one deployment silently replace another's engines."""
        provider = _provider(expected_num_reporters=2)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        ack = await _apply(provider, _snapshot([_cell(0)], sequence=1, reporter_id="east"))

        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]
        assert provider._cells[f"{_POOL_ID}-0"].reporter_id == _REPORTER

    async def test_a_cell_id_that_does_not_name_its_own_pool_is_refused(self):
        """The controller addresses workers by parsing their cell id, so the two must agree."""
        provider = _provider()
        cell = _cell(0).model_copy(update=dict(cell_id="other-pool-0"))

        ack = await _apply(
            provider,
            _snapshot(
                [cell],
                sequence=1,
                digest=compute_snapshot_digest(cells=[cell], expected_num_cells_by_model={"default": 1}),
            ),
        )

        assert ack.excluded_cell_ids == ["other-pool-0"]
        assert provider.cell_ids() == []

    async def test_a_cell_id_carried_twice_by_one_snapshot_is_refused(self):
        """Keeping either entry would confirm a digest for a membership the run does not hold."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)

        ack = await _apply(provider, _snapshot([_cell(0), _cell(0, port=9999), _cell(1)], sequence=1))

        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]
        assert provider.cell_ids() == [f"{_POOL_ID}-1"]
        assert provider._reporters[_REPORTER].digest is None

    async def test_a_cell_id_that_does_not_parse_is_excluded_alone(self):
        """One malformed id from an older peer must not keep a whole datacenter out of the run."""
        provider = _provider()
        broken = _cell(0).model_copy(update=dict(cell_id="abc", pool_id="abc"))
        watcher = _Watcher()
        await provider.watch_cells(watcher)

        ack = await _apply(provider, _snapshot([broken, _cell(1)], sequence=1))

        assert ack.excluded_cell_ids == ["abc"]
        assert provider.cell_ids() == [f"{_POOL_ID}-1"]

    async def test_a_worker_name_that_does_not_parse_excludes_only_its_cell(self):
        """A worker this run cannot name is a worker it cannot address, and the rest is still servable."""
        provider = _provider()
        broken = _cell(0).model_copy(
            update=dict(workers=[RegisteredWorker(name="nope", addrs={}, gpu_ids=[0])]),
        )
        watcher = _Watcher()
        await provider.watch_cells(watcher)

        ack = await _apply(provider, _snapshot([broken, _cell(1)], sequence=1))

        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]
        assert provider.cell_ids() == [f"{_POOL_ID}-1"]


class TestExpectedNumCells:
    async def test_the_expected_counts_of_every_reporter_are_summed(self):
        """The startup barrier has to cover the engines of every deployment, not only the local ones."""
        provider = _provider(expected_num_reporters=2)
        await _apply(provider, _snapshot([_cell(0)], sequence=1, expected={"default": 4}))
        await _apply(provider, _snapshot([], sequence=1, expected={"default": 2}, reporter_id="east"))

        assert provider.extra_expected_num_cells(model_id="default") == 6

    async def test_a_reporter_that_has_not_registered_yet_holds_the_barrier(self):
        """Otherwise a run whose remote engines never arrive would start against the local ones alone."""
        provider = _provider(expected_num_reporters=2)
        await _apply(provider, _snapshot([_cell(0)], sequence=1, expected={"default": 1}))

        with pytest.raises(AssertionError, match="1/2"):
            provider.extra_expected_num_cells(model_id="default")

    async def test_a_reporter_whose_pods_are_not_up_yet_declares_them_anyway(self):
        """A cold datacenter reports no cell but knows how many it brings, and the barrier waits for them."""
        provider = _provider()

        await _apply(provider, _snapshot([], sequence=1, expected={"default": 4}))
        assert provider.extra_expected_num_cells(model_id="default") == 4
        assert provider.cell_ids() == []

        await _apply(provider, _snapshot([_cell(0)], sequence=2, expected={"default": 4}))

        assert provider.extra_expected_num_cells(model_id="default") == 4
        assert provider.cell_ids() == [f"{_POOL_ID}-0"]

    async def test_a_model_no_reporter_serves_expects_nothing_extra(self):
        """A run may have models that only its own deployment serves."""
        provider = _provider()
        await _apply(provider, _snapshot([_cell(0)], sequence=1, expected={"default": 1}))

        assert provider.extra_expected_num_cells(model_id="eval") == 0


class TestAddressingRegisteredCells:
    async def test_a_worker_is_addressed_by_the_addresses_its_reporter_gave(self):
        """The controller talks to a remote engine directly, so it needs a reachable address for it."""
        provider = _provider()
        await _apply(provider, _snapshot([_cell(1)], sequence=1))

        addrs = await provider.get_addrs(f"{_POOL_ID}-1-0")

        assert addrs["primary"] == HostAndPort(host="10.9.0.1", port=8001)

    async def test_worker_infos_carry_the_gpu_ids_of_the_reported_workers(self):
        """The dashboard reads worker infos to draw a cell, and a remote cell is drawn like a local one."""
        provider = _provider()
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        ((info,),) = provider.get_worker_infos(cell_ids=[f"{_POOL_ID}-0"])

        assert info.name == f"{_POOL_ID}-0-0"
        assert info.gpu_ids == [0]


class TestInvalidatingAProbedCell:
    async def test_an_invalidated_cell_leaves_the_run_at_once(self):
        """A cell the run probed dead keeps taking requests until the controller is told to remove it."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        provider.invalidate_cell(f"{_POOL_ID}-0")
        await provider._wait_pending_dispatches()

        assert watcher.removed == [f"{_POOL_ID}-0"]
        assert provider.cell_ids() == []

    async def test_an_invalidated_cell_is_reported_again_by_the_next_snapshot(self):
        """A cell the controller probed dead has to be able to come back."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        provider.invalidate_cell(f"{_POOL_ID}-0")
        await provider._wait_pending_dispatches()
        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        assert watcher.added == [f"{_POOL_ID}-0", f"{_POOL_ID}-0"]

    async def test_an_invalidation_during_a_slow_reconcile_is_not_covered_by_the_digest(self):
        """A digest claiming a cell the run no longer holds keeps the reporter on heartbeats for good."""
        provider = _provider()
        watcher = _BlockingWatcher(blocked_cell_ids={f"{_POOL_ID}-1"})
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        applying = asyncio.create_task(provider.apply_snapshot(_snapshot([_cell(0), _cell(1)], sequence=2)))
        await asyncio.wait_for(watcher.entered.wait(), timeout=5.0)
        provider.invalidate_cell(f"{_POOL_ID}-0")
        watcher.release.set()
        await asyncio.wait_for(applying, timeout=5.0)
        await provider._wait_pending_dispatches()

        assert provider.cell_ids() == [f"{_POOL_ID}-1"]
        assert provider._reporters[_REPORTER].digest is None


class TestConcurrentSnapshotsOfOneReporter:
    async def test_the_run_ends_at_the_highest_sequence_it_was_sent(self):
        """A slow weight update can hold a reconcile for minutes, so two snapshots of one reporter can overlap."""
        provider = _provider()
        watcher = _BlockingWatcher(blocked_cell_ids={f"{_POOL_ID}-0"})
        await provider.watch_cells(watcher)

        first = asyncio.create_task(provider.apply_snapshot(_snapshot([_cell(0)], sequence=1)))
        await asyncio.wait_for(watcher.entered.wait(), timeout=5.0)
        second = asyncio.create_task(provider.apply_snapshot(_snapshot([_cell(0), _cell(1)], sequence=2)))
        watcher.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=5.0)
        await provider._wait_pending_dispatches()

        assert provider._reporters[_REPORTER].sequence == 2
        assert provider.cell_ids() == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]

    async def test_a_late_snapshot_overlapping_a_newer_one_undoes_nothing(self):
        """Between reading the sequence and committing the membership, a newer snapshot must survive."""
        provider = _provider()
        watcher = _BlockingWatcher(blocked_cell_ids={f"{_POOL_ID}-1"})
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        newer = asyncio.create_task(provider.apply_snapshot(_snapshot([_cell(0), _cell(1)], sequence=3)))
        await asyncio.wait_for(watcher.entered.wait(), timeout=5.0)
        late = asyncio.create_task(provider.apply_snapshot(_snapshot([_cell(0)], sequence=2)))
        watcher.release.set()
        await asyncio.wait_for(asyncio.gather(newer, late), timeout=5.0)
        await provider._wait_pending_dispatches()

        assert provider._reporters[_REPORTER].sequence == 3
        assert provider.cell_ids() == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]


class TestWatchLifecycle:
    async def test_a_stopped_watch_stops_receiving_observations(self):
        """A disposed controller must not be handed cells of a deployment it no longer drives."""
        provider = _provider()
        watcher = _Watcher()
        stop_watch = await provider.watch_cells(watcher)

        await stop_watch()
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        assert watcher.calls == []

    async def test_a_watch_that_failed_its_replay_can_be_established_again(self):
        """A failed replay must not leave the provider believing it is already watched."""
        provider = _provider()
        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        with pytest.raises(RuntimeError):
            await provider.watch_cells(_Watcher(failing_cell_ids={f"{_POOL_ID}-0"}))

        watcher = _Watcher()
        await provider.watch_cells(watcher)

        assert watcher.added == [f"{_POOL_ID}-0"]

    async def test_stopping_the_watch_waits_for_the_reconciles_already_in_flight(self):
        """A dispatch still running against a controller that is tearing down races its own disposal."""
        provider = _provider()
        watcher = _BlockingWatcher(blocked_cell_ids={f"{_POOL_ID}-0"})
        stop_watch = await provider.watch_cells(watcher)
        await provider.apply_snapshot(_snapshot([_cell(0)], sequence=1))
        await asyncio.wait_for(watcher.entered.wait(), timeout=5.0)

        stopping = asyncio.create_task(stop_watch())
        watcher.release.set()
        await asyncio.wait_for(stopping, timeout=5.0)

        assert provider._dispatcher.done()


class TestDispatchOrder:
    async def test_the_reconciles_run_in_the_order_they_were_committed(self):
        """Replaying an add after the removal that followed it would resurrect an engine that is gone."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)

        await provider.apply_snapshot(_snapshot([_cell(0)], sequence=1))
        await provider.apply_snapshot(_snapshot([_cell(0), _cell(1)], sequence=2))
        provider.invalidate_cell(f"{_POOL_ID}-0")
        await provider._wait_pending_dispatches()

        assert [(cell_id, observed is None) for cell_id, observed in watcher.calls] == [
            (f"{_POOL_ID}-0", False),
            (f"{_POOL_ID}-1", False),
            (f"{_POOL_ID}-0", True),
        ]


class TestUndoingAFailedReconcile:
    async def test_a_failed_removal_keeps_addressing_the_cell_until_the_next_snapshot(self):
        """The run still holds that engine, so forgetting its addresses turns every call into a KeyError."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=1))
        watcher._failing_cell_ids.add(f"{_POOL_ID}-1")

        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        assert provider.cell_ids() == [f"{_POOL_ID}-0", f"{_POOL_ID}-1"]
        assert provider._reporters[_REPORTER].digest is None

    async def test_a_failed_invalidation_does_not_bring_the_cell_back(self):
        """The cell was probed dead, so undoing its removal would send requests to a dead engine again."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        watcher._failing_cell_ids.add(f"{_POOL_ID}-0")

        provider.invalidate_cell(f"{_POOL_ID}-0")
        await provider._wait_pending_dispatches()

        assert provider.cell_ids() == []

    async def test_a_failed_reconcile_withholds_the_digest_so_the_whole_snapshot_returns(self):
        """A digest confirming a cell the run does not hold keeps its reporter on heartbeats for good."""
        provider = _provider()
        await provider.watch_cells(_Watcher(failing_cell_ids={f"{_POOL_ID}-0"}))

        await _apply(provider, _snapshot([_cell(0)], sequence=1))

        assert provider._reporters[_REPORTER].digest is None


class TestEpochChurnDetection:
    async def test_an_epoch_changing_faster_than_a_deployment_is_rebuilt_is_an_error(self, caplog):
        """Two deployments sharing one instance name keep deleting each other's cells, and nothing else does that."""
        clock = _Clock()
        provider = _provider(clock=clock)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        clock.advance(EPOCH_CHURN_ERROR_SECONDS - 1.0)

        with caplog.at_level(logging.ERROR):
            await _apply(provider, _snapshot([_cell(0)], sequence=1, epoch="epoch-2"))

        assert "sharing the instance name" in caplog.text

    async def test_an_epoch_changing_after_a_pod_could_be_rebuilt_is_only_a_warning(self, caplog):
        """A container restarting in place is ordinary, and an error naming a name collision would be a lie."""
        clock = _Clock()
        provider = _provider(clock=clock)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        clock.advance(EPOCH_CHURN_ERROR_SECONDS + 1.0)

        with caplog.at_level(logging.WARNING):
            await _apply(provider, _snapshot([_cell(0)], sequence=1, epoch="epoch-2"))

        assert "sharing the instance name" not in caplog.text
        assert "new incarnation" in caplog.text


class _ControllerLikeWatcher:
    def __init__(self, *, failing_cell_ids: set[str] | None = None) -> None:
        self.server_cells: dict[str, str] = {}
        self._failing_cell_ids = set(failing_cell_ids or set())

    async def __call__(self, cell_id: str, observed: CellInfo | None) -> None:
        if observed is None:
            self.server_cells.pop(cell_id, None)
            return
        if self.server_cells.get(cell_id) == observed.workers_hash:
            return
        self.server_cells.pop(cell_id, None)
        if cell_id in self._failing_cell_ids:
            self._failing_cell_ids.discard(cell_id)
            raise RuntimeError(f"bringing {cell_id} up failed once")
        self.server_cells[cell_id] = observed.workers_hash


class TestUndoingAHalfAppliedReplacement:
    async def test_a_replacement_that_dropped_the_old_cell_is_dispatched_again(self):
        """The run let the old cell go before the new one failed, so both sides must be made to agree again."""
        provider = _provider()
        watcher = _ControllerLikeWatcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        watcher._failing_cell_ids.add(f"{_POOL_ID}-0")

        await _apply(provider, _snapshot([_cell(0, workers_hash="hash-2")], sequence=2))

        assert watcher.server_cells == {}

    async def test_a_snapshot_repeating_the_old_hash_still_brings_the_cell_back(self):
        """Without this the two sides diverge for good: the provider holds a cell the run threw away."""
        provider = _provider()
        watcher = _ControllerLikeWatcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        watcher._failing_cell_ids.add(f"{_POOL_ID}-0")
        await _apply(provider, _snapshot([_cell(0, workers_hash="hash-2")], sequence=2))

        await _apply(provider, _snapshot([_cell(0)], sequence=3))

        assert watcher.server_cells == {f"{_POOL_ID}-0": "hash-1"}

    async def test_a_cell_that_was_taken_in_is_not_dispatched_over_and_over(self):
        """Marking every cell dirty forever would restart healthy engines on every snapshot."""
        provider = _provider()
        watcher = _ControllerLikeWatcher(failing_cell_ids={f"{_POOL_ID}-0"})
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        await _apply(provider, _snapshot([_cell(0)], sequence=3))

        assert provider._to_reannounce == set()


class TestASnapshotThatLandedAfterItsSenderGaveUp:
    async def test_the_resend_of_a_snapshot_that_already_landed_reconciles_nothing_again(self):
        """The reporter cannot tell whether the slow one landed, so resending it must cost nothing."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)

        await _apply(provider, _snapshot([_cell(0)], sequence=1))
        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        assert len(watcher.calls) == 1
        assert provider.cell_ids() == [f"{_POOL_ID}-0"]

    async def test_the_slow_snapshot_arriving_after_its_resend_is_dropped(self):
        """It carries the same truth but an older sequence, and taking it in would undo the newer one."""
        provider = _provider()
        watcher = _Watcher()
        await provider.watch_cells(watcher)
        await _apply(provider, _snapshot([_cell(0)], sequence=2))

        ack = await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=1))

        assert ack.applied_sequence == 2
        assert provider.cell_ids() == [f"{_POOL_ID}-0"]
        assert len(watcher.calls) == 1


class TestTheCellContractOfTheRunItReportsInto:
    async def test_a_cell_the_run_itself_refuses_is_excluded_and_named(self, caplog):
        """What a cell's metadata must carry is the run's business, and the provider only carries the verdict."""
        provider = RegistrationWorkerProvider(
            expected_num_reporters=1,
            refuse_cell=lambda info: None if info.cell_id.endswith("-1") else "this run cannot build it",
        )
        watcher = _Watcher()
        await provider.watch_cells(watcher)

        with caplog.at_level(logging.ERROR):
            ack = await _apply(provider, _snapshot([_cell(0), _cell(1)], sequence=1))

        assert ack.excluded_cell_ids == [f"{_POOL_ID}-0"]
        assert ack.applied_digest is None
        assert watcher.added == [f"{_POOL_ID}-1"]
        assert "this run cannot build it" in caplog.text

    async def test_the_run_is_only_asked_about_cells_the_protocol_already_accepted(self):
        """A cell whose id or workers do not parse has no addressable worker to build metadata from."""
        asked: list[str] = []
        provider = RegistrationWorkerProvider(
            expected_num_reporters=1, refuse_cell=lambda info: asked.append(info.cell_id) or None
        )
        broken = _cell(0).model_copy(update=dict(workers=[]))
        await provider.watch_cells(_Watcher())

        await _apply(
            provider,
            _snapshot(
                [broken, _cell(1)],
                sequence=1,
                digest=compute_snapshot_digest(cells=[broken, _cell(1)], expected_num_cells_by_model={"default": 2}),
            ),
        )

        assert asked == [f"{_POOL_ID}-1"]
