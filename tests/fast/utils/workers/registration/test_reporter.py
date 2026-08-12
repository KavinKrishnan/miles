from __future__ import annotations

import asyncio
import logging
import os
import random
import time

import pytest

from miles.utils.workers.registration.models import RegistrationAck, RegistrationSnapshot
from miles.utils.workers.registration.reporter import (
    MAX_SEND_ATTEMPTS,
    RegistrationReporter,
    RegistrationReporterWorker,
)
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

_POOL_ID = "inference-engine-0-0"
_INSTANCE = "west"


class _FakeEngineProvider(BaseWorkerProvider):
    def __init__(self, *, cell_indices: list[int], worker_type: str = "regular") -> None:
        self.cell_indices = list(cell_indices)
        self.worker_type = worker_type
        self.reconcile: ReconcileFn | None = None
        self.stopped = False

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise NotImplementedError

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [
            [
                WorkerInfo(
                    name=f"{cell_id}-0",
                    generation=0,
                    self_addrs={"primary": HostAndPort(host="10.0.0.5", port=8000)},
                    gpu_ids=[0],
                    handle=None,
                    worker_class=None,
                )
            ]
            for cell_id in cell_ids
        ]

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        self.reconcile = reconcile
        for cell_index in self.cell_indices:
            await reconcile(f"{_POOL_ID}-{cell_index}", _cell_info(cell_index, worker_type=self.worker_type))

        async def _stop() -> None:
            self.stopped = True

        return _stop


class _FakeController:
    def __init__(self, *, acks: list[RegistrationAck] | None = None) -> None:
        self.snapshots: list[RegistrationSnapshot] = []
        self.ready_timeouts: list[float] = []
        self._acks = list(acks or [])

    async def wait_ready(self, *, timeout: float) -> None:
        self.ready_timeouts.append(timeout)

    async def apply_registration_snapshot(self, *, snapshot: RegistrationSnapshot) -> RegistrationAck:
        self.snapshots.append(snapshot)
        if self._acks:
            return self._acks.pop(0)
        return RegistrationAck(applied_sequence=snapshot.sequence, applied_digest=snapshot.digest)


class _DeadController(_FakeController):
    def __init__(self, *, sends_before_restart: int = 0) -> None:
        super().__init__()
        self._sends_before_restart = sends_before_restart

    async def apply_registration_snapshot(self, *, snapshot: RegistrationSnapshot) -> RegistrationAck:
        if len(self.snapshots) < self._sends_before_restart:
            return await super().apply_registration_snapshot(snapshot=snapshot)
        self.snapshots.append(snapshot)
        raise ServerRestartedError("rpc server restarted")


def _cell_info(cell_index: int, *, workers_hash: str = "hash-1", worker_type: str = "regular") -> CellInfo:
    return CellInfo(
        cell_id=f"{_POOL_ID}-{cell_index}",
        pool_id=_POOL_ID,
        alive=True,
        worker_names=[f"{_POOL_ID}-{cell_index}-0"],
        workers_hash=workers_hash,
        meta=dict(model_id="default", worker_type=worker_type),
    )


def _reporter(
    *,
    provider: _FakeEngineProvider,
    controller: _FakeController,
    restarted_controllers: list[_FakeController] | None = None,
    external_host_by_host: dict[str, str] | None = None,
    interval_seconds: float = 15.0,
    debounce_seconds: float = 0.01,
    jitter_ratio: float = 0.2,
    send_budget_seconds: float = 15.0,
    rng: random.Random | None = None,
) -> RegistrationReporter:
    incarnations = iter([controller, *(restarted_controllers or [])])
    return RegistrationReporter(
        reporter_id=_INSTANCE,
        create_controller=lambda: next(incarnations),
        engine_provider=provider,
        expected_num_cells_by_model={"default": 2},
        pool_id_prefix=_INSTANCE,
        external_host_by_host=external_host_by_host or {},
        token="secret",
        interval_seconds=interval_seconds,
        jitter_ratio=jitter_ratio,
        debounce_seconds=debounce_seconds,
        send_budget_seconds=send_budget_seconds,
        rng=rng,
    )


class TestColdStartGating:
    async def test_a_reporter_that_has_not_synced_refuses_to_send(self):
        """An empty cold-start snapshot would drop every engine of this deployment from the run."""
        reporter = _reporter(provider=_FakeEngineProvider(cell_indices=[0]), controller=_FakeController())

        with pytest.raises(AssertionError, match="first look"):
            await reporter.send_once()

    async def test_the_first_snapshot_after_the_sync_carries_the_observed_cells(self):
        """The first snapshot the run sees has to be the deployment as it actually is."""
        provider = _FakeEngineProvider(cell_indices=[0, 1])
        controller = _FakeController()
        reporter = _reporter(provider=provider, controller=controller)
        await _sync(reporter, provider)

        await reporter.send_once()

        [snapshot] = controller.snapshots
        assert [cell.cell_id for cell in snapshot.cells] == [f"{_INSTANCE}-{_POOL_ID}-0", f"{_INSTANCE}-{_POOL_ID}-1"]


class TestSnapshotContents:
    async def test_pool_ids_are_namespaced_by_the_instance_name(self):
        """Two datacenters run the same pools, and colliding cell ids would replace each other."""
        snapshot = await _synced_snapshot(_FakeEngineProvider(cell_indices=[0]))

        [cell] = snapshot.cells
        assert cell.pool_id == f"{_INSTANCE}-{_POOL_ID}"
        assert cell.workers[0].name == f"{_INSTANCE}-{_POOL_ID}-0-0"

    async def test_addresses_are_translated_to_ones_the_controller_can_reach(self):
        """The engines are seen under cluster-internal addresses that mean nothing in another datacenter."""
        snapshot = await _synced_snapshot(
            _FakeEngineProvider(cell_indices=[0]), external_host_by_host={"10.0.0.5": "engine-west.example"}
        )

        assert snapshot.cells[0].workers[0].addrs["primary"].host == "engine-west.example"

    async def test_an_address_without_a_translation_is_reported_as_it_is(self):
        """Only the addresses a deployment actually rewrites need an entry."""
        snapshot = await _synced_snapshot(_FakeEngineProvider(cell_indices=[0]))

        assert snapshot.cells[0].workers[0].addrs["primary"].host == "10.0.0.5"

    async def test_the_snapshot_carries_the_token_and_the_expected_cell_counts(self):
        """The run authenticates the reporter and learns how many cells to wait for from it."""
        snapshot = await _synced_snapshot(_FakeEngineProvider(cell_indices=[0]))

        assert snapshot.token == "secret"
        assert snapshot.expected_num_cells_by_model == {"default": 2}

    async def test_the_cell_meta_is_passed_through_untouched(self):
        """The controller groups cells by model through meta, so a reporter must not reshape it."""
        snapshot = await _synced_snapshot(_FakeEngineProvider(cell_indices=[0]))

        assert snapshot.cells[0].meta == dict(model_id="default", worker_type="regular")


class TestSequenceAndDigest:
    async def test_every_snapshot_carries_a_higher_sequence_number(self):
        """The run rejects late snapshots by their sequence, which only works if it grows."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController()
        reporter = _reporter(provider=provider, controller=controller)
        await _sync(reporter, provider)

        await reporter.send_once()
        await reporter.send_once()

        assert [snapshot.sequence for snapshot in controller.snapshots] == [1, 2]

    async def test_an_unchanged_deployment_sends_a_heartbeat_without_its_cells(self):
        """Parsing and validating ten thousand unchanged cells every period is pure waste."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController()
        reporter = _reporter(provider=provider, controller=controller)
        await _sync(reporter, provider)

        await reporter.send_once()
        await reporter.send_once()

        assert controller.snapshots[0].cells is not None
        assert controller.snapshots[1].cells is None
        assert controller.snapshots[1].digest == controller.snapshots[0].digest

    async def test_a_changed_deployment_sends_its_cells_again(self):
        """The short circuit must not hide a membership change."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController()
        reporter = _reporter(provider=provider, controller=controller)
        await _sync(reporter, provider)
        await reporter.send_once()

        await provider.reconcile(f"{_POOL_ID}-1", _cell_info(1))
        await reporter.send_once()

        assert [cell.cell_id for cell in controller.snapshots[1].cells] == [
            f"{_INSTANCE}-{_POOL_ID}-0",
            f"{_INSTANCE}-{_POOL_ID}-1",
        ]

    async def test_a_snapshot_the_run_did_not_keep_is_sent_in_full_again(self):
        """The ack is the only way to learn that the run applied only part of a snapshot."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController(acks=[RegistrationAck(applied_sequence=1, applied_digest=None)])
        reporter = _reporter(provider=provider, controller=controller)
        await _sync(reporter, provider)

        await reporter.send_once()
        await reporter.send_once()

        assert controller.snapshots[1].cells is not None

    async def test_a_refused_heartbeat_is_followed_by_the_whole_snapshot_at_once(self):
        """A cell the run probed dead would otherwise be missing for two whole periods rather than one."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController()
        reporter = _reporter(provider=provider, controller=controller)
        await _sync(reporter, provider)
        await reporter.send_once()
        controller._acks = [RegistrationAck(applied_sequence=2, applied_digest=None)]

        await reporter.send_once()

        assert [snapshot.cells is None for snapshot in controller.snapshots] == [False, True, False]

    async def test_every_snapshot_of_one_reporter_carries_the_same_epoch(self):
        """The run orders snapshots by sequence only inside one epoch, so a live reporter must keep its own."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController()
        reporter = _reporter(provider=provider, controller=controller)
        await _sync(reporter, provider)

        await reporter.send_once()
        await reporter.send_once()

        assert len({snapshot.epoch for snapshot in controller.snapshots}) == 1

    async def test_two_reporter_incarnations_draw_different_epochs(self):
        """A restarted pod counts from one again, and only a fresh epoch tells the run to accept that."""
        provider = _FakeEngineProvider(cell_indices=[0])
        epochs = set()
        for _incarnation in range(2):
            controller = _FakeController()
            reporter = _reporter(provider=provider, controller=controller)
            await _sync(reporter, provider)
            await reporter.send_once()
            epochs.add(controller.snapshots[0].epoch)

        assert len(epochs) == 2

    async def test_cells_the_run_refused_are_reported_loudly(self, caplog):
        """A datacenter whose engines are silently dropped looks healthy on every dashboard there is."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController(
            acks=[RegistrationAck(applied_sequence=1, applied_digest=None, excluded_cell_ids=["west-pool-0"])]
        )
        reporter = _reporter(provider=provider, controller=controller)
        await _sync(reporter, provider)

        with caplog.at_level(logging.ERROR):
            await reporter.send_once()

        assert "west-pool-0" in caplog.text


class TestARestartedController:
    async def test_a_restarted_controller_is_dialed_through_a_rebuilt_handle(self):
        """The old handle pins the dead incarnation's boot uuid, so keeping it freezes the reporter forever."""
        provider = _FakeEngineProvider(cell_indices=[0, 1])
        dead = _DeadController()
        fresh = _FakeController()
        reporter = _reporter(provider=provider, controller=dead, restarted_controllers=[fresh])
        await _sync(reporter, provider)

        await reporter.send_once()

        assert len(dead.snapshots) == 1
        [snapshot] = fresh.snapshots
        assert [cell.cell_id for cell in snapshot.cells] == [f"{_INSTANCE}-{_POOL_ID}-0", f"{_INSTANCE}-{_POOL_ID}-1"]

    async def test_the_reporter_keeps_its_epoch_and_counts_on_across_the_restart(self):
        """The reporter did not restart, so a new epoch would tell the run two deployments share one name."""
        provider = _FakeEngineProvider(cell_indices=[0])
        dead = _DeadController()
        fresh = _FakeController()
        reporter = _reporter(provider=provider, controller=dead, restarted_controllers=[fresh])
        await _sync(reporter, provider)

        await reporter.send_once()

        assert fresh.snapshots[0].epoch == dead.snapshots[0].epoch
        assert fresh.snapshots[0].sequence > dead.snapshots[0].sequence

    async def test_a_heartbeat_that_hits_a_restarted_controller_is_followed_by_the_whole_snapshot(self):
        """A fresh controller holds none of this deployment's cells, so a digest it never saw restores nothing."""
        provider = _FakeEngineProvider(cell_indices=[0])
        dead = _DeadController(sends_before_restart=1)
        fresh = _FakeController()
        reporter = _reporter(provider=provider, controller=dead, restarted_controllers=[fresh])
        await _sync(reporter, provider)
        await reporter.send_once()

        await reporter.send_once()

        assert dead.snapshots[1].cells is None
        assert [cell.cell_id for cell in fresh.snapshots[0].cells] == [f"{_INSTANCE}-{_POOL_ID}-0"]

    async def test_a_controller_that_keeps_restarting_is_left_to_the_next_period(self):
        """Rebuilding without a bound would spin the reporter's thread against a crash looping controller."""
        provider = _FakeEngineProvider(cell_indices=[0])
        dead = [_DeadController() for _ in range(MAX_SEND_ATTEMPTS + 1)]
        reporter = _reporter(provider=provider, controller=dead[0], restarted_controllers=dead[1:])
        await _sync(reporter, provider)

        await reporter.send_once()

        assert [len(controller.snapshots) for controller in dead] == [1] * MAX_SEND_ATTEMPTS + [0]


class TestASendThatNeverAnswers:
    async def test_a_send_that_hangs_is_given_up_on_within_the_period(self):
        """One half open connection would otherwise freeze this datacenter's membership for a whole rpc timeout."""

        class _HangingController(_FakeController):
            async def apply_registration_snapshot(self, *, snapshot: RegistrationSnapshot) -> RegistrationAck:
                self.snapshots.append(snapshot)
                await asyncio.Event().wait()

        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _HangingController()
        reporter = _reporter(provider=provider, controller=controller, send_budget_seconds=0.3)
        await _sync(reporter, provider)

        await asyncio.wait_for(reporter.send_once(), timeout=5.0)

        assert len(controller.snapshots) == MAX_SEND_ATTEMPTS

    async def test_the_snapshot_after_a_timeout_carries_the_cells_again(self):
        """The run may or may not have applied the slow one, and a whole replacement is safe to resend."""

        class _SlowThenReadyController(_FakeController):
            async def apply_registration_snapshot(self, *, snapshot: RegistrationSnapshot) -> RegistrationAck:
                if not self.snapshots:
                    self.snapshots.append(snapshot)
                    await asyncio.Event().wait()
                return await super().apply_registration_snapshot(snapshot=snapshot)

        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _SlowThenReadyController()
        reporter = _reporter(provider=provider, controller=controller, send_budget_seconds=0.3)
        await _sync(reporter, provider)

        await asyncio.wait_for(reporter.send_once(), timeout=5.0)

        assert [snapshot.cells is None for snapshot in controller.snapshots] == [False, False]

    async def test_a_period_that_landed_nothing_says_so(self, caplog):
        """A datacenter whose snapshots all fail looks exactly like one that has nothing to report."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _DeadController()
        reporter = _reporter(
            provider=provider,
            controller=controller,
            restarted_controllers=[_DeadController() for _ in range(MAX_SEND_ATTEMPTS)],
        )
        await _sync(reporter, provider)

        with caplog.at_level(logging.ERROR):
            await reporter.send_once()

        assert f"None of the {MAX_SEND_ATTEMPTS} snapshots" in caplog.text


class TestPdDisaggregationIsRefusedAtTheSource:
    async def test_a_deployment_of_prefill_engines_refuses_to_report_at_all(self):
        """Failing at startup beats a warning every fifteen seconds that nobody reads."""
        provider = _FakeEngineProvider(cell_indices=[0], worker_type="prefill")
        reporter = _reporter(provider=provider, controller=_FakeController())

        with pytest.raises(AssertionError, match="pairing a prefill"):
            await reporter.run()

    async def test_a_prefill_engine_appearing_later_stops_the_snapshot(self):
        """A deployment can grow a prefill group after it started, and it must not travel in a snapshot."""
        provider = _FakeEngineProvider(cell_indices=[0])
        reporter = _reporter(provider=provider, controller=_FakeController())
        await _sync(reporter, provider)

        await provider.reconcile(f"{_POOL_ID}-1", _cell_info(1, worker_type="decode"))

        with pytest.raises(AssertionError, match="pairing a prefill"):
            await reporter.send_once()

    async def test_a_prefill_engine_appearing_later_takes_the_run_loop_down_with_it(self):
        """Such a deployment is as broken as one that started that way, and a warning loop only hides it."""
        provider = _FakeEngineProvider(cell_indices=[0])
        reporter = _reporter(provider=provider, controller=_FakeController(), interval_seconds=0.01)

        task = asyncio.create_task(reporter.run())
        await asyncio.sleep(0.05)
        await provider.reconcile(f"{_POOL_ID}-1", _cell_info(1, worker_type="decode"))

        with pytest.raises(AssertionError, match="pairing a prefill"):
            await asyncio.wait_for(task, timeout=5.0)
        assert provider.stopped


class TestSendSchedule:
    async def test_a_membership_change_is_reported_without_waiting_for_the_period(self):
        """A whole period of latency on every engine coming and going is the point of the watch."""
        provider = _FakeEngineProvider(cell_indices=[0])
        reporter = _reporter(provider=provider, controller=_FakeController(), interval_seconds=1000.0)
        await _sync(reporter, provider)

        await provider.reconcile(f"{_POOL_ID}-1", _cell_info(1))
        await asyncio.wait_for(reporter._wait_next_send(), timeout=5.0)

    async def test_a_burst_of_changes_is_debounced_into_one_send(self):
        """A rolling pool would otherwise send one whole snapshot per engine it replaces."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController()
        reporter = _reporter(provider=provider, controller=controller, interval_seconds=1000.0, debounce_seconds=0.05)
        await _sync(reporter, provider)

        async def _churn() -> None:
            for cell_index in range(1, 4):
                await asyncio.sleep(0.005)
                await provider.reconcile(f"{_POOL_ID}-{cell_index}", _cell_info(cell_index))

        churn = asyncio.create_task(_churn())
        await asyncio.wait_for(reporter._wait_next_send(), timeout=5.0)
        await churn
        await reporter.send_once()

        assert len(controller.snapshots) == 1
        assert len(controller.snapshots[0].cells) == 4

    async def test_the_period_is_jittered_around_the_configured_interval(self):
        """Reporters that all wake up together would arrive as one storm every period."""
        reporter = _reporter(
            provider=_FakeEngineProvider(cell_indices=[]),
            controller=_FakeController(),
            interval_seconds=15.0,
            rng=random.Random(0),
        )

        intervals = {reporter._compute_next_interval_seconds() for _ in range(20)}

        assert len(intervals) == 20
        assert all(12.0 <= interval <= 18.0 for interval in intervals)


class TestRun:
    async def test_the_run_loop_waits_for_the_controller_and_stops_when_asked(self):
        """A reporter that keeps its thread alive after dispose would outlive its deployment."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController()
        reporter = _reporter(provider=provider, controller=controller, interval_seconds=0.01)

        task = asyncio.create_task(reporter.run())
        while not controller.snapshots:
            await asyncio.sleep(0.01)
        reporter.request_stop()
        await asyncio.wait_for(task, timeout=5.0)

        assert controller.ready_timeouts
        assert provider.stopped

    async def test_a_failing_send_is_retried_on_the_next_period(self):
        """The wan drops requests, and a reporter that gave up would strand its whole deployment."""
        provider = _FakeEngineProvider(cell_indices=[0])

        class _FlakyController(_FakeController):
            async def apply_registration_snapshot(self, *, snapshot: RegistrationSnapshot) -> RegistrationAck:
                if not self.snapshots:
                    self.snapshots.append(snapshot)
                    raise RuntimeError("connection reset")
                return await super().apply_registration_snapshot(snapshot=snapshot)

        controller = _FlakyController()
        reporter = _reporter(provider=provider, controller=controller, interval_seconds=0.01)

        task = asyncio.create_task(reporter.run())
        while len(controller.snapshots) < 2:
            await asyncio.sleep(0.01)
        reporter.request_stop()
        await asyncio.wait_for(task, timeout=5.0)


async def _sync(reporter: RegistrationReporter, provider: _FakeEngineProvider) -> None:
    await provider.watch_cells(reporter._observe)
    reporter._has_synced = True


async def _synced_snapshot(
    provider: _FakeEngineProvider, *, external_host_by_host: dict[str, str] | None = None
) -> RegistrationSnapshot:
    controller = _FakeController()
    reporter = _reporter(provider=provider, controller=controller, external_host_by_host=external_host_by_host)
    await _sync(reporter, provider)
    await reporter.send_once()
    return controller.snapshots[0]


class TestRegistrationReporterWorker:
    def test_the_worker_runs_the_reporter_on_a_thread_and_stops_it_on_dispose(self):
        """Nothing calls a reporter worker, so it has to drive itself and still stop when its release goes down."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController()
        reporter = _reporter(provider=provider, controller=controller, interval_seconds=0.01)

        worker = RegistrationReporterWorker(reporter=reporter)
        deadline = time.monotonic() + 5.0
        while not controller.snapshots and time.monotonic() < deadline:
            time.sleep(0.01)
        asyncio.run(worker.dispose())
        worker._thread.join(timeout=5.0)

        assert controller.snapshots
        assert not worker._thread.is_alive()


class TestTheSendBudgetOfOnePeriod:
    async def test_every_retry_of_one_period_fits_inside_that_period(self):
        """Retrying past the interval makes this reporter look stale to the very run it registers into."""

        class _HangingController(_FakeController):
            async def apply_registration_snapshot(self, *, snapshot: RegistrationSnapshot) -> RegistrationAck:
                self.snapshots.append(snapshot)
                await asyncio.Event().wait()

        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _HangingController()
        reporter = _reporter(provider=provider, controller=controller, send_budget_seconds=0.3)
        await _sync(reporter, provider)

        started_at = time.monotonic()
        await asyncio.wait_for(reporter.send_once(), timeout=5.0)

        assert time.monotonic() - started_at < 0.3 * 2
        assert len(controller.snapshots) == MAX_SEND_ATTEMPTS

    async def test_a_reporter_asked_to_stop_sends_nothing_more(self):
        """A deployment being torn down would otherwise keep announcing cells that are already going away."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController()
        reporter = _reporter(provider=provider, controller=controller)
        await _sync(reporter, provider)
        reporter.request_stop()

        await reporter.send_once()

        assert controller.snapshots == []


class TestTheReporterWorkerFailsFast:
    def test_a_reporter_that_stopped_on_its_own_takes_the_deployment_down(self, monkeypatch):
        """Nothing else registers these engines, so a pod that kept running would look healthy while the run waits."""
        exit_codes: list[int] = []
        monkeypatch.setattr(os, "_exit", exit_codes.append)

        worker = RegistrationReporterWorker(reporter=_BrokenReporter(stop_requested=False))
        worker._thread.join(timeout=5.0)

        assert exit_codes == [1]

    def test_a_reporter_that_was_asked_to_stop_leaves_the_exit_code_alone(self, monkeypatch):
        """A clean shutdown must not look like a crash to kubernetes."""
        exit_codes: list[int] = []
        monkeypatch.setattr(os, "_exit", exit_codes.append)

        worker = RegistrationReporterWorker(reporter=_BrokenReporter(stop_requested=True))
        worker._thread.join(timeout=5.0)

        assert exit_codes == []

    def test_disposing_the_worker_waits_for_its_thread(self):
        """A reporter still sending after dispose announces cells of a deployment that is already being removed."""
        provider = _FakeEngineProvider(cell_indices=[0])
        controller = _FakeController()
        reporter = _reporter(provider=provider, controller=controller, interval_seconds=0.01)

        worker = RegistrationReporterWorker(reporter=reporter)
        deadline = time.monotonic() + 5.0
        while not controller.snapshots and time.monotonic() < deadline:
            time.sleep(0.01)
        asyncio.run(worker.dispose())

        assert not worker._thread.is_alive()


class _BrokenReporter:
    def __init__(self, *, stop_requested: bool) -> None:
        self.stop_requested = stop_requested

    async def run(self) -> None:
        raise RuntimeError("injected reporter failure")
