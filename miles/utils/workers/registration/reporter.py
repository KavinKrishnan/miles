from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
import uuid
from collections.abc import Callable

from miles.utils.workers.naming import compute_cell_id, compute_worker_name, parse_cell_id, parse_worker_name
from miles.utils.workers.registration.models import (
    CONTROLLER_READY_TIMEOUT_SECONDS,
    SNAPSHOT_DEBOUNCE_SECONDS,
    SNAPSHOT_INTERVAL_SECONDS,
    SNAPSHOT_JITTER_RATIO,
    SNAPSHOT_SEND_BUDGET_SECONDS,
    SUPPORTED_WORKER_TYPE,
    RegisteredCell,
    RegisteredWorker,
    RegistrationAck,
    RegistrationSnapshot,
    compute_snapshot_digest,
)
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

logger = logging.getLogger(__name__)

MAX_SEND_ATTEMPTS = 3
REPORTER_STOP_TIMEOUT_SECONDS = 30.0


class RegistrationReporter:
    def __init__(
        self,
        *,
        reporter_id: str,
        create_controller: Callable[[], BaseWorkerHandle],
        engine_provider: BaseWorkerProvider,
        expected_num_cells_by_model: dict[str, int],
        pool_id_prefix: str,
        external_host_by_host: dict[str, str],
        token: str | None = None,
        interval_seconds: float = SNAPSHOT_INTERVAL_SECONDS,
        jitter_ratio: float = SNAPSHOT_JITTER_RATIO,
        debounce_seconds: float = SNAPSHOT_DEBOUNCE_SECONDS,
        send_budget_seconds: float = SNAPSHOT_SEND_BUDGET_SECONDS,
        epoch: str | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._reporter_id = reporter_id
        self._create_controller = create_controller
        self._controller = create_controller()
        self._engine_provider = engine_provider
        self._expected_num_cells_by_model = dict(expected_num_cells_by_model)
        self._pool_id_prefix = pool_id_prefix
        self._external_host_by_host = dict(external_host_by_host)
        self._token = token
        self._interval_seconds = interval_seconds
        self._jitter_ratio = jitter_ratio
        self._debounce_seconds = debounce_seconds
        self._send_budget_seconds = send_budget_seconds
        self._rng = rng if rng is not None else random.Random()
        self._observed: dict[str, CellInfo] = {}
        self._changed = asyncio.Event()
        self._stopped = threading.Event()
        self._has_synced = False
        self._epoch = epoch if epoch is not None else uuid.uuid4().hex
        self._sequence = 0
        self._acknowledged_digest: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._controller.wait_ready(timeout=CONTROLLER_READY_TIMEOUT_SECONDS)
        stop_watch = await self._engine_provider.watch_cells(self._observe)
        self._has_synced = True

        try:
            self._assert_every_cell_is_regular()
            logger.info(
                f"Reporter {self._reporter_id} observes {len(self._observed)} cells of its own deployment and "
                f"reports them every {self._interval_seconds}s"
            )
            while not self._stopped.is_set():
                await self._wait_next_send()
                if self._stopped.is_set():
                    return
                self._assert_every_cell_is_regular()
                try:
                    await self.send_once()
                except Exception:
                    logger.warning(f"Reporting the cells of {self._reporter_id} failed", exc_info=True)
        finally:
            await stop_watch()

    def request_stop(self) -> None:
        self._stopped.set()
        self._wake_up()

    @property
    def stop_requested(self) -> bool:
        return self._stopped.is_set()

    async def send_once(self) -> None:
        assert self._has_synced, (
            f"reporter {self._reporter_id} has not finished its first look at its own deployment, and an empty "
            f"snapshot would drop every cell of this deployment from the run"
        )

        deadline = time.monotonic() + self._send_budget_seconds
        attempt_timeout_seconds = self._send_budget_seconds / MAX_SEND_ATTEMPTS
        for _attempt in range(MAX_SEND_ATTEMPTS):
            if self._stopped.is_set() or (remaining_seconds := deadline - time.monotonic()) <= 0.0:
                break
            snapshot = self._compute_snapshot()
            try:
                ack = await asyncio.wait_for(
                    self._controller.apply_registration_snapshot(snapshot=snapshot),
                    timeout=min(attempt_timeout_seconds, remaining_seconds),
                )
            except ServerRestartedError:
                self._rebuild_controller()
                continue
            except TimeoutError:
                logger.warning(
                    f"Registering the cells of {self._reporter_id} was not answered within "
                    f"{attempt_timeout_seconds:.1f}s, so this attempt is dropped rather than holding this "
                    f"deployment's whole membership on one half open connection; a snapshot is a whole "
                    f"replacement, so sending it again costs nothing even if the slow one did land"
                )
                continue
            self._report_refused_cells(ack)
            self._acknowledged_digest = ack.applied_digest
            if snapshot.cells is not None or ack.applied_digest == snapshot.digest:
                return

        logger.error(
            f"None of the {MAX_SEND_ATTEMPTS} snapshots reporter {self._reporter_id} sent this period were applied, "
            f"so the run keeps the cells of this deployment as it last saw them until a later period lands"
        )

    def _rebuild_controller(self) -> None:
        logger.warning(
            f"The inference controller reporter {self._reporter_id} reports into restarted, so this reporter takes "
            f"a handle on the new incarnation, pins its boot uuid instead of the dead one, and sends the whole "
            f"snapshot again because the new controller holds none of this deployment's cells"
        )
        self._controller = self._create_controller()
        self._acknowledged_digest = None

    def _report_refused_cells(self, ack: RegistrationAck) -> None:
        if not ack.excluded_cell_ids:
            return
        logger.error(
            f"The run refused {len(ack.excluded_cell_ids)} of the cells reporter {self._reporter_id} reported "
            f"({ack.excluded_cell_ids}); they serve no request of this run, and the run's own log says why"
        )

    def _assert_every_cell_is_regular(self) -> None:
        offenders = {
            cell_id: worker_type
            for cell_id, info in sorted(self._observed.items())
            if (worker_type := info.meta.get("worker_type")) != SUPPORTED_WORKER_TYPE
        }
        assert not offenders, (
            f"reporter {self._reporter_id} observes {offenders} in its own deployment, and pairing a prefill "
            f"engine of one deployment with a decode engine of another needs a router-side pairing policy that "
            f"does not exist yet, so this deployment cannot register its engines into another run"
        )

    async def _observe(self, cell_id: str, observed: CellInfo | None) -> None:
        if observed is None:
            self._observed.pop(cell_id, None)
        else:
            self._observed[cell_id] = observed
        self._changed.set()

    def _compute_snapshot(self) -> RegistrationSnapshot:
        self._assert_every_cell_is_regular()
        cells = self._compute_cells()
        digest = compute_snapshot_digest(cells=cells, expected_num_cells_by_model=self._expected_num_cells_by_model)
        self._sequence += 1
        return RegistrationSnapshot(
            reporter_id=self._reporter_id,
            epoch=self._epoch,
            sequence=self._sequence,
            digest=digest,
            expected_num_cells_by_model=self._expected_num_cells_by_model,
            token=self._token,
            cells=None if digest == self._acknowledged_digest else cells,
        )

    def _compute_cells(self) -> list[RegisteredCell]:
        observed = sorted(self._observed.items())
        infos_per_cell = self._engine_provider.get_worker_infos(cell_ids=[cell_id for cell_id, _ in observed])
        return [
            self._compute_cell(info, worker_infos=worker_infos)
            for (_cell_id, info), worker_infos in zip(observed, infos_per_cell, strict=True)
        ]

    def _compute_cell(self, info: CellInfo, *, worker_infos: list[WorkerInfo]) -> RegisteredCell:
        pool_id = f"{self._pool_id_prefix}-{info.pool_id}"
        cell_index = parse_cell_id(info.cell_id).cell_index
        return RegisteredCell(
            cell_id=compute_cell_id(pool_id=pool_id, cell_index=cell_index),
            pool_id=pool_id,
            workers_hash=info.workers_hash,
            workers=[
                RegisteredWorker(
                    name=compute_worker_name(
                        pool_id=pool_id,
                        cell_index=cell_index,
                        worker_in_cell_index=parse_worker_name(worker_info.name)[2],
                    ),
                    addrs=self._compute_external_addrs(worker_info.self_addrs),
                    gpu_ids=list(worker_info.gpu_ids),
                )
                for worker_info in sorted(worker_infos, key=lambda one: parse_worker_name(one.name)[2])
            ],
            meta=dict(info.meta),
        )

    def _compute_external_addrs(self, addrs: NamedHostAndPorts) -> NamedHostAndPorts:
        return {
            name: HostAndPort(host=self._external_host_by_host.get(addr.host, addr.host), port=addr.port)
            for name, addr in addrs.items()
        }

    async def _wait_next_send(self) -> None:
        try:
            await asyncio.wait_for(self._changed.wait(), timeout=self._compute_next_interval_seconds())
        except TimeoutError:
            return
        self._changed.clear()
        if self._stopped.is_set():
            return
        await asyncio.sleep(self._debounce_seconds)
        self._changed.clear()

    def _compute_next_interval_seconds(self) -> float:
        return self._interval_seconds * (1.0 + self._rng.uniform(-self._jitter_ratio, self._jitter_ratio))

    def _wake_up(self) -> None:
        if (loop := self._loop) is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._changed.set)


class RegistrationReporterWorker:
    def __init__(self, *, reporter: RegistrationReporter) -> None:
        self._reporter = reporter
        self._thread = threading.Thread(target=self._run, name="registration-reporter", daemon=True)
        self._thread.start()

    async def dispose(self) -> None:
        self._reporter.request_stop()
        await asyncio.to_thread(self._thread.join, REPORTER_STOP_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            logger.warning(
                f"The registration reporter of this deployment did not stop within {REPORTER_STOP_TIMEOUT_SECONDS}s, "
                f"so it may announce this deployment's cells once more before the process exits"
            )

    def _run(self) -> None:
        try:
            asyncio.run(self._reporter.run())
        except BaseException:
            if self._reporter.stop_requested:
                logger.warning(
                    "The registration reporter of this deployment stopped while it was being disposed, so this "
                    "deployment is shutting down anyway and its exit code stays its own",
                    exc_info=True,
                )
                return
            logger.error(
                "The registration reporter of this deployment stopped, so the deployment exits with it: nothing "
                "else here registers these engines into the run, and a pod that keeps running would look healthy "
                "while the run waits for cells that are never announced",
                exc_info=True,
            )
            os._exit(1)
