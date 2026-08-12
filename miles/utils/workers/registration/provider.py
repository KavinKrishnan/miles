from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from miles.utils.context_lock import compute_context_without_lock_grant
from miles.utils.workers.launch_gate import GATE_TIMEOUT_META_KEY
from miles.utils.workers.naming import ParsedCellId, compute_cell_id, parse_cell_id, parse_worker_name
from miles.utils.workers.registration.models import (
    REGISTERED_LAUNCH_GATE_TIMEOUT_SECONDS,
    SNAPSHOT_INTERVAL_SECONDS,
    SUPPORTED_WORKER_TYPE,
    RegisteredCell,
    RegistrationAck,
    RegistrationSnapshot,
    compute_snapshot_digest,
)
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import (
    BaseWorkerProvider,
    CellInfo,
    ReconcileFn,
    StopWatchFn,
    cell_id_of_worker,
)
from miles.utils.workers.worker_spec import NamedHostAndPorts

logger = logging.getLogger(__name__)

EPOCH_CHURN_ERROR_SECONDS = 4 * SNAPSHOT_INTERVAL_SECONDS


@dataclass
class _ReporterState:
    epoch: str = ""
    sequence: int = -1
    digest: str | None = None
    expected_num_cells_by_model: dict[str, int] = field(default_factory=dict)
    last_snapshot_at: float = 0.0
    retired_epochs: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _OwnedCell:
    reporter_id: str
    info: CellInfo
    addrs_by_worker: dict[str, NamedHostAndPorts]
    gpu_ids_by_worker: dict[str, list[int]]


@dataclass(frozen=True)
class _ParsedSnapshot:
    cells: dict[str, _OwnedCell]
    excluded_cell_ids: list[str]


@dataclass(frozen=True)
class _PendingDispatch:
    reporter_id: str
    cell_id: str
    observed: CellInfo | None
    applied: _OwnedCell | None
    previous: _OwnedCell | None


class RegistrationWorkerProvider(BaseWorkerProvider):
    def __init__(
        self,
        *,
        expected_num_reporters: int,
        token: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        refuse_cell: Callable[[CellInfo], str | None] = lambda _info: None,
    ) -> None:
        self._expected_num_reporters = expected_num_reporters
        self._token = token
        self._clock = clock
        self._refuse_cell = refuse_cell
        self._reporters: dict[str, _ReporterState] = {}
        self._cells: dict[str, _OwnedCell] = {}
        self._to_reannounce: set[str] = set()
        self._pending: deque[_PendingDispatch] = deque()
        self._dispatcher: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._reconcile_lock = asyncio.Lock()
        self._reconcile: ReconcileFn | None = None

    def reporter_ids(self) -> list[str]:
        return sorted(self._reporters)

    def cell_ids(self) -> list[str]:
        return sorted(self._cells)

    def seconds_since_last_snapshot(self, reporter_id: str) -> float:
        return self._clock() - self._reporters[reporter_id].last_snapshot_at

    async def apply_snapshot(self, snapshot: RegistrationSnapshot) -> RegistrationAck:
        self._assert_authentic(snapshot)
        parsed = None if snapshot.cells is None else self._parse_cells(snapshot)

        async with self._lock:
            ack = self._commit_snapshot(snapshot, parsed=parsed)

        self._ensure_dispatcher()
        return ack

    def extra_expected_num_cells(self, *, model_id: str) -> int:
        assert len(self._reporters) >= self._expected_num_reporters, (
            f"{len(self._reporters)}/{self._expected_num_reporters} engine deployments have reported themselves "
            f"({sorted(self._reporters)}), so the cells of the missing ones are not known yet"
        )
        return sum(state.expected_num_cells_by_model.get(model_id, 0) for state in self._reporters.values())

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        cell = self._cells[cell_id_of_worker(worker_name)]
        assert worker_name in cell.addrs_by_worker, (
            f"{worker_name} is not one of the workers {sorted(cell.addrs_by_worker)} that reporter "
            f"{cell.reporter_id} reported for cell {cell.info.cell_id}"
        )
        return cell.addrs_by_worker[worker_name]

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [self._worker_infos_of_cell(cell_id) for cell_id in cell_ids]

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        async with self._reconcile_lock:
            assert self._reconcile is None, "a registration provider reports to exactly one watcher"
            for cell_id, cell in sorted(self._cells.items()):
                await reconcile(cell_id, cell.info)
            self._reconcile = reconcile

        async def _stop() -> None:
            async with self._reconcile_lock:
                self._reconcile = None
            await self._wait_pending_dispatches()

        return _stop

    def invalidate_cell(self, cell_id: str) -> None:
        if (cell := self._cells.pop(cell_id, None)) is None:
            return
        if (state := self._reporters.get(cell.reporter_id)) is not None:
            state.digest = None
        self._pending.append(
            _PendingDispatch(reporter_id=cell.reporter_id, cell_id=cell_id, observed=None, applied=None, previous=None)
        )
        self._ensure_dispatcher()

    def _assert_authentic(self, snapshot: RegistrationSnapshot) -> None:
        assert snapshot.token == self._token, (
            f"reporter {snapshot.reporter_id} presented a registration token this run does not accept; both sides "
            f"read it from --registration-token"
        )

    def _parse_cells(self, snapshot: RegistrationSnapshot) -> _ParsedSnapshot:
        assert (
            compute_snapshot_digest(
                cells=snapshot.cells, expected_num_cells_by_model=snapshot.expected_num_cells_by_model
            )
            == snapshot.digest
        ), f"the snapshot of reporter {snapshot.reporter_id} does not hash to the digest it carries"

        occurrences = Counter(cell.cell_id for cell in snapshot.cells)
        parsed: dict[str, _OwnedCell] = {}
        excluded: list[str] = []
        for cell in snapshot.cells:
            info = _compute_cell_info(cell)
            reason = self._compute_cell_refusal_reason(cell, info=info, occurrences=occurrences)
            if reason is not None:
                logger.error(
                    f"Excluding cell {cell.cell_id} of reporter {snapshot.reporter_id} from this run: {reason}. "
                    f"The rest of the snapshot is taken in, and the whole snapshot is asked for again"
                )
                excluded.append(cell.cell_id)
                continue
            parsed[cell.cell_id] = _OwnedCell(
                reporter_id=snapshot.reporter_id,
                info=info,
                addrs_by_worker={worker.name: worker.addrs for worker in cell.workers},
                gpu_ids_by_worker={worker.name: list(worker.gpu_ids) for worker in cell.workers},
            )
        return _ParsedSnapshot(cells=parsed, excluded_cell_ids=sorted(set(excluded)))

    def _compute_cell_refusal_reason(
        self, cell: RegisteredCell, *, info: CellInfo, occurrences: Counter[str]
    ) -> str | None:
        if occurrences[cell.cell_id] > 1:
            return (
                f"one snapshot carries it {occurrences[cell.cell_id]} times, and a cell id names exactly one cell, "
                f"so keeping either entry would confirm a digest for a membership this run does not hold"
            )
        if (reason := _compute_refusal_reason(cell)) is not None:
            return reason
        return self._refuse_cell(info)

    def _commit_snapshot(self, snapshot: RegistrationSnapshot, *, parsed: _ParsedSnapshot | None) -> RegistrationAck:
        state = self._reporters.get(snapshot.reporter_id)
        if state is not None and snapshot.epoch in state.retired_epochs:
            logger.warning(
                f"Ignoring snapshot {snapshot.sequence} of reporter {snapshot.reporter_id}: its epoch was replaced "
                f"by {state.epoch}, so a later incarnation of that deployment is already reporting and this one "
                f"crossed the wan late"
            )
            return _ack(state)

        if state is not None and state.epoch != snapshot.epoch:
            self._reset_to_new_incarnation(snapshot, state=state)
        elif state is not None and snapshot.sequence <= state.sequence:
            logger.warning(
                f"Ignoring snapshot {snapshot.sequence} of reporter {snapshot.reporter_id}: snapshot "
                f"{state.sequence} of the same incarnation is at least as new, so this one arrived late"
            )
            return _ack(state)

        if parsed is None:
            return self._apply_heartbeat(snapshot, state=state)

        state = self._reporters.setdefault(snapshot.reporter_id, _ReporterState())
        excluded = self._replace_membership(reporter_id=snapshot.reporter_id, parsed=parsed)
        state.epoch = snapshot.epoch
        state.sequence = snapshot.sequence
        state.expected_num_cells_by_model = dict(snapshot.expected_num_cells_by_model)
        state.last_snapshot_at = self._clock()
        state.digest = None if excluded else snapshot.digest
        return _ack(state, excluded_cell_ids=excluded)

    def _reset_to_new_incarnation(self, snapshot: RegistrationSnapshot, *, state: _ReporterState) -> None:
        seconds = self._clock() - state.last_snapshot_at
        if seconds < EPOCH_CHURN_ERROR_SECONDS:
            logger.error(
                f"Reporter {snapshot.reporter_id} changed its epoch {seconds:.1f}s after the last snapshot of the "
                f"previous one, which is faster than a deployment is rebuilt: two deployments are very likely "
                f"sharing the instance name of --deploy-component inference:{snapshot.reporter_id}, and they will "
                f"keep deleting each other's cells"
            )
        else:
            logger.warning(
                f"Reporter {snapshot.reporter_id} reports under a new epoch, so it is a new incarnation of that "
                f"deployment; its sequence and digest are reset and its snapshot is taken as the whole truth"
            )
        state.retired_epochs.add(state.epoch)
        state.epoch = snapshot.epoch
        state.sequence = -1
        state.digest = None

    def _apply_heartbeat(self, snapshot: RegistrationSnapshot, *, state: _ReporterState | None) -> RegistrationAck:
        if state is None or state.digest != snapshot.digest:
            logger.warning(
                f"Reporter {snapshot.reporter_id} sent a heartbeat for a snapshot this run does not hold, so it "
                f"will be asked for the whole snapshot again"
            )
            if state is not None:
                state.sequence = snapshot.sequence
                state.last_snapshot_at = self._clock()
            return _ack(state)

        state.sequence = snapshot.sequence
        state.last_snapshot_at = self._clock()
        return _ack(state)

    def _replace_membership(self, *, reporter_id: str, parsed: _ParsedSnapshot) -> list[str]:
        excluded = list(parsed.excluded_cell_ids)
        cells: dict[str, _OwnedCell] = {}
        for cell_id, cell in parsed.cells.items():
            if (owner := self._cells.get(cell_id)) is not None and owner.reporter_id != reporter_id:
                logger.error(
                    f"Excluding cell {cell_id} of reporter {reporter_id} from this run: reporter "
                    f"{owner.reporter_id} already reported it, so two deployments share a pool id"
                )
                excluded.append(cell_id)
                continue
            cells[cell_id] = cell

        owned = {cell_id: cell for cell_id, cell in self._cells.items() if cell.reporter_id == reporter_id}
        removed = sorted(set(owned) - set(cells))
        changed = sorted(
            cell_id for cell_id, cell in cells.items() if owned.get(cell_id) != cell or cell_id in self._to_reannounce
        )

        for cell_id in removed:
            del self._cells[cell_id]
            self._pending.append(
                _PendingDispatch(
                    reporter_id=reporter_id,
                    cell_id=cell_id,
                    observed=None,
                    applied=None,
                    previous=owned[cell_id],
                )
            )
        for cell_id in changed:
            self._cells[cell_id] = cells[cell_id]
            self._pending.append(
                _PendingDispatch(
                    reporter_id=reporter_id,
                    cell_id=cell_id,
                    observed=cells[cell_id].info,
                    applied=cells[cell_id],
                    previous=owned.get(cell_id),
                )
            )
        return sorted(set(excluded))

    def _ensure_dispatcher(self) -> None:
        if not self._pending:
            return
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._drain_pending(), context=compute_context_without_lock_grant())

    async def _drain_pending(self) -> None:
        try:
            while self._pending:
                pending = self._pending.popleft()
                if not await self._dispatch(pending):
                    self._pending.appendleft(pending)
                    return
        except Exception:
            logger.error(
                "Draining the registered cells this run has to reconcile failed, so the ones still queued wait for "
                "the next snapshot of their reporter",
                exc_info=True,
            )

    async def _wait_pending_dispatches(self) -> None:
        while (dispatcher := self._dispatcher) is not None and not dispatcher.done():
            await asyncio.wait([dispatcher])
        if (dispatcher := self._dispatcher) is None or dispatcher.cancelled():
            return
        if (error := dispatcher.exception()) is not None:
            logger.error("Draining the registered cells this run has to reconcile ended abruptly", exc_info=error)

    async def _dispatch(self, pending: _PendingDispatch) -> bool:
        failure: BaseException | None = None
        async with self._reconcile_lock:
            if (reconcile := self._reconcile) is None:
                return False
            try:
                await reconcile(pending.cell_id, pending.observed)
            except BaseException as e:
                logger.error(
                    f"Reconciling registered cell {pending.cell_id} failed, so it is reported again", exc_info=True
                )
                failure = e

        if failure is None:
            self._to_reannounce.discard(pending.cell_id)
            return True

        await self._undo(pending)
        if not isinstance(failure, Exception):
            raise failure
        return True

    async def _undo(self, pending: _PendingDispatch) -> None:
        async with self._lock:
            self._to_reannounce.add(pending.cell_id)
            if self._cells.get(pending.cell_id) is pending.applied:
                if pending.previous is None:
                    self._cells.pop(pending.cell_id, None)
                else:
                    self._cells[pending.cell_id] = pending.previous
            if (state := self._reporters.get(pending.reporter_id)) is not None:
                state.digest = None

    def _worker_infos_of_cell(self, cell_id: str) -> list[WorkerInfo]:
        cell = self._cells[cell_id]
        return [
            WorkerInfo(
                name=worker_name,
                generation=0,
                self_addrs=cell.addrs_by_worker[worker_name],
                gpu_ids=cell.gpu_ids_by_worker[worker_name],
                handle=None,
                worker_class=None,
            )
            for worker_name in cell.info.worker_names
        ]


def _compute_cell_info(cell: RegisteredCell) -> CellInfo:
    return CellInfo(
        cell_id=cell.cell_id,
        pool_id=cell.pool_id,
        alive=True,
        worker_names=[worker.name for worker in cell.workers],
        workers_hash=cell.workers_hash,
        meta=dict(cell.meta) | {GATE_TIMEOUT_META_KEY: REGISTERED_LAUNCH_GATE_TIMEOUT_SECONDS},
    )


def _compute_refusal_reason(cell: RegisteredCell) -> str | None:
    if (parsed := _parse_cell_id_or_none(cell.cell_id)) is None:
        return (
            "its cell id does not read as <pool id>-<cell index>, and this run parses a cell id to address the "
            "workers of that cell"
        )
    pool_id, cell_index = parsed
    if pool_id != cell.pool_id or compute_cell_id(pool_id=pool_id, cell_index=cell_index) != cell.cell_id:
        return (
            f"it does not name its own pool {cell.pool_id}, and a reporter namespaces its pool ids so that two "
            f"deployments never collide"
        )
    if not cell.workers:
        return "it carries no worker to address"
    for worker in cell.workers:
        if _parse_worker_cell_or_none(worker.name) != (pool_id, cell_index):
            return f"worker {worker.name} does not belong to it"
    if cell.meta.get("worker_type") != SUPPORTED_WORKER_TYPE:
        return (
            f"it is a {cell.meta.get('worker_type')!r} engine, and prefill/decode engines of another deployment "
            f"cannot be paired yet, so this run only takes {SUPPORTED_WORKER_TYPE!r} ones"
        )
    return None


def _parse_cell_id_or_none(cell_id: str) -> ParsedCellId | None:
    try:
        return parse_cell_id(cell_id)
    except ValueError:
        return None


def _parse_worker_cell_or_none(worker_name: str) -> tuple[str, int] | None:
    try:
        return parse_worker_name(worker_name)[:2]
    except ValueError:
        return None


def _ack(state: _ReporterState | None, *, excluded_cell_ids: list[str] | None = None) -> RegistrationAck:
    if state is None:
        return RegistrationAck(applied_sequence=-1, applied_digest=None, excluded_cell_ids=excluded_cell_ids or [])
    return RegistrationAck(
        applied_sequence=state.sequence,
        applied_digest=state.digest,
        excluded_cell_ids=excluded_cell_ids or [],
    )
