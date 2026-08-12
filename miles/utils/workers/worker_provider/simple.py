from __future__ import annotations

from collections.abc import Iterable

from miles.utils.http_utils import wait_tcp_ready
from miles.utils.workers.naming import compute_cell_id, compute_worker_name, parse_cell_id, parse_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.utils import attach_rpc_handle
from miles.utils.workers.worker_spec import RPC_PORT_NAME, HostAndPort, NamedHostAndPorts

STATIC_ADDRS_READY_TIMEOUT_SECONDS = 600.0


class SimpleWorkerProvider(BaseWorkerProvider):
    def __init__(self, *, pool_id: str, addrs: list[NamedHostAndPorts], worker_class: str | None = None) -> None:
        assert addrs, f"pool {pool_id} is addressed statically, so it needs at least one address"
        self._pool_id = pool_id
        self._worker_class = worker_class
        self._addrs_by_cell_index = list(addrs)
        self._waited_ready = False

    @classmethod
    def of_rpc_urls(cls, *, pool_id: str, urls: list[str], worker_class: str) -> SimpleWorkerProvider:
        return cls(
            pool_id=pool_id,
            addrs=[{RPC_PORT_NAME: parse_host_and_port(url)} for url in urls],
            worker_class=worker_class,
        )

    def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        self._wait_ready_once()
        return super().get_handle(worker_name)

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return self._addrs_of_worker(worker_name)

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [[self._worker_info(self._worker_name_of_cell(cell_id))] for cell_id in cell_ids]

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        for cell_index in range(len(self._addrs_by_cell_index)):
            await reconcile(self._cell_id(cell_index), self._cell_info(cell_index))

        async def _stop() -> None:
            return None

        return _stop

    def _wait_ready_once(self) -> None:
        if self._waited_ready:
            return
        for addrs in self._addrs_by_cell_index:
            wait_static_addrs_ready(addrs.values())
        self._waited_ready = True

    def _cell_info(self, cell_index: int) -> CellInfo:
        worker_name = compute_worker_name(pool_id=self._pool_id, cell_index=cell_index)
        return CellInfo(
            cell_id=self._cell_id(cell_index),
            pool_id=self._pool_id,
            alive=True,
            worker_names=[worker_name],
            workers_hash=self._addrs_of_worker(worker_name)[RPC_PORT_NAME].addr,
            meta=dict(cell_index=cell_index),
        )

    def _cell_id(self, cell_index: int) -> str:
        return compute_cell_id(pool_id=self._pool_id, cell_index=cell_index)

    def _worker_name_of_cell(self, cell_id: str) -> str:
        pool_id, cell_index = parse_cell_id(cell_id)
        assert pool_id == self._pool_id and cell_index < len(self._addrs_by_cell_index), (
            f"{cell_id} is not one of the {len(self._addrs_by_cell_index)} statically given cells "
            f"of pool {self._pool_id}"
        )
        return compute_worker_name(pool_id=self._pool_id, cell_index=cell_index)

    def _worker_info(self, worker_name: str) -> WorkerInfo:
        return attach_rpc_handle(
            WorkerInfo(
                name=worker_name,
                generation=0,
                self_addrs=self._addrs_of_worker(worker_name),
                gpu_ids=[],
                handle=None,
                worker_class=self._worker_class,
            )
        )

    def _addrs_of_worker(self, worker_name: str) -> NamedHostAndPorts:
        pool_id, cell_index, worker_in_cell_index = parse_worker_name(worker_name)
        assert pool_id == self._pool_id, f"this provider answers for pool {self._pool_id}, not for {worker_name}"
        assert worker_in_cell_index == 0, (
            f"pool {self._pool_id} is addressed statically, so each of its cells holds exactly one worker, "
            f"and {worker_name} is not it"
        )
        assert cell_index < len(self._addrs_by_cell_index), (
            f"{worker_name} is not one of the {len(self._addrs_by_cell_index)} statically given workers "
            f"of pool {self._pool_id}"
        )
        return self._addrs_by_cell_index[cell_index]


def wait_static_addrs_ready(
    addrs: Iterable[HostAndPort], *, timeout: float = STATIC_ADDRS_READY_TIMEOUT_SECONDS
) -> None:
    for addr in addrs:
        wait_tcp_ready(addr.host, addr.port, timeout=timeout)


def parse_host_and_port(addr: str) -> HostAndPort:
    rest = addr.split("://", 1)[1] if "://" in addr else addr
    host, separator, port = rest.rstrip("/").rpartition(":")
    assert separator and port.isdigit(), f"static address {addr!r} must be host:port or http://host:port"
    return HostAndPort(host=host.strip("[]"), port=int(port))
