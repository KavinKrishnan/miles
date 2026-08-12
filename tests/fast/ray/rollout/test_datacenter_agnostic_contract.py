from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import compute_server_cell_meta_from_info
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo
from miles.utils.workers.worker_spec import NamedHostAndPorts

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CORE_MODULES = [
    "miles/ray/rollout/inference_controller.py",
    "miles/ray/rollout/rollout_server.py",
    "miles/ray/rollout/server_cell.py",
]
_DATACENTER_WORDS = ["datacenter", "data_center", "instance_name", "is_remote", "remote_cell"]
_FLEET_MODULES = ["miles/ray/rollout/rollout_server.py", "miles/ray/rollout/server_cell.py"]


def _source_of(module_path: str) -> str:
    return (_REPO_ROOT / module_path).read_text()


def _imported_modules(module_path: str) -> set[str]:
    tree = ast.parse(_source_of(module_path))
    ans: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            ans.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            ans.add(node.module)
    return ans


class TestTheCoreNeverBranchesOnDatacenter:
    @pytest.mark.parametrize("module_path", _CORE_MODULES)
    def test_no_core_module_names_a_datacenter_concept(self, module_path: str):
        """A per-datacenter branch here is how a fleet abstraction rots; the meta dict carries it instead."""
        source = _source_of(module_path).lower()

        assert [word for word in _DATACENTER_WORDS if word in source] == []

    @pytest.mark.parametrize("module_path", _FLEET_MODULES)
    def test_no_fleet_module_knows_that_a_reporter_exists(self, module_path: str):
        """The controller owns the registration endpoint; below it a cell has no origin at all."""
        assert "reporter" not in _source_of(module_path).lower()

    @pytest.mark.parametrize("module_path", _FLEET_MODULES)
    def test_the_fleet_never_imports_the_registration_layer(self, module_path: str):
        """Registered engines reach the fleet as ordinary cells of an ordinary provider, or not at all."""
        registration_imports = [
            module for module in _imported_modules(module_path) if "workers.registration" in module
        ]

        assert registration_imports == []


class TestRegisteredCellsTakeTheLocalPath:
    def test_a_registered_cell_and_a_local_cell_read_the_same_way(self):
        """The controller groups a remote engine by its model exactly as it groups a local one."""
        local = _cell_info(pool_id="inference-engine-0-0", cell_index=1)
        registered = _cell_info(pool_id="east-inference-engine-0-0", cell_index=1)

        local_meta = compute_server_cell_meta_from_info(local)
        registered_meta = compute_server_cell_meta_from_info(registered)

        assert local_meta.model_id == registered_meta.model_id == "actor"
        assert local_meta.num_gpus_per_engine == registered_meta.num_gpus_per_engine
        assert registered_meta.cell_id == "east-inference-engine-0-0-1"
        assert registered_meta.worker_name == "east-inference-engine-0-0-1-0"


class _RecordingProvider(BaseWorkerProvider):
    def __init__(self) -> None:
        self.invalidated: list[str] = []

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise AssertionError(f"no cell of this module is ever addressed ({worker_name=})")

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list]:
        raise AssertionError(f"no cell of this module is ever inspected ({cell_ids=})")

    def invalidate_cell(self, cell_id: str) -> None:
        self.invalidated.append(cell_id)


class TestTheProbeRemovalPathDoesNotBranchOnOrigin:
    async def test_a_registered_cell_is_dropped_by_the_probe_exactly_like_a_local_one(self):
        """The removal path is where an origin branch would hide; a remote engine has to leave the same way."""
        provider = _RecordingProvider()
        srv = _make_server(
            provider,
            cells={
                "inference-engine-0-0-0": SimpleNamespace(is_unreachable=True),
                "east-inference-engine-0-0-0": SimpleNamespace(is_unreachable=True),
            },
        )

        async with srv.context_lock:
            srv.remove_unreachable_cells()

        assert provider.invalidated == ["inference-engine-0-0-0", "east-inference-engine-0-0-0"]

    async def test_a_reachable_cell_is_left_alone(self):
        """Dropping a healthy engine would restart a whole datacenter every five seconds."""
        provider = _RecordingProvider()
        srv = _make_server(provider, cells={"east-inference-engine-0-0-0": SimpleNamespace(is_unreachable=False)})

        async with srv.context_lock:
            srv.remove_unreachable_cells()

        assert provider.invalidated == []

    async def test_a_provider_that_re_observes_its_own_cells_treats_invalidation_as_a_no_op(self):
        """A platform provider reads its own source of truth, so it needs nothing from the probe but a hint."""
        assert BaseWorkerProvider.invalidate_cell(_RecordingProvider(), "any-cell-0") is None


def _make_server(provider: BaseWorkerProvider, *, cells: dict) -> RolloutServer:
    return RolloutServer(
        server_cells=cells,
        args=SimpleNamespace(colocate=False),
        context_lock=ContextLock("InferenceController"),
        engine_provider=provider,
        model_name="actor",
    )


def _cell_info(*, pool_id: str, cell_index: int) -> CellInfo:
    cell_id = f"{pool_id}-{cell_index}"
    return CellInfo(
        cell_id=cell_id,
        pool_id=pool_id,
        alive=True,
        worker_names=[f"{cell_id}-0"],
        workers_hash="hash-1",
        meta=dict(
            model_id="actor",
            worker_type="regular",
            num_gpus_per_engine=2,
            gpu_offset=0,
            sglang_api_key=None,
            needs_offload=False,
            update_weights=True,
        ),
    )
