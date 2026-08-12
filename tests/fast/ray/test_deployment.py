from __future__ import annotations

from argparse import Namespace

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability

from miles.ray import deployment
from miles.ray.specs.static_addrs import inference_controller_urls, trainer_controller_urls
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import RPC_PORT_NAME, HostAndPort, NamedHostAndPorts

pytestmark = pytest.mark.asyncio


class _AddressBookProvider(BaseWorkerProvider):
    def __init__(self, addrs_by_worker_name: dict[str, HostAndPort]) -> None:
        self._addrs_by_worker_name = addrs_by_worker_name

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return {RPC_PORT_NAME: self._addrs_by_worker_name[worker_name]}

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        raise NotImplementedError


def _describe(monkeypatch, *, args: Namespace, component: DeployComponent, addrs: dict[str, HostAndPort]) -> str:
    capability = FakeBackendCapability(static_provider=_AddressBookProvider(addrs))
    monkeypatch.setattr(deployment, "get_backend_capability", lambda _args: capability)
    return deployment._describe_controller_addrs(args, component=component)


def _entries_after(description: str, flag: str) -> list[str]:
    tokens = description.split()
    return tokens[tokens.index(flag) + 1 :]


class TestDescribeControllerAddrs:
    async def test_the_inference_address_it_prints_is_the_one_the_next_launch_takes(self, monkeypatch):
        """This string is the only place a user reads the address from, so it has to parse back unchanged."""
        description = await _describe(
            monkeypatch,
            args=Namespace(use_critic=False),
            component=DeployComponent.INFERENCE,
            addrs={"inference-controller-0-0": HostAndPort(host="inference-host", port=8000)},
        )

        entries = _entries_after(description, "--inference-controller-addrs")

        assert inference_controller_urls(Namespace(inference_controller_addrs=entries)) == ["inference-host:8000"]

    async def test_the_trainer_address_it_prints_is_the_one_the_next_launch_takes(self, monkeypatch):
        """Same round trip for the trainer, whose entries carry a role prefix the parser has to accept."""
        description = await _describe(
            monkeypatch,
            args=Namespace(use_critic=False),
            component=DeployComponent.TRAINER,
            addrs={"trainer-controller-actor-0-0": HostAndPort(host="trainer-host", port=8000)},
        )

        entries = _entries_after(description, "--trainer-controller-addrs")

        assert trainer_controller_urls(Namespace(trainer_controller_addrs=entries), role="actor") == [
            "trainer-host:8000"
        ]

    async def test_a_run_with_a_critic_prints_both_of_its_controllers(self, monkeypatch):
        """A critic is its own controller, and an unnamed one leaves the next launch unable to reach it."""
        description = await _describe(
            monkeypatch,
            args=Namespace(use_critic=True),
            component=DeployComponent.TRAINER,
            addrs={
                "trainer-controller-actor-0-0": HostAndPort(host="actor-host", port=8000),
                "trainer-controller-critic-0-0": HostAndPort(host="critic-host", port=8000),
            },
        )

        entries = _entries_after(description, "--trainer-controller-addrs")
        args = Namespace(trainer_controller_addrs=entries)

        assert trainer_controller_urls(args, role="actor") == ["actor-host:8000"]
        assert trainer_controller_urls(args, role="critic") == ["critic-host:8000"]
