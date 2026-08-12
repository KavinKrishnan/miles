from __future__ import annotations

import pytest

from miles.utils.workers.types import ClusterBackend, WorkerCommBackend, resolve_worker_comm_backend


class TestTheAutomaticChoice:
    def test_ray_keeps_talking_over_ray_until_the_default_flips(self):
        """The flag exists so both modes coexist; today an unset flag must change nothing for ray users."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.RAY, requested=None)

        assert chosen == WorkerCommBackend.RAY

    def test_kubernetes_talks_over_rpc(self):
        """A pod is not an actor, so the only way to call it is the server it serves."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.KUBERNETES, requested=None)

        assert chosen == WorkerCommBackend.RPC


class TestTheExplicitChoice:
    @pytest.mark.parametrize("requested", ["ray", "rpc"])
    def test_ray_accepts_both_modes(self, requested: str):
        """Ray is where the two modes coexist, which is what makes a gradual switch possible."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.RAY, requested=requested)

        assert chosen == WorkerCommBackend(requested)

    def test_kubernetes_accepts_rpc(self):
        """Naming the backend that is already in use must not be an error."""
        chosen = resolve_worker_comm_backend(cluster_backend=ClusterBackend.KUBERNETES, requested="rpc")

        assert chosen == WorkerCommBackend.RPC

    def test_kubernetes_refuses_ray_communication(self):
        """There is no actor to call, so accepting the flag would fail much later and far less clearly."""
        with pytest.raises(AssertionError, match="worker-comm-backend"):
            resolve_worker_comm_backend(cluster_backend=ClusterBackend.KUBERNETES, requested="ray")

    def test_an_unknown_backend_is_rejected(self):
        """A typo must not silently fall back to the default."""
        with pytest.raises(ValueError):
            resolve_worker_comm_backend(cluster_backend=ClusterBackend.RAY, requested="grpc")
