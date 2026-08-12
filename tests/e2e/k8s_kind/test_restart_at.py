from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from kubernetes_asyncio import client as kubernetes_client
from tests.ci.ci_register import register_cpu_ci
from tests.e2e.k8s_apiserver.utils import BUSYBOX_IMAGE, unique_name
from tests.e2e.k8s_kind.kind_cluster import KindCluster
from tests.e2e.k8s_kind.utils import build_kubeconfig_api_client

register_cpu_ci(est_time=480, suite="stage-b-cpu", labels=[])

RESTART_AT_ANNOTATION = "miles.radixark.io/restart-at"

_STARTUP_TIMEOUT = 180.0
_REPLACEMENT_TIMEOUT = 180.0
_SETTLE_SECONDS = 20.0


def _stateful_set(name: str, *, restart_at: str | None = None) -> kubernetes_client.V1StatefulSet:
    annotations = {} if restart_at is None else {RESTART_AT_ANNOTATION: restart_at}
    return kubernetes_client.V1StatefulSet(
        metadata=kubernetes_client.V1ObjectMeta(name=name),
        spec=kubernetes_client.V1StatefulSetSpec(
            replicas=1,
            service_name=name,
            selector=kubernetes_client.V1LabelSelector(match_labels={"app": name}),
            template=kubernetes_client.V1PodTemplateSpec(
                metadata=kubernetes_client.V1ObjectMeta(labels={"app": name}, annotations=annotations),
                spec=kubernetes_client.V1PodSpec(
                    termination_grace_period_seconds=1,
                    containers=[
                        kubernetes_client.V1Container(name="worker", image=BUSYBOX_IMAGE, command=["sleep", "3600"])
                    ],
                ),
            ),
        ),
    )


class TestRestartAtReplacesOnlyItsOwnStatefulSet:
    async def test_stamping_the_annotation_replaces_that_pod_and_leaves_the_other_running(
        self, kind_cluster: KindCluster, cluster_core_v1: kubernetes_client.CoreV1Api, cluster_namespace: str
    ) -> None:
        """This is the whole kubernetes-side mechanism of a hot restart: two components roll, the rest do not."""
        api_client = await build_kubeconfig_api_client(kubeconfig=kind_cluster.kubeconfig)
        apps = kubernetes_client.AppsV1Api(api_client)
        try:
            restarted = unique_name("orchestrator")
            untouched = unique_name("trainer")
            for name in (restarted, untouched):
                await apps.create_namespaced_stateful_set(namespace=cluster_namespace, body=_stateful_set(name))

            restarted_before = await _wait_for_pod(cluster_core_v1, cluster_namespace, restarted)
            untouched_before = await _wait_for_pod(cluster_core_v1, cluster_namespace, untouched)

            await apps.patch_namespaced_stateful_set(
                name=restarted,
                namespace=cluster_namespace,
                body={
                    "spec": {
                        "template": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "2026-08-12T09:00:00+00:00"}}}
                    }
                },
            )

            replacement = await _wait_until_replaced(
                cluster_core_v1, cluster_namespace, restarted, previous_uid=restarted_before
            )

            assert replacement != restarted_before
            assert await _pod_uid(cluster_core_v1, cluster_namespace, untouched) == untouched_before
        finally:
            await api_client.close()

    async def test_relaunching_without_a_new_stamp_replaces_nothing(
        self, kind_cluster: KindCluster, cluster_core_v1: kubernetes_client.CoreV1Api, cluster_namespace: str
    ) -> None:
        """An ordinary relaunch renders the same template, and it must not roll a live run's pods."""
        api_client = await build_kubeconfig_api_client(kubeconfig=kind_cluster.kubeconfig)
        apps = kubernetes_client.AppsV1Api(api_client)
        try:
            name = unique_name("orchestrator-stable")
            stamp = "2026-08-12T09:00:00+00:00"
            await apps.create_namespaced_stateful_set(
                namespace=cluster_namespace, body=_stateful_set(name, restart_at=stamp)
            )
            before = await _wait_for_pod(cluster_core_v1, cluster_namespace, name)

            await apps.patch_namespaced_stateful_set(
                name=name,
                namespace=cluster_namespace,
                body={"spec": {"template": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: stamp}}}}},
            )
            await asyncio.sleep(_SETTLE_SECONDS)

            assert await _pod_uid(cluster_core_v1, cluster_namespace, name) == before
        finally:
            await api_client.close()


async def _pod_uid(core_v1: kubernetes_client.CoreV1Api, namespace: str, name: str) -> str | None:
    listed = await core_v1.list_namespaced_pod(namespace=namespace, label_selector=f"app={name}")
    running = [pod for pod in listed.items if pod.metadata.deletion_timestamp is None]
    return running[0].metadata.uid if running else None


async def _wait_for_pod(core_v1: kubernetes_client.CoreV1Api, namespace: str, name: str) -> str:
    holder: dict[str, str] = {}

    async def pod_is_running() -> bool:
        listed = await core_v1.list_namespaced_pod(namespace=namespace, label_selector=f"app={name}")
        for pod in listed.items:
            if pod.status.phase == "Running":
                holder["uid"] = pod.metadata.uid
                return True
        return False

    await _wait_until_async(pod_is_running, description=f"{name} to reach Running", timeout=_STARTUP_TIMEOUT)
    return holder["uid"]


async def _wait_until_replaced(
    core_v1: kubernetes_client.CoreV1Api, namespace: str, name: str, *, previous_uid: str
) -> str:
    holder: dict[str, str] = {}

    async def replaced() -> bool:
        listed = await core_v1.list_namespaced_pod(namespace=namespace, label_selector=f"app={name}")
        for pod in listed.items:
            if pod.metadata.uid != previous_uid and pod.status.phase == "Running":
                holder["uid"] = pod.metadata.uid
                return True
        return False

    await _wait_until_async(replaced, description=f"{name} to be replaced", timeout=_REPLACEMENT_TIMEOUT)
    return holder["uid"]


async def _wait_until_async(condition: Callable[[], Awaitable[bool]], *, description: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await condition():
            return
        await asyncio.sleep(1.0)
    raise AssertionError(f"timed out after {timeout}s waiting for {description}")
