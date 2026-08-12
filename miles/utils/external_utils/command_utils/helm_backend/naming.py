from __future__ import annotations

import random
import re
import time
from pathlib import Path

from miles.utils.workers.types import DeployComponent, DeploySelector
from miles.utils.workers.worker_provider.kubernetes.helm.naming import CHART_NAME, component_name

ORCHESTRATOR_COMPONENT = "orchestrator"

_UNINSTALL_COMPONENT = "uninstall"
_UNINSTALL_MANIFEST_COMPONENT = "uninstall-manifest"

_RUNS_DIR_NAME = "miles-runs"
_STATE_DIR_NAME = "state"
_VALUES_DIR_NAME = "values"
_RECORDS_DIR_NAME = "launches"
_STATE_FILE_GLOB = "orchestrator-*.state"


class RunNames:
    @staticmethod
    def release(
        *,
        run_id: str,
        deploy_component: DeployComponent = DeployComponent.ALL,
        deploy_instance: str | None = None,
    ) -> str:
        if deploy_component is DeployComponent.ALL:
            assert deploy_instance is None, "`all` deploys the whole run, so no instance of it is named"
            return f"{CHART_NAME}-{run_id}"
        suffix = deploy_component.value
        if deploy_instance is not None:
            suffix = f"{suffix}-{sanitize_release_instance(deploy_instance)}"
        return f"{CHART_NAME}-{run_id}-{suffix}"

    @staticmethod
    def release_of(*, run_id: str, selector: DeploySelector) -> str:
        return RunNames.release(run_id=run_id, deploy_component=selector.component, deploy_instance=selector.instance)

    @staticmethod
    def service_fqdn(*, name: str, namespace: str) -> str:
        return f"{name}.{namespace}.svc.cluster.local"

    @staticmethod
    def orchestrator_host(*, release: str, namespace: str) -> str:
        return RunNames.service_fqdn(name=component_name(release, ORCHESTRATOR_COMPONENT), namespace=namespace)

    @staticmethod
    def uninstall_job(*, release: str) -> str:
        return component_name(release, _UNINSTALL_COMPONENT)

    @staticmethod
    def uninstall_manifest(*, release: str) -> str:
        return component_name(release, _UNINSTALL_MANIFEST_COMPONENT)


class RunFiles:
    @staticmethod
    def run_dir(*, shared_root: str | Path, run_id: str) -> Path:
        return Path(shared_root) / _RUNS_DIR_NAME / run_id

    @staticmethod
    def new_values_file(*, run_directory: str | Path) -> Path:
        return Path(run_directory) / _VALUES_DIR_NAME / f"values-{_new_launch_token()}.yaml"

    @staticmethod
    def new_state_file(*, run_directory: str | Path) -> Path:
        return _orchestrator_state_path(run_directory, _new_launch_token())

    @staticmethod
    def new_record_file(*, run_directory: str | Path) -> Path:
        return Path(run_directory) / _RECORDS_DIR_NAME / f"launch-{_new_launch_token()}.json"

    @staticmethod
    def latest_state_file(*, run_directory: str | Path) -> Path | None:
        """The newest launch's file, by the launch token in its name; see _new_launch_token."""
        written = sorted((Path(run_directory) / _STATE_DIR_NAME).glob(_STATE_FILE_GLOB))
        return written[-1] if written else None


def _orchestrator_state_path(run_directory: str | Path, launch_token: str) -> Path:
    return Path(run_directory) / _STATE_DIR_NAME / f"orchestrator-{launch_token}.state"


def _new_launch_token() -> str:
    return f"{time.strftime('%y%m%d-%H%M%S')}-{random.Random().randint(0, 999999):06d}"


def sanitize_release_instance(instance: str) -> str:
    sanitized = _INSTANCE_ILLEGAL_PATTERN.sub("-", instance.lower()).strip("-")
    assert sanitized, (
        f"the instance {instance!r} names every object its release installs, so it has to hold at least one "
        f"letter or digit"
    )
    return sanitized


_INSTANCE_ILLEGAL_PATTERN = re.compile(r"[^a-z0-9]+")
