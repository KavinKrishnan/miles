import logging

import yaml

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    RESTART_AT_ANNOTATION,
    Manifest,
)

ORCHESTRATOR = "myrun-miles-run-orchestrator"


def _rendered(*documents: dict) -> str:
    return "---\n" + "---\n".join(yaml.safe_dump(document, sort_keys=True) for document in documents)


def _stateful_set(*, name: str = ORCHESTRATOR, command: list[str] | None = None) -> dict:
    container = {"name": "orchestrator", "image": "miles:dev"}
    if command is not None:
        container["command"] = command
    return {
        "apiVersion": "apps/v1",
        "kind": "StatefulSet",
        "metadata": {"name": name},
        "spec": {"replicas": 1, "template": {"spec": {"containers": [container]}}},
    }


class TestParse:
    def test_keys_an_object_by_its_kind_and_name(self):
        """Two kinds can share a name in one release, so neither alone identifies an object."""
        manifest = Manifest.parse(_rendered(_stateful_set()))

        assert list(manifest.by_key) == [f"StatefulSet/{ORCHESTRATOR}"]

    def test_skips_the_empty_documents_helm_leaves_behind(self):
        """A template whose guard is off renders to nothing, and yaml reads that as a None document."""
        manifest = Manifest.parse("---\n---\n" + _rendered(_stateful_set()))

        assert len(manifest.objects) == 1

    def test_reads_the_replica_count_a_resize_moves(self):
        """The upgrade check compares this number, and a kind that has none must not read as zero."""
        manifest = Manifest.parse(_rendered(_stateful_set(), {"kind": "ConfigMap", "metadata": {"name": "values"}}))

        assert manifest.by_key[f"StatefulSet/{ORCHESTRATOR}"].replicas == 1
        assert manifest.by_key["ConfigMap/values"].replicas is None


class TestStateFile:
    def test_finds_the_file_the_installed_orchestrator_already_writes(self):
        """Re-attaching means waiting on the verdict of the launch that is running, not opening a second one."""
        manifest = Manifest.parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert str(manifest.state_file(container="orchestrator")) == "/runs/a.state"

    def test_names_nothing_when_no_container_of_that_name_carries_the_flag(self):
        """A release installed without an orchestrator has no verdict to inherit."""
        manifest = Manifest.parse(_rendered(_stateful_set(command=["python", "-m", "something"])))

        assert manifest.state_file(container="orchestrator") is None

    def test_reads_only_the_container_it_was_asked_about(self):
        """Every pod of a run is launched by the same image, and the flag only means this on the orchestrator."""
        manifest = Manifest.parse(_rendered(_stateful_set(command=["python", "--state-file", "/runs/a.state"])))

        assert manifest.state_file(container="worker") is None


class TestKindsItDoesNotModel:
    def test_carries_every_kind_this_chart_renders_through_untouched(self):
        """Only replicas and a pod template are read; the rest of a spec still has to reach the diff verbatim."""
        documents = [
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": "engine"},
                "spec": {"clusterIP": "None", "ports": [{"port": 30000, "name": "primary"}]},
            },
            {
                "apiVersion": "leaderworkerset.x-k8s.io/v1",
                "kind": "LeaderWorkerSet",
                "metadata": {"name": "engine"},
                "spec": {
                    "replicas": 2,
                    "leaderWorkerTemplate": {"size": 2, "workerTemplate": {"spec": {"containers": []}}},
                },
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "Role",
                "metadata": {"name": "pairing"},
                "rules": [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}],
            },
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": "uninstall"},
                "spec": {
                    "completions": 1,
                    "template": {"spec": {"containers": [{"name": "helm", "image": "miles:dev"}]}},
                },
            },
        ]

        manifest = Manifest.parse(_rendered(*documents))

        assert [described.body for described in manifest.objects] == documents

    def test_reads_no_replicas_off_a_kind_that_has_none(self):
        """A Service and a Role never scale, and inventing a count for them would read as a resize."""
        manifest = Manifest.parse(
            _rendered({"kind": "Service", "metadata": {"name": "engine"}, "spec": {"clusterIP": "None"}})
        )

        assert manifest.objects[0].replicas is None

    def test_finds_no_container_in_a_workload_it_does_not_model(self):
        """The state file flag means the orchestrator's container, and every other pod runs the same image."""
        manifest = Manifest.parse(
            _rendered(
                {
                    "apiVersion": "leaderworkerset.x-k8s.io/v1",
                    "kind": "LeaderWorkerSet",
                    "metadata": {"name": "engine"},
                    "spec": {
                        "leaderWorkerTemplate": {
                            "workerTemplate": {
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "orchestrator",
                                            "command": ["python", "--state-file", "/runs/a.state"],
                                        }
                                    ]
                                }
                            }
                        }
                    },
                }
            )
        )

        assert manifest.state_file(container="orchestrator") is None


_STAMP = "2026-08-12T09:00:00+00:00"


def _stamped(name: str, stamp: str | None) -> dict:
    described = _stateful_set(name=name)
    if stamp is not None:
        described["spec"]["template"]["metadata"] = {"annotations": {RESTART_AT_ANNOTATION: stamp}}
    return described


class TestTheRestartStamp:
    def test_a_run_that_was_never_hot_restarted_carries_none(self):
        """An ordinary run has no annotation to preserve, and inventing one would roll its pods."""
        manifest = Manifest.parse(_rendered(_stateful_set()))

        assert manifest.restart_at(preferred_object_name=ORCHESTRATOR) is None

    def test_the_stamp_of_a_hot_restarted_run_is_readable(self):
        """The next ordinary relaunch renders this value back, so it has to be recoverable from the cluster."""
        manifest = Manifest.parse(_rendered(_stamped(ORCHESTRATOR, _STAMP)))

        assert manifest.restart_at(preferred_object_name=ORCHESTRATOR) == _STAMP

    def test_the_objects_of_one_hot_restart_share_one_stamp(self):
        """One hot restart writes one timestamp onto both objects it replaces."""
        manifest = Manifest.parse(_rendered(_stamped(ORCHESTRATOR, _STAMP), _stamped("executor", _STAMP)))

        assert manifest.restart_at(preferred_object_name=ORCHESTRATOR) == _STAMP

    def test_objects_stamped_differently_keep_the_orchestrators_stamp(self, caplog):
        """An interrupted upgrade leaves two stamps, and refusing every later launch would strand the run."""
        manifest = Manifest.parse(
            _rendered(_stamped(ORCHESTRATOR, _STAMP), _stamped("executor", "2026-01-01T00:00:00+00:00"))
        )

        with caplog.at_level(logging.WARNING):
            assert manifest.restart_at(preferred_object_name=ORCHESTRATOR) == _STAMP

        assert "restart stamps" in caplog.text

    def test_a_stamp_of_another_object_is_carried_when_the_orchestrator_has_none(self):
        """The orchestrator may be renamed or absent, and a stamp that exists still has to be preserved."""
        manifest = Manifest.parse(_rendered(_stamped("executor", _STAMP)))

        assert manifest.restart_at(preferred_object_name=ORCHESTRATOR) == _STAMP

    def test_two_kinds_sharing_a_name_are_two_stamps(self, caplog):
        """Kind and name identify an object, so keying the stamps by name alone would hide a real divergence."""
        deployment = _stamped("executor", "2026-01-01T00:00:00+00:00")
        deployment["kind"] = "Deployment"
        manifest = Manifest.parse(_rendered(_stamped("executor", _STAMP), deployment))

        with caplog.at_level(logging.WARNING):
            assert manifest.restart_at(preferred_object_name="executor") == _STAMP

        assert "restart stamps" in caplog.text
