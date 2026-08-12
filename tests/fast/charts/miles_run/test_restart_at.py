import json

from tests.fast.charts.utils import named_object, render_run, requires_helm, with_object_names

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import RESTART_AT_ANNOTATION

ORCHESTRATOR = "myrun-miles-run-orchestrator"
ROLLOUT_EXECUTOR = "myrun-miles-run-rollout-executor"
STAMP = "2026-08-12T09:00:00+00:00"

WORKERS = [
    {"name": "rollout-executor", "command": ["python", "-m", "serve"]},
    {"name": "router", "command": ["python", "-m", "router"]},
]


def _render(*args: str, workers=WORKERS):
    return render_run("--set-json", f"run.staticWorkers={json.dumps(with_object_names(workers))}", *args)


def _pod_annotations(objects, name: str) -> dict:
    return named_object(objects, "StatefulSet", name)["spec"]["template"]["metadata"].get("annotations", {})


@requires_helm
class TestRestartAtAnnotation:
    def test_an_ordinary_launch_stamps_no_annotation(self):
        """Stamping a value on every launch would roll a live run's pods for nothing."""
        objects = _render()

        assert _pod_annotations(objects, ORCHESTRATOR) == {}
        assert _pod_annotations(objects, ROLLOUT_EXECUTOR) == {}

    def test_the_orchestrator_carries_the_stamp_it_is_given(self):
        """The pod template only changes, and the StatefulSet only rolls, because of this annotation."""
        objects = _render("--set", f"run.orchestrator.restartAt={STAMP}")

        assert _pod_annotations(objects, ORCHESTRATOR) == {RESTART_AT_ANNOTATION: STAMP}

    def test_a_static_worker_carries_the_stamp_it_is_given(self):
        """The rollout executor is a static worker, and it is the second component a hot restart replaces."""
        workers = [{**WORKERS[0], "restartAt": STAMP}, WORKERS[1]]
        objects = _render(workers=workers)

        assert _pod_annotations(objects, ROLLOUT_EXECUTOR) == {RESTART_AT_ANNOTATION: STAMP}

    def test_a_worker_without_a_stamp_is_left_alone(self):
        """A hot restart replaces two components; every other pod of the run must keep running."""
        workers = [{**WORKERS[0], "restartAt": STAMP}, WORKERS[1]]
        objects = _render(workers=workers)

        assert _pod_annotations(objects, "myrun-miles-run-router") == {}

    def test_the_stamp_does_not_reach_the_statefulset_metadata(self):
        """Only a pod template change rolls the pods, so the annotation has to sit on the template."""
        objects = _render("--set", f"run.orchestrator.restartAt={STAMP}")

        assert RESTART_AT_ANNOTATION not in named_object(objects, "StatefulSet", ORCHESTRATOR)["metadata"].get(
            "annotations", {}
        )
