import pytest

from miles.utils.external_utils.command_utils.helm_backend.naming import RunFiles, RunNames, _orchestrator_state_path
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    OrchestratorState,
    OrchestratorStatus,
)
from miles.utils.workers.types import DeployComponent, DeploySelector

RUN_ID = "260101-000000-000"


def _write(path, status: OrchestratorStatus, *, exit_code: int | None = None) -> None:
    OrchestratorState(status=status, exit_code=exit_code).write(path)


def _state_file(tmp_path):
    return _orchestrator_state_path(tmp_path, "260101-000000-000001")


class TestRelease:
    def test_an_instance_is_sanitized_into_the_release_name(self):
        """A release name is a dns label, so a model id holding underscores, dots or capitals cannot go in as is."""
        release = RunNames.release(
            run_id=RUN_ID, deploy_component=DeployComponent.TRAINER, deploy_instance="Policy_A.1"
        )

        assert release == f"miles-run-{RUN_ID}-trainer-policy-a-1"

    def test_two_instances_of_one_run_get_two_releases(self):
        """A shared name would make the second trainer's install overwrite the first one's objects."""
        first = RunNames.release(run_id=RUN_ID, deploy_component=DeployComponent.TRAINER, deploy_instance="policy_a")
        second = RunNames.release(run_id=RUN_ID, deploy_component=DeployComponent.TRAINER, deploy_instance="policy_b")

        assert first != second

    def test_an_instance_of_a_component_still_names_that_component(self):
        """The component in the name is what tells a trainer release from an inference one at a glance."""
        release = RunNames.release(
            run_id=RUN_ID, deploy_component=DeployComponent.INFERENCE, deploy_instance="instance_b"
        )

        assert release == f"miles-run-{RUN_ID}-inference-instance-b"

    def test_the_whole_run_names_no_instance(self):
        """`all` installs every worker of the run, so an instance of it would name nothing."""
        with pytest.raises(AssertionError, match="no instance of it is named"):
            RunNames.release(run_id=RUN_ID, deploy_component=DeployComponent.ALL, deploy_instance="policy_a")

    def test_an_instance_that_sanitizes_to_nothing_is_refused(self):
        """Every object the release installs is named after it, and an empty name would install nothing addressable."""
        with pytest.raises(AssertionError, match="at least one"):
            RunNames.release(run_id=RUN_ID, deploy_component=DeployComponent.TRAINER, deploy_instance="__")


class TestReleaseOf:
    def test_a_selector_names_the_release_its_launch_installs(self):
        """The selector is what a launch is told, so it has to resolve to the same release the component does."""
        release = RunNames.release_of(run_id=RUN_ID, selector=DeploySelector.parse("trainer:policy_a"))

        assert release == RunNames.release(
            run_id=RUN_ID, deploy_component=DeployComponent.TRAINER, deploy_instance="policy_a"
        )

    def test_a_selector_without_an_instance_names_the_component_release(self):
        """A run with one trainer installs it under the component alone, exactly as before instances existed."""
        release = RunNames.release_of(run_id=RUN_ID, selector=DeploySelector.parse("trainer"))

        assert release == f"miles-run-{RUN_ID}-trainer"

    def test_the_whole_run_resolves_to_the_run_release(self):
        """An unsplit run keeps the one release name every existing run already has."""
        release = RunNames.release_of(run_id=RUN_ID, selector=DeploySelector.parse("all"))

        assert release == f"miles-run-{RUN_ID}"


class TestRunDir:
    def test_places_a_run_under_the_shared_root(self):
        """Every pod resolves the same run directory from the shared storage mount and the run id."""
        assert str(RunFiles.run_dir(shared_root="/cluster-storage/miles_data", run_id="260101-000000-000")).endswith(
            "/cluster-storage/miles_data/miles-runs/260101-000000-000"
        )

    def test_keeps_the_state_file_in_a_state_subdirectory(self):
        """Grouping the machine-written state keeps it out of the way of a run's own outputs."""
        path = _orchestrator_state_path("/runs/abc", "abc123")

        assert path.as_posix() == "/runs/abc/state/orchestrator-abc123.state"

    def test_gives_every_launch_its_own_record_file(self):
        """Two launches of one run must not overwrite each other's record of what they launched."""
        first = RunFiles.new_record_file(run_directory="/runs/abc")
        second = RunFiles.new_record_file(run_directory="/runs/abc")

        assert first.parent.as_posix() == "/runs/abc/launches"
        assert first != second


class TestLatestExitFile:
    def test_names_no_file_before_a_launch_has_written_one(self, tmp_path):
        """A run directory a launch has only just created holds no verdict to collect."""
        assert RunFiles.latest_state_file(run_directory=tmp_path) is None

    def test_picks_the_newest_launch_rather_than_the_newest_write(self, tmp_path):
        """An earlier launch torn down after a later one started writes last, and its verdict is not the run's."""
        later = _orchestrator_state_path(tmp_path, "260101-000200-000001")
        earlier = _orchestrator_state_path(tmp_path, "260101-000100-000002")
        _write(later, OrchestratorStatus.EXITED, exit_code=0)
        _write(earlier, OrchestratorStatus.EXITED, exit_code=1)

        assert RunFiles.latest_state_file(run_directory=tmp_path) == later
