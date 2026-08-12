from argparse import Namespace

from miles.utils.trainer_artifacts import compute_trainer_artifact_dir, compute_trainer_artifact_name


class TestComputeTrainerArtifactDir:
    def test_a_single_policy_run_keeps_the_root_untouched(self):
        """Without a trainer model id the artifact dir is the root itself."""
        assert compute_trainer_artifact_dir(Namespace(trainer_model_id=None), root="/shared/delta") == "/shared/delta"

    def test_each_policy_gets_its_own_subdirectory_of_the_root(self):
        """A trainer model id namespaces the artifact dir under the root."""
        assert (
            compute_trainer_artifact_dir(Namespace(trainer_model_id="policy_a"), root="/shared/delta")
            == "/shared/delta/policy_a"
        )

    def test_two_policies_never_share_an_artifact_dir(self):
        """Two policies of one run write their disk deltas into different dirs."""
        first = compute_trainer_artifact_dir(Namespace(trainer_model_id="policy_a"), root="/shared/delta")
        second = compute_trainer_artifact_dir(Namespace(trainer_model_id="policy_b"), root="/shared/delta")

        assert first != second


class TestComputeTrainerArtifactName:
    def test_a_single_policy_run_keeps_the_name_untouched(self):
        """Without a trainer model id the artifact name is unchanged."""
        assert compute_trainer_artifact_name(Namespace(trainer_model_id=None), name="weights") == "weights"

    def test_each_policy_prefixes_the_name_with_its_model_id(self):
        """A trainer model id prefixes the artifact name."""
        assert compute_trainer_artifact_name(Namespace(trainer_model_id="policy_a"), name="weights") == (
            "policy_a_weights"
        )

    def test_two_policies_never_share_an_artifact_name(self):
        """Two policies of one run name their artifacts differently."""
        first = compute_trainer_artifact_name(Namespace(trainer_model_id="policy_a"), name="weights")
        second = compute_trainer_artifact_name(Namespace(trainer_model_id="policy_b"), name="weights")

        assert first != second
