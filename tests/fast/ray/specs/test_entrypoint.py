from __future__ import annotations

from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.ray.specs.entrypoint import compute_specs


class TestComputeSpecs:
    def test_launches_the_controller_then_routers_then_the_session_server_then_every_engine(self, tmp_path):
        """The manager's whole inventory comes from here, so every component must be listed exactly once."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2},
                    {"worker_type": "placeholder", "num_gpus": 4, "num_gpus_per_engine": 4},
                    {"worker_type": "decode", "num_gpus": 8, "num_gpus_per_engine": 4},
                ]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=16, use_session_server=False)

        specs = compute_specs(args)

        assert [spec.name for spec in specs] == [
            "rollout-executor",
            "multi-lora-controller",
            "inference-controller",
            "inference-router-0",
            "session-server",
            "inference-engine-0-0",
            "inference-engine-0-2",
            "trainer-controller-actor",
            "trainer-engine-actor",
        ]

    def test_a_disabled_session_server_is_listed_with_no_cells(self, tmp_path):
        """Disabling the session server must not remove it from the inventory, only empty it."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4, use_session_server=False)

        specs = {spec.name: spec for spec in compute_specs(args)}

        assert specs["session-server"].scheduling.num_cells == 0
        assert specs["inference-engine-0-0"].scheduling.num_cells == 2

    def test_debug_train_only_lists_no_inference_engine(self, tmp_path):
        """--debug-train-only must instantiate no sglang engine, since its bundles are the trainer's own gpus."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=8,
            use_session_server=False,
            colocate=True,
            debug_train_only=True,
        )

        specs = compute_specs(args)

        assert [spec.name for spec in specs if spec.name.startswith("inference-engine")] == []


class TestDeployComponentFiltering:
    @staticmethod
    def _args(tmp_path, **overrides):
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        return make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=4,
            use_session_server=True,
            use_critic=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=2,
            **overrides,
        )

    def test_a_trainer_deployment_holds_the_trainer_controller_and_its_ranks_only(self, tmp_path):
        """A trainer release that also installed engines would double the run's gpu bill."""
        specs = compute_specs(self._args(tmp_path, deploy_component="trainer"))

        assert [spec.name for spec in specs] == [
            "trainer-controller-actor",
            "trainer-controller-critic",
            "trainer-engine-actor",
            "trainer-engine-critic",
        ]

    def test_naming_one_trainer_instance_installs_that_instance_alone(self, tmp_path):
        """A release per trainer instance is the point of the instance selector, so the other role must drop out."""
        specs = compute_specs(self._args(tmp_path, deploy_component="trainer:critic"))

        assert [spec.name for spec in specs] == ["trainer-controller-critic", "trainer-engine-critic"]

    def test_an_inference_deployment_holds_the_controller_its_routers_and_its_engines(self, tmp_path):
        """The router belongs to the engines it fronts, so it can only be installed with them."""
        specs = compute_specs(self._args(tmp_path, deploy_component="inference"))

        assert [spec.name for spec in specs] == [
            "inference-controller",
            "inference-router-0",
            "inference-engine-0-0",
        ]

    def test_the_primary_deployment_holds_everything_the_two_sides_do_not(self, tmp_path):
        """primary is defined by subtraction, so anything unclaimed has to land here rather than nowhere."""
        specs = compute_specs(self._args(tmp_path, deploy_component="primary"))

        assert [spec.name for spec in specs] == [
            "rollout-executor",
            "multi-lora-controller",
            "session-server",
        ]

    def test_the_three_subsets_partition_the_whole_run(self, tmp_path):
        """A worker in neither subset would never be deployed, and one in two would be deployed twice."""
        whole = [spec.name for spec in compute_specs(self._args(tmp_path, deploy_component="all"))]
        parts = [
            spec.name
            for component in ("primary", "trainer", "inference")
            for spec in compute_specs(self._args(tmp_path, deploy_component=component))
        ]

        assert sorted(whole) == sorted(parts)
