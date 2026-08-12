from types import SimpleNamespace

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher.hot_restart import plan_hot_restart
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    RESTART_AT_ANNOTATION,
    Manifest,
)
from miles.utils.workers.types import DeployComponent, DeploySelector, HotRestartComponent

_RELEASE = "miles-run-260101-000000-000-primary"
_BOTH = frozenset(HotRestartComponent)


def _args(**overrides) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "inference_controller_addrs": ["http://host:8000"],
            "inference_router_addrs": ["actor=http://router:8000"],
            "trainer_controller_addrs": {"actor": "host:8000"},
            "indep_dp": False,
            "debug_train_only": False,
            "multi_lora": False,
            **overrides,
        }
    )


def _primary() -> DeploySelector:
    return DeploySelector(component=DeployComponent.PRIMARY)


class TestPlanHotRestart:
    def test_no_components_plans_nothing(self):
        """An ordinary launch must not stamp a restart annotation onto a live run."""
        plan = plan_hot_restart(_args(), components=frozenset(), selector=_primary(), release=_RELEASE)

        assert plan.restart_at == ""
        assert plan.restart_pools == frozenset()
        assert plan.rebuilt_object_names == frozenset()
        assert plan.restarts_orchestration is False

    def test_a_restart_stamps_a_timestamp(self):
        """The pod template only changes, and the StatefulSet only rolls, if this value moves."""
        plan = plan_hot_restart(_args(), components=_BOTH, selector=_primary(), release=_RELEASE)

        assert plan.restart_at != ""

    def test_the_rollout_executor_pool_is_named_for_the_annotation(self):
        """The annotation is per pool, so the plan has to say which pool carries it."""
        plan = plan_hot_restart(_args(), components=_BOTH, selector=_primary(), release=_RELEASE)

        assert plan.restart_pools == {"rollout-executor"}

    def test_both_replaced_objects_are_exempted_from_the_diff_gate(self):
        """Changing the orchestration args is the point of the feature, so their objects may differ."""
        plan = plan_hot_restart(_args(), components=_BOTH, selector=_primary(), release=_RELEASE)

        assert plan.rebuilt_object_names == {
            f"{_RELEASE}-orchestrator",
            f"{_RELEASE}-rollout-executor",
        }

    def test_the_orchestrator_is_always_replaced_alongside_the_executor(self):
        """The two components are hot restarted together, so the plan never widens the gate over one alone."""
        plan = plan_hot_restart(_args(), components=_BOTH, selector=_primary(), release=_RELEASE)

        assert plan.restarts_orchestration is True

    @pytest.mark.parametrize("component", [DeployComponent.ALL, DeployComponent.TRAINER, DeployComponent.INFERENCE])
    def test_only_the_primary_release_may_hot_restart(self, component: DeployComponent):
        """Under any other component this release also carries the trainers, so restarting it restarts them."""
        selector = DeploySelector(component=component, instance="a" if component.takes_instance() else None)

        with pytest.raises(AssertionError, match="releases of their own"):
            plan_hot_restart(_args(), components=_BOTH, selector=selector, release=_RELEASE)

    def test_a_statically_addressed_inference_controller_is_required(self):
        """A controller this launch would build is a controller it would also restart, losing its registrations."""
        with pytest.raises(AssertionError, match="inference-controller-addrs"):
            plan_hot_restart(
                _args(inference_controller_addrs=None),
                components=_BOTH,
                selector=_primary(),
                release=_RELEASE,
            )

    def test_statically_addressed_trainers_are_required(self):
        """A trainer this launch would build is a trainer it would also restart."""
        with pytest.raises(AssertionError, match="trainer-controller-addrs"):
            plan_hot_restart(
                _args(trainer_controller_addrs=None), components=_BOTH, selector=_primary(), release=_RELEASE
            )

    def test_indep_dp_is_mutually_exclusive_with_a_hot_restart(self):
        """The indep-DP quorum and store are built by the trainer controller's one-time init and never re-derived."""
        with pytest.raises(AssertionError, match="indep-dp"):
            plan_hot_restart(_args(indep_dp=True), components=_BOTH, selector=_primary(), release=_RELEASE)

    def test_the_preconditions_are_not_checked_without_a_hot_restart(self):
        """Every ordinary launch calls this, and an ordinary launch needs none of those flags."""
        plan = plan_hot_restart(
            _args(inference_controller_addrs=None, trainer_controller_addrs=None, indep_dp=True),
            components=frozenset(),
            selector=DeploySelector(component=DeployComponent.ALL),
            release=_RELEASE,
        )

        assert plan.restart_at == ""


_STAMP = "2026-08-12T09:00:00+00:00"


def _installed_manifest(*, stamped: bool) -> Manifest:
    annotations = {RESTART_AT_ANNOTATION: _STAMP} if stamped else {}
    return Manifest(
        objects=[
            dict(
                kind="StatefulSet",
                metadata=dict(name=f"{_RELEASE}-orchestrator"),
                spec=dict(template=dict(metadata=dict(annotations=annotations))),
            )
        ]
    )


class TestTheStampAnOrdinaryRelaunchRenders:
    def test_a_relaunch_of_a_never_hot_restarted_run_stamps_nothing(self):
        """The annotation must not appear out of nowhere, or the first relaunch would roll the pods."""
        plan = plan_hot_restart(
            _args(),
            components=frozenset(),
            selector=_primary(),
            release=_RELEASE,
            installed_manifest=_installed_manifest(stamped=False),
        )

        assert plan.restart_at == ""
        assert plan.restart_pools == frozenset()

    def test_a_relaunch_after_a_hot_restart_renders_the_stamp_it_finds(self):
        """Dropping it would make the pod template differ, so the diff gate refuses an ordinary relaunch forever."""
        plan = plan_hot_restart(
            _args(),
            components=frozenset(),
            selector=_primary(),
            release=_RELEASE,
            installed_manifest=_installed_manifest(stamped=True),
        )

        assert plan.restart_at == _STAMP
        assert plan.restart_pools == {"rollout-executor"}

    def test_carrying_the_stamp_forward_replaces_no_object(self):
        """Nothing is being restarted, so the relaunch gate must stay as strict as it is for any other run."""
        plan = plan_hot_restart(
            _args(),
            components=frozenset(),
            selector=_primary(),
            release=_RELEASE,
            installed_manifest=_installed_manifest(stamped=True),
        )

        assert plan.rebuilt_object_names == frozenset()
        assert plan.restarts_orchestration is False

    def test_a_hot_restart_stamps_a_value_of_its_own_over_the_installed_one(self):
        """The pod only rolls because the value moved, so a carried-forward stamp would restart nothing."""
        plan = plan_hot_restart(
            _args(),
            components=_BOTH,
            selector=_primary(),
            release=_RELEASE,
            installed_manifest=_installed_manifest(stamped=True),
        )

        assert plan.restart_at != _STAMP

    def test_a_first_install_has_no_manifest_to_read(self):
        """Every launch calls this, including the one that installs the release."""
        plan = plan_hot_restart(
            _args(), components=frozenset(), selector=_primary(), release=_RELEASE, installed_manifest=None
        )

        assert plan.restart_at == ""


class TestTheRemainingPreconditions:
    def test_statically_addressed_routers_are_required(self):
        """A router this launch deploys is a router it replaces, and the engines lose their front door."""
        with pytest.raises(AssertionError, match="inference-router-addrs"):
            plan_hot_restart(
                _args(inference_router_addrs=None), components=_BOTH, selector=_primary(), release=_RELEASE
            )

    def test_a_run_without_any_inference_side_needs_no_routers(self):
        """--debug-train-only deploys no engine and no router, so demanding their addresses would be nonsense."""
        plan = plan_hot_restart(
            _args(inference_router_addrs=None, debug_train_only=True),
            components=_BOTH,
            selector=_primary(),
            release=_RELEASE,
        )

        assert plan.restart_at != ""

    def test_multi_lora_is_mutually_exclusive_with_a_hot_restart(self):
        """A reload resets the slot bookkeeping while the adapter parameters survive it in the megatron slots."""
        with pytest.raises(AssertionError, match="multi-lora"):
            plan_hot_restart(_args(multi_lora=True), components=_BOTH, selector=_primary(), release=_RELEASE)
