from __future__ import annotations

import datetime
from argparse import Namespace

from miles.ray.specs.rollout import ROLLOUT_EXECUTOR_POOL_ID
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.multi_lora import is_multi_lora_enabled
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.types import DeployComponent, DeploySelector, HotRestartComponent

_COMPONENT_POOLS = {HotRestartComponent.ROLLOUT_EXECUTOR: ROLLOUT_EXECUTOR_POOL_ID}


class HotRestartPlan(FrozenStrictBaseModel):
    restart_at: str = ""
    restart_pools: frozenset[str] = frozenset()
    rebuilt_object_names: frozenset[str] = frozenset()
    restarts_orchestration: bool = False


def plan_hot_restart(
    args: Namespace,
    *,
    components: frozenset[HotRestartComponent],
    selector: DeploySelector,
    release: str,
    installed_manifest: Manifest | None = None,
) -> HotRestartPlan:
    if not components:
        return _carry_installed_stamp(installed_manifest, release=release)

    _assert_preconditions(args, selector=selector)

    restart_pools = frozenset(_COMPONENT_POOLS[component] for component in components if component in _COMPONENT_POOLS)
    restarts_orchestration = HotRestartComponent.ORCHESTRATION in components
    rebuilt = {naming.component_name(release, pool) for pool in restart_pools}
    if restarts_orchestration:
        rebuilt.add(naming.component_name(release, naming.ORCHESTRATOR_COMPONENT))

    return HotRestartPlan(
        restart_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="microseconds"),
        restart_pools=restart_pools,
        rebuilt_object_names=frozenset(rebuilt),
        restarts_orchestration=restarts_orchestration,
    )


def _carry_installed_stamp(installed_manifest: Manifest | None, *, release: str) -> HotRestartPlan:
    """Render the stamp a previous hot restart left, so an ordinary relaunch is a zero diff and rolls no pod."""
    if installed_manifest is None:
        return HotRestartPlan()

    stamp = installed_manifest.restart_at(
        preferred_object_name=naming.component_name(release, naming.ORCHESTRATOR_COMPONENT)
    )
    if stamp is None:
        return HotRestartPlan()

    return HotRestartPlan(restart_at=stamp, restart_pools=frozenset(_COMPONENT_POOLS.values()))


def _assert_preconditions(args: Namespace, *, selector: DeploySelector) -> None:
    assert selector.component is DeployComponent.PRIMARY, (
        f"--hot-restart replaces the orchestration script while the trainers and the inference side keep running, "
        f"so they have to be releases of their own; launch it with --deploy-component "
        f"{DeployComponent.PRIMARY.value}, not {selector.value}"
    )
    assert args.inference_controller_addrs is not None, (
        "--hot-restart takes over an inference controller that outlives the orchestration script, so it has to be "
        "a release of its own named by --inference-controller-addrs rather than built by this launch"
    )
    assert args.debug_train_only or args.inference_router_addrs is not None, (
        "--hot-restart keeps the routers of the run serving, so this launch must not deploy any of its own; name "
        "the running ones with --inference-router-addrs"
    )
    assert args.trainer_controller_addrs is not None, (
        "--hot-restart resumes trainers that outlive the orchestration script, so they have to be named by "
        "--trainer-controller-addrs rather than built by this launch"
    )
    assert not args.indep_dp, (
        "--hot-restart is not supported with --indep-dp: the independent-DP quorum and its store are rebuilt by "
        "the trainer controller's init, and nothing yet re-derives that state for a new orchestration script"
    )
    assert not is_multi_lora_enabled(args), (
        "--hot-restart is not supported with --multi-lora: a reload hides the adapters in the megatron slots while "
        "it loads the base checkpoint, so they physically survive it while the bookkeeping that owns their slots "
        "does not, and the next reconcile would load every adapter into an occupied slot"
    )
