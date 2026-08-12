import asyncio
import logging
from collections.abc import Awaitable, Callable

from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.inference import INFERENCE_CONTROLLER_POOL_ID, inference_controller_worker_name
from miles.ray.specs.static_addrs import INFERENCE_CONTROLLER_ADDRS_FLAG, TRAINER_CONTROLLER_ADDRS_FLAG
from miles.ray.specs.train import compute_trainer_controller_pool_id, trainer_controller_worker_name
from miles.ray.wiring import get_backend_capability, launch_worker_manager
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.logging_utils import configure_logger
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_spec import RPC_PORT_NAME

logger = logging.getLogger(__name__)


def run_deployment(args, *, run_orchestration_script: Callable[[object], Awaitable[None]]) -> None:
    if DeployComponent(args.deploy_component).deploys_orchestration_script():
        asyncio.run(run_orchestration_script(args))
        return

    asyncio.run(_serve_deployed_workers(args))


async def _serve_deployed_workers(args) -> None:
    configure_logger(args, source=SimpleProcessIdentity(component="main"))
    component = DeployComponent(args.deploy_component)

    _worker_manager = launch_worker_manager(args)
    logger.info(
        f"Deployed the {component.value} workers of this run: "
        f"{[spec.name for spec in compute_specs(args)]}. "
        f"{await _describe_controller_addrs(args, component=component)}"
    )
    logger.info(
        "This deployment carries no orchestration script, so it has no training to finish and stays up until it is "
        "uninstalled"
    )

    await asyncio.Event().wait()


async def _describe_controller_addrs(args, *, component: DeployComponent) -> str:
    capability = get_backend_capability(args)

    if component is DeployComponent.INFERENCE:
        addr = await _rpc_addr(
            capability, pool_id=INFERENCE_CONTROLLER_POOL_ID, worker_name=inference_controller_worker_name()
        )
        return f"Reach it with {INFERENCE_CONTROLLER_ADDRS_FLAG} {addr}"

    roles = ["actor", *(["critic"] if args.use_critic else [])]
    addrs = await asyncio.gather(
        *[
            _rpc_addr(
                capability,
                pool_id=compute_trainer_controller_pool_id(role),
                worker_name=trainer_controller_worker_name(role),
            )
            for role in roles
        ]
    )
    entries = [f"{role}={addr}" for role, addr in zip(roles, addrs, strict=True)]
    return f"Reach it with {TRAINER_CONTROLLER_ADDRS_FLAG} {' '.join(entries)}"


async def _rpc_addr(capability: BackendCapability, *, pool_id: str, worker_name: str) -> str:
    addrs = await capability.static_worker_provider(pool_id=pool_id).get_addrs(worker_name)
    return f"{addrs[RPC_PORT_NAME].host}:{addrs[RPC_PORT_NAME].port}"
