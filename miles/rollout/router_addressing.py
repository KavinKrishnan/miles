from argparse import Namespace

from miles.utils.types import Sample


def compute_router_url(args: Namespace, *, model_id: str | None = None, endpoint: str = "") -> str:
    routers = args.sglang_model_routers
    assert routers is not None, (
        "sglang_model_routers is not set: the routers are resolved before any rollout runs; "
        "an unresolved map means a misconfigured run"
    )

    assert model_id is not None or len(routers) == 1, (
        f"this run serves {sorted(routers)}, so a request that names no model has no one router to go to; name "
        f"the model, or the request would generate one model's samples on another model's engines"
    )

    assert model_id is None or model_id in routers or len(routers) == 1, (
        f"this run serves {sorted(routers)}, so model {model_id!r} has no router to fall back on; falling back "
        f"would generate one model's samples on another model's engines"
    )

    if model_id is not None and model_id in routers:
        host, port = routers[model_id]
    else:
        host, port = next(iter(routers.values()))

    return f"http://{host}:{port}{endpoint}"


def compute_sample_router_url(args: Namespace, sample: Sample, *, endpoint: str = "") -> str:
    return compute_router_url(args, model_id=sample.trainer_model_id, endpoint=endpoint)


def compute_any_router_url(args: Namespace, *, endpoint: str = "") -> str:
    routers = args.sglang_model_routers
    assert routers, (
        "sglang_model_routers is not set: the routers are resolved before any rollout runs; "
        "an unresolved map means a misconfigured run"
    )

    host, port = min(routers.items())[1]
    return f"http://{host}:{port}{endpoint}"
