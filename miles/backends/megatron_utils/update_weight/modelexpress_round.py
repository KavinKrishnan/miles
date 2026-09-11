"""Miles-owned cohort coordination for exact-version ModelExpress refits."""

from __future__ import annotations

import logging
import uuid

import ray
import torch.distributed as dist

from miles.backends.megatron_utils.update_weight.common import (
    _check_weight_sync_results,
    begin_weight_update,
    end_weight_update,
    weight_update_selector,
)
from miles.backends.megatron_utils.update_weight.modelexpress import ModelExpressPublishRequest
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.refit_timing import refit_span, set_refit_identity, trace_refit

logger = logging.getLogger(__name__)


def _all_ranks(label, operation):
    value, error = None, None
    try:
        with refit_span(label + ".operation"):
            value = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    statuses = [None] * dist.get_world_size()
    with refit_span(label + ".trainer_collective"):
        dist.all_gather_object(statuses, (value, error), group=get_gloo_group())
    failures = [(rank, item[1]) for rank, item in enumerate(statuses) if item[1]]
    if failures:
        raise RuntimeError(f"ModelExpress {label} failed closed on trainer ranks: {failures}")
    return [item[0] for item in statuses]


def _root(label, operation):
    return _all_ranks(label, lambda: operation() if dist.get_rank() == 0 else None)[0]


def _prepare(update):
    tensors, units = update._published_tensors_and_units()
    slot = update.publisher.prepare(tensors) if update._publishes else None
    return tensors, units, slot


def _pause(update):
    mode = update.args.pause_generation_mode
    with refit_span("pause_generation"):
        ray.get([engine.pause_generation.remote(mode=mode) for engine in update.rollout_engines])
    if mode != "in_place":
        with refit_span("cache_flush"):
            ray.get([engine.flush_cache.remote() for engine in update.rollout_engines])
    with refit_span("begin_weight_update"):
        begin_weight_update(update.rollout_engines, weight_update_selector(update.args))


def _receive(update, version_id):
    timeout = getattr(update.args, "modelexpress_update_timeout", 600.0)
    refs = [
        engine.update_weights_from_modelexpress.remote(
            payload=update._receiver_payload(version_id),
            timeout=timeout,
        )
        for engine in update.rollout_engines
    ]
    with refit_span("receiver_rpc_wait"):
        results = ray.get(refs, timeout=timeout + 30)
    _check_weight_sync_results(results, is_lora=False)
    for result, selected in zip(results, update.rollout_workers, strict=True):
        workers = result.get("workers", [])
        expected = selected.get("gpu_count")
        if not workers or (expected is not None and len(workers) != expected):
            raise RuntimeError("ModelExpress receiver did not acknowledge every selected rank")
        if len({worker["rank"] for worker in workers}) != len(workers):
            raise RuntimeError("ModelExpress receiver returned duplicate ranks")
        for worker in workers:
            response = worker["response"]
            if (
                not response.get("success")
                or response.get("receiver_poisoned")
                or response.get("version_id") != version_id
                or response.get("installed_training_step") != update.weight_version
            ):
                raise RuntimeError("ModelExpress receiver acknowledged a different version")


def _activate(update):
    with refit_span("end_weight_update"):
        end_weight_update(update.rollout_engines)
    ray.get(
        [
            engine.update_weight_version.remote(weight_version=str(update.weight_version))
            for engine in update.rollout_engines
        ]
    )
    try:
        with refit_span("continue_generation"):
            ray.get([engine.continue_generation.remote() for engine in update.rollout_engines])
    except Exception:
        # A resume RPC may have succeeded on a subset before another failed.
        ray.get(
            [
                engine.pause_generation.remote(mode=update.args.pause_generation_mode)
                for engine in update.rollout_engines
            ]
        )
        raise


@trace_refit("refit_round")
def run_update(update):
    set_refit_identity(rank=dist.get_rank(), target_training_step=update.weight_version + 1)
    if update.rollout_engines is None:
        raise RuntimeError("ModelExpress rollout engines are not connected")
    if getattr(update, "_refit_failed", False):
        raise RuntimeError("prior ModelExpress round failed; replace the selected fleet before resuming")
    local = []

    def prepare():
        tensors, units, slot = _prepare(update)
        local.extend((tensors, units))
        return slot

    slots = _all_ranks("tensor geometry preparation", prepare)
    source_slots = tuple(slot for slot in slots if slot is not None)
    if len(set(source_slots)) != len(source_slots):
        raise RuntimeError("ModelExpress trainer source slots must be unique")
    update.weight_version += 1
    version_id = _root(
        "version creation",
        lambda: update.publisher.create_version(
            source_slots=source_slots,
            step=update.weight_version,
            update_id=uuid.uuid4().hex,
        ),
    )
    set_refit_identity(version_id=version_id, target_training_step=update.weight_version)
    primary_error = None
    try:
        _root("pause", lambda: _pause(update))
        request = ModelExpressPublishRequest(
            version=version_id,
            training_step=update.weight_version,
            logical_group="model",
            cohort_id=update.cohort_id,
            worker_id=update.worker_id,
            source_geometry=update.source_geometry,
            tensors=local[0],
            atomic_units=local[1],
        )
        _all_ranks("publication", lambda: update.publisher.publish_and_execute(request) if update._publishes else None)
        _root("source readiness", lambda: update.publisher.mark_ready(version_id))
        _root("receiver installation", lambda: _receive(update, version_id))
    except Exception as exc:
        primary_error = exc
        update._refit_failed = True
        raise
    finally:
        try:
            _root("version retirement", lambda: update.publisher.retire(version_id))
            _all_ranks("source release", lambda: update.publisher.release(version_id) if update._publishes else None)
        except Exception:
            update._refit_failed = True
            if primary_error is None:
                raise
            logger.exception("ModelExpress cleanup failed; source storage remains fenced")
    try:
        _root("fleet activation", lambda: _activate(update))
    except Exception:
        update._refit_failed = True
        raise
