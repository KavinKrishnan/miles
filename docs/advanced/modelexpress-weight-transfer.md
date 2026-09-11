---
title: ModelExpress Weight Transfer
description: Remote full-model refit from actor to rollout through per-worker ModelExpress/NIXL clients.
---
miles supports weight transfer through [ModelExpress](https://github.com/ai-dynamo/modelexpress)
via `--update-weight-transfer-mode modelexpress`. Each rollout engine pulls the slices its
own rank needs from per-worker clients over NIXL RDMA, rather than receiving a broadcast.

This integration requires compatible Miles and SGLang endpoint bindings and a
ModelExpress package/server with the `modelexpress_rl` API. The default publisher
factory is `modelexpress_rl.integrations.miles.create_miles_publisher`.
The earlier `modelexpress.integrations.miles` prototype uses a different lifecycle
and is not compatible with this version.

## Usage

```
--update-weight-transfer-mode modelexpress
--modelexpress-publisher-adapter <import.path.to.factory>
```

The adapter is an import path to a zero-argument factory returning a trainer-side
publisher. Keeping the transport behind a factory means it is not wired into the training
loop, and swapping it does not touch miles.

Validate with `--check-weight-update-equal`. This matters more here than for a collective
transport: a misconfigured RDMA fabric does not raise, it delivers wrong bytes, and
generation continues on them without complaint.

## How it works

`broadcast` sends the same weights to every rollout rank over NCCL. `p2p` has each
training rank write the shards its target needs directly into remote memory.
`modelexpress` inverts the direction: the receiver *pulls*.

Each rollout rank knows its own parallelism layout, works out which byte ranges of which
publisher it needs, and reads exactly those. The publisher never needs to know the
engine's geometry, which is what lets the two sides disagree about parallelism -- a
trainer at EP=2 can feed an engine at EP=1 with the expert dimension rearranged in
flight.

Miles binds each trainer's stable Megatron storage once, gathers its source slot,
and creates an exact MX weight version. It pauses the selected rollout fleet,
publishes all source shards, marks the version READY, then sends its opaque
`version_id` together with the monotonically increasing `target_training_step`.
Each SGLang rank stages, verifies coverage, and installs through the shared
`ModelExpressGeneratorClient`. Miles requires every selected rank to acknowledge
the exact version before ending the engine update and resuming generation.

READY describes source availability. Retirement releases trainer resources; it
does not activate the rollout fleet. Miles retires versions and releases source
shards on error paths too. Active leases prevent premature source reuse.

Staging does not change live weights. Installation copies in place, so a partial
copy failure can leave a mixed version and has no rollback guarantee. Failed
workers and rounds remain fenced and require restart. Rank-zero lifecycle errors
are shared with the other trainer ranks, preventing them from continuing training
while the rollout fleet is incomplete.

## What resharding does and does not save

Pull-mode resharding removes the layout mismatch between trainer and engine, and where
the engine is sharded it reduces the bytes any single rank pulls -- at inference TP=2,
per-rank installed bytes roughly halve.

It does not reduce total bytes while every engine needs the whole model. Engines
co-located on one host each pull their own full copy, so bytes crossing the wire scale
with the number of engines per host rather than with the model. Fan-out between
co-located engines is the obvious next step and is not implemented.

## Operational notes

**Pin the NIC on rail-isolated fabrics.** Where each RDMA NIC sits on its own subnet, two
ranks only reach each other if they picked the same one. NIC selection is by PCIe
proximity and does not check reachability, so ranks on different hosts can choose
mutually unreachable NICs. This does not fail loudly -- it corrupts weights. Set
`MX_RDMA_NIC_PIN` to an explicit device instead of `auto` in that topology.

**Leave `expandable_segments` off.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
interferes with NIXL memory registration.

## Performance and validation

Measure trainer and rollout on separate hosts, with the same model, parallelism,
NIC and verification settings. Keep cold setup separate from warm reuse and use
the slowest rank for each version. `MX_REFIT_TIMING` records receiver discovery,
capture, planning, registration, wire transfer, transformation and installation;
framework timing additionally includes publication and fleet coordination.

The supported integration surface is Qwen3 dense/MoE BF16. Coverage validation is
unconditional. `--check-weight-update-equal` compares the initial refit with a
native load; it does not prove byte equality for every subsequent trained version.
`MX_RESHARD_PUBLISH_DIGEST` enables additional full-shard integrity checking at a
transfer cost. Quantization, LoRA, speculative models and unsupported storage
layouts are rejected. Prototype results do not establish validation of this API
migration; record correctness and performance for the exact source revisions used.

## Related

- [P2P Weight Transfer](/docs/advanced/p2p-weight-transfer) -- push-mode RDMA alternative

### Benchmark timing

Set MILES_REFIT_TIMING=1 in trainer worker environments to emit
MILES_REFIT_TIMING records. Each actor update records nested preparation,
publication, trainer collective, receiver wait, and fleet activation intervals,
correlated with the exact MX version and training step. Relative monotonic
timestamps support local interval unions; wall-clock anchors alone do not prove
cross-host clock alignment. Normal execution leaves this tracing disabled.
