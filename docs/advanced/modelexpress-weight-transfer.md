---
title: ModelExpress Weight Transfer
description: Remote full-model refit from actor to rollout through per-worker ModelExpress/NIXL clients.
---
miles supports weight transfer through [ModelExpress](https://github.com/ai-dynamo/modelexpress)
via `--update-weight-transfer-mode modelexpress`. Each rollout engine pulls the slices its
own rank needs from per-worker clients over NIXL RDMA, rather than receiving a broadcast.

This mode spans three repositories and all of them are required:

| Repository | Branch | Contains |
|---|---|---|
| `radixark/miles` | `kavink/mx-live-refit` | this change: the `modelexpress` transfer mode |
| [`ai-dynamo/modelexpress`](https://github.com/ai-dynamo/modelexpress/tree/kavink/miles-sglang-refit) | `kavink/miles-sglang-refit` | publisher, reshard receiver, Miles adapter |
| [`sgl-project/sglang`](https://github.com/KavinKrishnan/sglang/tree/kavink/modelexpress-live-refit) | `kavink/modelexpress-live-refit` | serving-side refit lifecycle and endpoints |

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

The refit is bracketed rather than a single call. Generation is paused, the transfer
runs, and only once every rank reports the weights installed does the trainer commit the
version. That ordering is what makes a failed refit safe: the previously committed
version stays intact and servable rather than the engine being left holding half of one
version and half of another.

Rank 0 performs the commit, and the other trainer ranks wait on a gloo barrier before
returning. Without the barrier they re-enter framework code while the commit is still in
flight, and the version check races.

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

## Performance

The bottleneck is control discovery, not wire transfer. This is worth stating plainly
because the first measurements suggested the opposite: they were taken with trainer and
rollout on the same host, where the transfer never touched the network and simply
measured NVLink. Once genuinely cross-host, wire time becomes a large minority of a warm
refit and runs near a single NIC's line rate, while discovery dominates the remainder.

Two implications. Same-host figures should not be quoted as transport measurements. And
the optimisation target is discovery cost, not bandwidth.

## Validated

Qwen3-30B-A3B (`Qwen3MoeForCausalLM`), trainer and rollout on separate hosts, verified
with `--check-weight-update-equal`: full parameter coverage, weight equality clean,
automatic replan when a rollout engine is replaced, and the prior committed version
preserved under injected pre-install failures.

## Related

- [P2P Weight Transfer](/docs/advanced/p2p-weight-transfer) -- push-mode RDMA alternative
