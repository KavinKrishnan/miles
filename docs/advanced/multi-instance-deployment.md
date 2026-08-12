---
title: Deploying Several Trainers and Engine Pools
description: Install one release per trainer of a run, and let engine-only deployments register their cells into the run's one inference controller.
---
[Split deployment](/advanced/split-deployment) installs the trainer, the inference side and the
orchestration script as releases of their own. Multi instance deployment goes one step further: a
component that comes in instances - one trainer per policy, engine pools in several datacenters - is
installed one release per instance, and one orchestration script drives all of them.

<Warning>

**Status.** Under active development. The shape below is verified by hand on kubernetes rather than
in CI, and it is not yet a hot restart.

</Warning>

## The deployment shape

- One `execute_train` invocation deploys one instance, and installs one helm release for it.
- `--deploy-component` names the instance after the component, as `trainer:<role>` or
  `inference:<name>`. A trainer's role is its model id in a multi policy run and `actor` in a single
  policy one, so a single policy split deployment is `trainer:actor`. The launcher passes it down to
  the pods itself; set it as
  `MILES_SCRIPT_DEPLOY_COMPONENT` plus `MILES_SCRIPT_DEPLOY_INSTANCE` (or
  `ExecuteTrainConfig.deploy_component` and `.deploy_instance`).
- The instance is baked into the release name, sanitized to what kubernetes accepts:
  `miles-run-<run id>-trainer-policy-a`. Two trainer releases of one run therefore never overwrite
  each other's objects, hostnames or labels.

| `--deploy-component` | What it deploys |
| --- | --- |
| `all` (default) | everything, as one deployment |
| `primary` | the rollout executor, the orchestration script, the api server, the session servers |
| `trainer` | every trainer instance of the run, in one release |
| `inference` | the one inference controller, its own engines and its per-model routers |
| `trainer:<role>` | only that trainer's controller and its megatron ranks (`actor` with one policy) |
| `inference:<name>` | engines alone, plus one reporter that registers them into the run's controller |

- A run has **exactly one** `InferenceController` and one router per model, always. `inference:<name>`
  does not deploy a second controller and no router of its own: it deploys engines and a
  `RegistrationReporter` that reports those engines to the controller of the `inference` release.
- Single cluster and cross cluster are the same shape: a release addresses another release only
  through the addresses it is given, so where the releases run does not change anything.

A two policy, two datacenter run is five launches:

```bash
MILES_SCRIPT_DEPLOY_COMPONENT=trainer MILES_SCRIPT_DEPLOY_INSTANCE=policy_a python scripts/run_qwen3_4b.py train
```

```bash
MILES_SCRIPT_DEPLOY_COMPONENT=trainer MILES_SCRIPT_DEPLOY_INSTANCE=policy_b python scripts/run_qwen3_4b.py train
```

```bash
MILES_SCRIPT_DEPLOY_COMPONENT=inference python scripts/run_qwen3_4b.py train
```

```bash
MILES_SCRIPT_DEPLOY_COMPONENT=inference MILES_SCRIPT_DEPLOY_INSTANCE=east python scripts/run_qwen3_4b.py train
```

```bash
MILES_SCRIPT_DEPLOY_COMPONENT=primary python scripts/run_qwen3_4b.py train
```

## Identity rules

- **A trainer is identified by its model id.** One `TrainerController` per model id, and every name
  it owns derives from it:
    - `--trainer-controller-addrs <model id>=host:port`, one entry per trainer;
    - pool ids `trainer-controller-<model id>` and `trainer-engine-<model id>`;
    - `--deploy-component trainer:<model id>`.
- In a single policy run that identity is spelled `actor`, and nothing about such a run changes: the
  addresses, the pool ids and the instance name are all `actor`, exactly as before. A critic keeps
  its own `critic` identity next to it.
- **An engine-only deployment is identified by its instance name**, taken from
  `--deploy-component inference:<name>`. That name is the reporter id and the prefix of every pool id
  it reports, so two deployments can run the same pools without colliding: the engines of `east`
  arrive as `east-inference-engine-0-0-<i>`.
- An engine is never identified by its datacenter anywhere in the controller. It is a cell like any
  other, and everything datacenter specific about it - its model, its worker type, its addresses -
  travels inside `CellInfo`.

## `CompositeTrainerController`

`CompositeTrainerController` is a drop-in replacement for `TrainerController`: API-identical, with an
optional `model_id` on every method.

- Membership is static, from `--trainer-controller-addrs` when they are given and from this release's
  own pools otherwise, so it runs under a single deployment just as well as under one release per
  trainer.
- `wait_ready()` waits for every trainer of the run before the first train step.
- `model_id` picks the trainer; `None` is allowed only when the run trains one policy.
- It is a plain object held by the orchestration script (`train_multi_policy.py`), not a worker. It is
  constructed by explicit injection of its handles, so it can become a worker later without changing
  its callers.
- There is no composite inference controller. Engines of every deployment reconcile into the one
  `InferenceController`, exactly as locally launched engines do.

## The registration protocol

An engine-only deployment announces its cells; nothing discovers them.

- The endpoint is an rpc method of the `InferenceController` itself,
  `apply_registration_snapshot(snapshot)`. The registry therefore lives in the controller's process,
  and restarting the orchestration script does not touch it.
- The reporter dials the controller through `--inference-controller-addrs`, the same static address
  the primary release uses. There is no separate registration endpoint flag.
- Every snapshot carries `--registration-token`. Both sides read the same value; a snapshot presenting
  anything else is refused. The flag is required, not optional: a launch that registers engines or
  waits for reporters without one is rejected, because the endpoint is reachable from outside the
  deployment.

**A snapshot is a whole replacement, not a delta.**

- The reporter sends the complete set of cells it observes. The controller validates and parses it
  outside every lock, then takes one critical section that covers the sequence check, the membership
  diff, the digest and the commit together, so two snapshots of one reporter can never interleave into
  a mixture of both.
- Reconciling the cells that changed happens **off the request path**, on a background worker draining
  a queue in commit order. A weight update holding the controller's context lock for minutes therefore
  delays the reconcile, never the reporter's request.
- A cell missing from a snapshot has left that deployment. A changed `workers_hash` is a restarted
  engine, and is removed and added again. Any change to a cell counts, addresses and gpu ids included,
  not only its `CellInfo`.
- Snapshots are idempotent: resending the same one reconciles nothing. A cell whose reconcile failed is
  rolled back and the reporter's digest is dropped, so the next snapshot brings it in again - and it is
  brought in as a real add, because the provider also remembers that the cell must be announced to the
  run again even if the next snapshot repeats the hash the failed one replaced. The rollback covers both
  sides of the reconcile: a cell whose engine did not initialize never entered its `RolloutServer` at
  all, because a half added cell belongs to no provider, matches no health check sweep, and would turn
  every later snapshot into a no-op on an engine that never serves.
- **Bringing a cell up happens outside the context lock.** Reconciling one observation is three steps:
  read under the lock what the run currently holds for that cell id, dial the cell's launch gate with
  the lock released, then take the lock again only to commit the finished cell into its `RolloutServer`.
  A gate that never answers therefore delays that one cell, not the tick that sweeps unreachable cells,
  not a weight update, and not the run's teardown. Local and registered cells take the same path.
- Observations of one cell id are serialized and versioned: a second observation of a cell that is still
  starting up supersedes the first, so the older bring-up is torn down at its commit instead of being
  installed, and a removal that arrives mid bring-up wins the same way. A cell that finishes starting up
  after its server was disposed is torn down rather than installed.
- **One send never outlives its period.** The three attempts of one `send_once` share a budget of one
  snapshot period rather than each getting one, so a slow controller can never stretch a period past the
  staleness threshold that period defines. A timed out attempt costs a third of the budget, and
  resending is free even if the slow one did land, because a snapshot is a whole replacement. A period
  that lands nothing at all is logged as an error.
- The controller's own patience is bounded the same way. A registered cell's launch gate is given two
  minutes rather than the half hour a locally launched pod gets, because the pod of a registered engine
  already exists by the time its reporter announces it. A gate that stays silent fails that cell's
  reconcile, both sides roll back, and the next snapshot brings it in again.

| Mechanism | What it is for |
| --- | --- |
| `(epoch, sequence)` per reporter | inside one reporter process the sequence only grows, so a request that crosses the wan out of order cannot resurrect dead cells; a restarted reporter draws a new random epoch, and a snapshot of an epoch the run has not seen is taken unconditionally with its sequence and digest reset, while a snapshot of an epoch the run has already replaced is dropped rather than rolling the live incarnation back |
| `digest` short circuit | an unchanged deployment sends a heartbeat without its cells, and the run skips parsing them; a heartbeat the run refuses is followed by the whole snapshot in the same period |
| `HasSynced` gating | the reporter never sends before its first full look at its own deployment, so a cold start can never send an empty snapshot that wipes the datacenter |
| watch-triggered snapshot + ~1s debounce | an engine coming or going is reported at once, and a rolling pool is coalesced into one snapshot |
| 15s ± jitter period | level-triggered backstop, staggered so reporters do not arrive as one storm |

- There is no lease and no TTL. An epoch change is a new reporter incarnation, not an expiry; when one
  arrives sooner than a whole deployment could be rebuilt, the run logs an error naming the reporter,
  because two deployments sharing one `inference:<name>` is the only other way to produce it.
- **A cell the run refuses is refused alone.** A cell whose id or worker names do not parse, that names
  a pool it does not belong to, carries a worker of another cell, has no worker at all, is carried
  twice by one snapshot, or declares a `prefill` or `decode` role is excluded with an error naming it
  and the reason, and counted in the ack so the reporter logs it too. The same per-cell refusal covers
  the metadata contract: the run builds each cell's `ServerCellMetadata` from `meta` while it parses the
  snapshot, so a cell whose `meta` is missing a field, carries one the run cannot read, or names a model
  this run does not serve is excluded by name instead of raising inside the reconcile and putting the
  reporter into a permanent retry loop that never says which cell is at fault. The rest of the snapshot is
  applied, and the digest is withheld so the whole snapshot is asked for again. PD-role engines are
  refused a second time on the reporter side, and that refusal takes the reporter's process down -
  whether the offending engine is there at startup or appears later - so the deployment fails instead
  of running on as a pod that looks healthy while registering nothing.
- **Reporter staleness never removes cells.** A reporter that goes quiet is a log line, written by the
  controller's own tick once the last snapshot is older than three periods. Disposing the reporter waits
  for its thread, so a deployment being torn down stops announcing its cells instead of sending one more
  snapshot on the way out, and a reporter that stops because it was asked to leaves the deployment's
  exit code alone. Liveness stays with the
  controller's per-cell probe: the same tick asks every `RolloutServer` to invalidate the cells whose
  health checks fail, local and registered alike, and an invalidated registered cell leaves the run at
  once and comes back with the next snapshot. There is no registration TTL.
- **A restarted controller is dialed again, not given up on.** A reporter's handle pins the
  controller's boot uuid, so the first call after the controller's process is replaced fails with
  `ServerRestartedError` rather than silently talking to a stranger. The reporter then builds a new
  handle on the same address, pinning the new incarnation, drops its acknowledged digest and sends the
  whole snapshot at once - a heartbeat would name a digest the fresh controller never held. Its epoch
  does not change, because the reporter did not restart. The controller side needs nothing: its registry
  died with the old process, and the resent snapshot is a first snapshot to it, counted by the startup
  barrier like any other. The pin is only rebaselined across incarnations; inside one incarnation it
  holds, so a mid-flight restart is never mistaken for a healthy call.
- `--expected-registration-reporters N` tells the controller how many engine-only deployments to
  expect. Each snapshot declares how many cells of each model that deployment will bring, and the
  startup barrier waits for the sum. Without it, a run whose remote engines never arrive would start
  silently against its local engines alone.

### Address translation

- A reporter sees its engines under addresses of its own cluster, which usually mean nothing in the
  datacenter that dials them.
- `--registration-external-hosts <host>=<external host> ...` rewrites the host of every address it
  reports. Ports are kept. An address without an entry is reported unchanged.
- The controller talks to a registered engine directly, so those addresses have to be reachable -
  through a gateway, a `Service`, or whatever the datacenter exposes.

## Routers

- One router per model, always, deployed with the `inference` release. Engines of every deployment
  enter that router.
- `pool_id` stays unique per deployment; the model grouping rides `meta.model_id`, which is shared
  across deployments. Merging pool ids would collide cell ids and is refused.
- The rollout data plane addresses routers per model, through `args.sglang_model_routers` and
  `Sample.trainer_model_id`. There is no single global router address written back onto the arguments,
  because with several models there is no one value to write.
- Asking for a router without naming a model is allowed only while the run serves exactly one, so a
  sample that lost its `trainer_model_id` fails loud instead of generating on another policy's engines.
- Custom rollout functions must therefore ask for the router they need:

```python
from miles.rollout.router_addressing import compute_router_url, compute_sample_router_url
```

- `compute_router_url(args, model_id=..., endpoint=...)` for a named model,
  `compute_sample_router_url(args, sample, endpoint=...)` for the model a sample belongs to, or the
  documented `get_model_url(args, model_name, endpoint)` which wraps the former.

## Weight versions and artifact namespacing

- Each policy keeps its own integer weight version space, published per model id by
  `set_weight_version(version, trainer_model_id=...)`. A version number is meaningless on its own.
- A sample carries both the version and its `trainer_model_id`, and staleness checks read the two
  together, so one policy's version can never be compared against another's.
- Everything a trainer writes is namespaced by its model id:

| Artifact | Namespacing |
| --- | --- |
| disk-delta weight directories | `<--update-weight-disk-dir>/<model id>/` |
| event logger per-process files | `actor_<model id>_cell<i>_rank<j>` |
| debug rollout dumps | `<rollout id>_<model id>` |

- Without this, two policies write to the same paths under the same run and silently overwrite each
  other's weights, logs and dumps.

## GPU budget

- The placement budget of a launch is the sum over the policy trainers **it deploys**, so a
  `trainer:<model id>` release budgets only its own instance, and an `inference` release budgets no
  trainer at all. The sglang gpu layout reads that same budget, so the two can never disagree.
- Each trainer instance gets its own placement slot offset, rebased per release, so instances of one
  release never overlap and an instance installed on its own starts at offset 0.
- A critic is budgeted as before, next to the policy trainers rather than after them.

## What this is not

<Warning>

**Restrictions.**

- **Non-PD only across deployments.** A registered cell that declares a `prefill` or `decode` worker
  type is refused: pairing a prefill engine in one datacenter with a decode engine in another needs an
  attribute-driven pairing policy in the router, which does not exist yet.
- `--colocate` is rejected with more than one policy - which trainer an engine sits beside would be
  undefined - rejected with a named instance, because colocated trainers and engines share gpus and are
  therefore one deployment unit, and rejected together with `--expected-registration-reporters`,
  because a registered engine holds gpus of its own deployment.
- **Not a hot restart.** A new orchestration script does not reattach to running trainers. The
  registry living inside the inference controller is the seam a hot restart builds on.
- **One router, in one place.** A per-datacenter data plane router is a later, purely data-plane
  increment; the controller managing N routers is no different from it managing one.
- **No per-deployment weighting.** Every engine is a worker of the one router, and the router balances
  over engines. Weighted load balancing is a later milestone.

</Warning>

## The invariant

`InferenceController`, `RolloutServer` and `ServerCell` never branch on which datacenter a cell comes
from. Datacenter specific information reaches them only through `CellInfo.meta`, and the fleet modules
never import the registration layer nor name a reporter at all. The probe and removal path is the same
call for a local and a registered cell: the fleet invalidates a cell in whichever provider announced it,
and what that means is the provider's business - the registration provider drops the cell and reconciles
its removal, a provider that watches its own platform re-observes it instead. A contract test pins this,
and it is the same contract a custom external provider is held to.

One gap is worth naming: the api server's `list_cells`, and therefore the mini ft controller polling it,
still enumerate only the pools of their own release, so a registered cell appears in
`InferenceController.get_cell_statuses()` but is not a cell the mini ft controller can suspend or resume.
Healing a registered engine is its own deployment's job.

## Verifying a multi-release run on kubernetes

A manual runbook, not CI: the gpu e2e tests cover multi policy training inside one deployment, and
nothing yet installs several releases automatically. Every command runs from the workbench of the
namespace, exactly as in [split deployment](/advanced/split-deployment).

1. Install the first policy's trainer.

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && MILES_SCRIPT_RUN_ID=$RUN_ID MILES_SCRIPT_DEPLOY_COMPONENT=trainer \
    MILES_SCRIPT_DEPLOY_INSTANCE=policy_a python scripts/run_qwen3_4b.py train"
```

2. Install the second policy's trainer.

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && MILES_SCRIPT_RUN_ID=$RUN_ID MILES_SCRIPT_DEPLOY_COMPONENT=trainer \
    MILES_SCRIPT_DEPLOY_INSTANCE=policy_b python scripts/run_qwen3_4b.py train"
```

3. Install the inference release, which carries the one controller and the per-model routers. It
   expects one engine-only deployment to register into it.

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && MILES_SCRIPT_RUN_ID=$RUN_ID MILES_SCRIPT_DEPLOY_COMPONENT=inference \
    python scripts/run_qwen3_4b.py train --expected-registration-reporters 1 --registration-token "$TOKEN""
```

4. Read the addresses the inference release prints for itself out of its orchestrator log.

```bash
kubectl -n "$MILES_NS" logs -l "app.kubernetes.io/instance=miles-run-$RUN_ID-inference,app.kubernetes.io/component=orchestrator" --tail=-1 | grep -- --inference-controller-addrs
```

5. Install the engine-only deployment, naming the controller it registers into and how its addresses
   are reachable from there. Its reporter waits until the controller answers, so ordering is free.

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && MILES_SCRIPT_RUN_ID=$RUN_ID MILES_SCRIPT_DEPLOY_COMPONENT=inference \
    MILES_SCRIPT_DEPLOY_INSTANCE=east python scripts/run_qwen3_4b.py train \
    --inference-controller-addrs http://<inference-host>:8000 --registration-token "$TOKEN" \
    --registration-external-hosts 10.0.0.5=engine-east.example"
```

6. Install the primary, naming every trainer and the one inference controller.

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && MILES_SCRIPT_RUN_ID=$RUN_ID MILES_SCRIPT_DEPLOY_COMPONENT=primary \
    python train_multi_policy.py \
    --trainer-controller-addrs policy_a=<policy-a-host>:8000 policy_b=<policy-b-host>:8000 \
    --inference-controller-addrs http://<inference-host>:8000 \
    --inference-router-addrs policy_a=<router-a-host>:8000 policy_b=<router-b-host>:8000"
```

7. Check which cells the controller took in, local and registered together.

```bash
kubectl -n "$MILES_NS" logs -l "app.kubernetes.io/instance=miles-run-$RUN_ID-inference,app.kubernetes.io/component=inference-controller" --tail=-1 | grep "Cell "
```

8. Check the router's worker list: one worker per engine, of every deployment.

```bash
curl -sS "http://$ROUTER_HOST:8000/workers"
```

9. Kill the engine-only release and watch its cells leave through the ordinary removal path, either
   because the next snapshot no longer carries them or because the controller's probe finds them dead.

```bash
python -m miles.utils.external_utils.miles_workbench uninstall -n "$MILES_NS" --release "miles-run-$RUN_ID-inference-east"
```

10. Reinstall it and watch its cells come back within one snapshot period, back in the router's worker
    list.

11. Uninstall each release when done, one release at a time, naming its instance.

```bash
python -m miles.utils.external_utils.miles_workbench stop -n "$MILES_NS" "$RUN_ID" --deploy-component trainer --deploy-instance policy_a
```
