---
title: Ray and Kubernetes Backend
description: Run the same training script on Ray, or let Kubernetes schedule every worker of the run.
---
Miles runs a job on one of two cluster backends. Ray is the default and needs no configuration.
Kubernetes installs the run as a helm release, so the cluster schedules every worker. The training
script is the same either way.

<Warning>

**Status.** Under active development: flags, chart values and failure semantics still change. Ray
also runs *inside* Kubernetes, and that is the well-trodden path — take this one only if you want
Miles to create the cluster's objects itself.

</Warning>

## Launch

Run everything from the repository root, with `kubectl` and `helm` on your PATH.

Install a workbench — the long-lived pod you launch from — once per namespace, from your
cluster's [`infra.yaml`](#for-cluster-administrator):

```bash
export MILES_NS="miles-$USER"
python -m miles.utils.external_utils.miles_workbench install -n "$MILES_NS" -f infra.yaml
```

It creates the namespace if missing, checks your rights, installs, and waits for Ready. Runs
launched from the pod inherit those values.

Launch from inside it:

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && python scripts/run_qwen3_4b.py train"
```

## Worker communication

The cluster backend says who *creates* a worker. `--worker-comm-backend` says how the driver *calls*
it, and the two are separate choices.

- `ray` — the driver holds an actor handle and calls the worker as a Ray actor method.
- `rpc` — every worker serves an HTTP server, and the driver calls it through a typed client.
- unset (the default) — the cluster backend picks: `ray` under `--cluster-backend ray`, `rpc` under
  `--cluster-backend kubernetes`, which has no actors to call.

Both modes coexist under the Ray backend, and the plan is to flip the default to `rpc` once it has
run long enough in production; the `ray` path is removed only after that.

Under `--cluster-backend ray --worker-comm-backend rpc`:

- Each served worker is built inside a `ServeActor` — a Ray actor running the RPC server in its own
  process, so RDT and everything else in the Ray ecosystem still sees the worker.
- The RPC port is allocated by the same dynamic port allocator the rest of the Ray path uses.
- If the server ever stops serving, the actor exits, so a worker nobody can call is never reported
  alive.
- SGLang engines and routers are launched by `CommandActor` and are unaffected.

### What a worker's methods have to look like

The RPC layer builds its client from the worker class, so every public method of a served worker is
part of the wire contract:

- Annotate every parameter and the return type. An unannotated public method makes the whole pool
  unreachable, and the error names the method.
- Use types the wire can carry: pydantic models, dataclasses, enums, and the plain built-ins.
- Make a method private when it never crosses the wire. That is the cheap answer to "what wire type
  does this argument even have".

One exception exists. `Pickled`, from `miles.utils.workers.rpc.common.wire_types`, sends a
parameter as pickle bytes:

- It is applied to exactly one parameter — the argparse `args` namespace a trainer is built from —
  because nothing else reproduces it losslessly, and Megatron reads hundreds of fields off it.
- It is per parameter. Every other argument of the same call stays strictly wire-typed.
- It is temporary. Once the arguments subsystem is split into wire-typed pieces, the hatch is
  removed; grep `MILES_PICKLED_HATCH` to find it.
- `tests/fast/utils/workers/test_pickled_hatch_boundary.py` fails if any other parameter reaches
  for it. It reads the whole annotation, so a union, a container, an import alias, a module-level
  alias and a quoted forward reference are all caught. Prefer `WireNamespace`, which carries a
  namespace as plain json, when the fields are simple enough to survive it.

## Where Ray communication is still allowed

Ray is the launcher, so Ray calls are confined to the launcher's own closure. Everything else must
be reachable over RPC.

| Area | Ray allowed | Why |
| --- | --- | --- |
| `RayWorkerManager`, worker provider, cell operations, `CommandActor`, port allocator | yes | the launcher talking to itself, the same way Kubernetes talks to the kube API |
| Ray object store (`StoreObjectRef`) | yes | a data plane of its own: under `rpc` the reference travels as `ray.cloudpickle` bytes, which pins the object, so `remove` frees it explicitly |
| Dashboard collector, prometheus collector | yes, as known debt | named actors with no Kubernetes form; prometheus is skipped there |
| multi-LoRA controller | no | already an independent worker, called through a worker handle |
| Everything else, including every worker method | no | `tests/fast/utils/workers/test_ray_communication_boundary.py` fails on a new `import ray` |

The check reads every shipped module — `miles/`, `miles_plugins/`, `scripts/`, `examples/`, `tools/` and the
orchestration scripts at the repository root, with only `tests/` out of scope — because the driver is where
a Ray-only `except` clause hides most easily. `import ray`, `import ray.exceptions`, `import ray as r`,
`from ray.… import …` and `importlib.import_module("ray")` all count, and every exemption carries the reason
it belongs to the launcher closure.

`remove` destroys rather than decrements: a freed object is gone for every holder. So only a reference this
process itself materialized from the wire is freed — that is the one whose reference count no longer
protects it. Under `--worker-comm-backend ray` a reference is passed through untouched and `remove` leaves
it to Ray's own reference counting.

## Observability

**Built in**

- The launcher follows every pod's logs, prints status changes and warning events, and ends with
  the run's exit code.
- On a failure, collect the namespace's logs, describes and events into one directory — before
  cleaning up, and the whole namespace, because the explanation is usually the pod next to it:
  `python -m miles.utils.external_utils.miles_workbench collect-diagnosis -n "$MILES_NS" --output-dir ~/artifacts/miles`

**External**

- A run's pods are ordinary pods, so whatever the cluster already runs — a metrics stack, a log
  collector, the platform's own dashboards — sees them with no wiring from Miles.
- Prefer it at scale. The built-in following is meant for watching one run, not hundreds of pods.

## Clean up

```bash
python -m miles.utils.external_utils.miles_workbench stop -n "$MILES_NS" 260811-143000-042
python -m miles.utils.external_utils.miles_workbench uninstall -n "$MILES_NS"
```

`stop` removes the run and frees its GPUs; `uninstall` removes the workbench.

## Folder convention

A run is many pods on many machines, and they share nothing but the storage `infra.yaml` mounts.
A path that is not on it is the most common way a run fails.

- Every path your script names — `/root/models`, `/root/datasets` — has to be on it.
- Copying a file into a pod is pointless: pods come and go, the mount survives.
- To run your own branch instead of the image's copy, name its sub-path under the storage root:
  `infra.paths.repos.miles: alice/miles`. `megatron` and `sglang` work the same way.

## For cluster administrator

Everything above assumes this was done once.

**Install LWS.** Miles deploys its worker pools as
[LeaderWorkerSets](https://github.com/kubernetes-sigs/lws). Install the CRDs and controller, and
grant users rights over them explicitly: LWS ships no aggregation labels, so a namespace `admin`
role does not include them.

**Give each user a namespace.** The namespace is the real boundary, not the Role: anything that
may create workloads can name another ServiceAccount and read its token. Keep privileged accounts
out of it.

**Write one `infra.yaml`.** The same file drives every Miles chart:

```yaml
infra:
  image:
    repository: radixark/miles
    tag: dev
  sharedStorage:
    type: hostPath
    hostPath: /cluster-storage
    mountPath: /cluster-storage
```

`charts/miles-run/values.yaml` shows the full shape, and each chart's `values.schema.json` is the
authoritative field list.
