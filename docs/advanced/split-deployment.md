---
title: Deploying the Trainer and the Inference Side Apart
description: Install a run's trainer, its inference side and everything else as separate deployments that address each other statically.
---
A run is normally one deployment: one launch brings up every worker. `--deploy-component` splits it, so
the trainer, the inference side and the orchestration script are installed by separate launches with
separate lifecycles.

<Warning>

**Status.** Under active development. Splitting a run is what lets one orchestration script drive
several trainers or several inference deployments; this page describes the single-instance split, and
[multi instance deployment](/advanced/multi-instance-deployment) the rest.

</Warning>

## The components

| `--deploy-component` | What it deploys |
| --- | --- |
| `all` (default) | everything, as one deployment |
| `trainer` | the trainer controller and its megatron ranks |
| `inference` | the inference controller, its sglang engines and its routers |
| `primary` | everything else: the rollout executor, the orchestration script, the api server, the mini ft controller, the session servers |

- `primary` is `all` minus `trainer` minus `inference`, so the four values partition the run's workers.
- One launch deploys one component. A split run is one launch per component, all with the same base
  arguments, plus the static addresses that describe the components *another* launch deploys - which only
  the `primary` launch has any of.
- `--colocate` is rejected under a split: colocated trainers and engines share gpus, so they are one
  deployment unit.

## Launching a split run

The part a launch deploys is a property of the launch, not of the training arguments: it names every
object the launch installs, so the launcher has to know it before it renders anything. Set
`MILES_SCRIPT_DEPLOY_COMPONENT` (or `ExecuteTrainConfig.deploy_component`), and the launcher passes
`--deploy-component` down to the pods itself. Naming a *different* component in the training arguments
stops the launch.

Deploy the two sides first, each of which prints the address to reach it by:

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && MILES_SCRIPT_DEPLOY_COMPONENT=trainer python scripts/run_qwen3_4b.py train"
```

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && MILES_SCRIPT_DEPLOY_COMPONENT=inference python scripts/run_qwen3_4b.py train"
```

Then run the training itself, naming what it has to reach:

```bash
python -m miles.utils.external_utils.miles_workbench exec -n "$MILES_NS" -- \
  bash -lc "cd /root/miles && MILES_SCRIPT_DEPLOY_COMPONENT=primary python scripts/run_qwen3_4b.py train \
    --trainer-controller-addrs actor=<trainer-controller-host>:8000 \
    --inference-controller-addrs <inference-controller-host>:8000 \
    --inference-router-addrs <router-host>:8000"
```

## Addresses are given, never discovered

A deployment finds its own workers by the names its own release gives them. Nothing derives another
deployment's names, so everything that crosses a deployment boundary is an argument. A static address
always describes a component *some other* launch deploys: passing one for a component this launch
deploys itself is refused while the arguments are validated, so `--trainer-controller-addrs` is refused
on a `trainer` launch, `--inference-controller-addrs` and `--inference-router-addrs` on an `inference`
launch, and all three on an unsplit `all` launch.

- `--trainer-controller-addrs` — `host:port`, or `<role>=host:port` when the run also trains a critic.
- `--inference-controller-addrs` — `host:port`, one entry per inference deployment. A run whose
  inference deployments come and go names none of them and lets them register themselves instead; see
  [multi instance deployment](/advanced/multi-instance-deployment).
- `--inference-router-addrs` — `host:port`, or `<model>=host:port` per model. The routers live with the
  engines they serve, so the rollout executor and the session servers are given their address too.
- The object store master lives with the orchestration script, so the other deployments have to name it:
  pass the primary deployment's mooncake master address in `--mooncake-store-init-kwargs`, together with
  `--object-store-backend mooncake`. A launch that carries no orchestration script and names no master is
  refused while its arguments are validated, because a store reference is only redeemable inside the
  deployment that created it. The master comes up with the deployment that carries the orchestration
  script, so a trainer or inference deployment installed before it restarts until that master answers.

All three are required when a launch carries the orchestration script without the component it names,
and the launch fails immediately when one is missing. `--debug-train-only` deploys no engines and no
routers, so it excuses `--inference-router-addrs` only; the inference controller is reached either way.

A given address is waited for rather than assumed: the orchestration script waits for the trainer
controller, the inference controller and every router to accept connections before it uses them, so a
deployment installed a moment earlier is not a race.

## Lifecycles are per deployment

- Each launch installs its own helm release, named `miles-run-<run id>-<component>`, and only a whole
  release is installed, upgraded or uninstalled at a time. Every object, hostname and label a deployment
  computes comes from its own release name - the api server's host included - so nothing it derives can
  point into another deployment.
- Only the release that carries the orchestration script has a "training finished" verdict: it writes
  the exit file, the launcher waits for it, and it uninstalls itself when the run ends.
- A trainer or inference release has no training to finish. Its launch returns once the release is
  installed, and the release stays up until you uninstall it:
  `python -m miles.utils.external_utils.miles_workbench stop -n "$MILES_NS" <run id> --deploy-component trainer`.
- Fault tolerance does not cross a release either. The api server and the mini ft controller answer for
  the cells of their own deployment, so a split run refuses them (`--api-server-port 0`) rather than
  reporting every trainer and inference cell as missing. A trainer deployment keeps the fault tolerance
  of its own ranks, which its own controller drives.
- Failure does not cross a release. An orchestration script that loses a controller fails loud in its
  own release and tears down nothing else, so the trainer and the inference side survive it — which is
  also what a later hot restart will need. A failed weight update is aborted on the inference controller
  before the failure is re-raised, so the surviving release keeps neither the update window's lock nor
  its paused health checking.
- A CI launch cleans up the leftover CI releases of *other* runs before installing its own; the sibling
  releases of the run it is launching are never touched.
- Relaunching a run id only resizes pools, per release, exactly as it does for an unsplit run.

## Arguments stay the user's responsibility

The launcher is the single source of a deployment's arguments: one render produces the command line of
every pod in that release, so a release is internally consistent. Across releases nothing checks yet
that the launches agree — pass the same base arguments to all of them, and change them in all of them.
The static addresses are not part of that base: they belong to the launch that has to reach what it
does not deploy.
Consistency checks between the releases of one run are a planned hardening, not something that exists
today.

## What this is not

- **Not, on its own, several trainers or several engine pools.** A split names one instance of each.
  Installing one release per policy trainer, and registering engine-only deployments into the run's
  one inference controller, is [multi instance deployment](/advanced/multi-instance-deployment).
- **Not a hot restart.** A new orchestration script does not reattach to a running trainer; the
  surviving releases are only the precondition for that.
- **Not external rollout.** [External rollout](/advanced/external-rollout) hands Miles engines it does
  not manage. Here Miles manages every worker, in deployments of its own.
