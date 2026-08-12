---
title: Training Several Policy Models in One Run
description: Give every policy model its own megatron arguments, its own rollout queue, its own weight updates and its own rollout_id.
---
One run normally trains one policy model. Multi policy training gives a run several of them: each
policy has its own trainer, its own training rhythm, and its own inference engines, while sharing one
data source and one rollout executor.

<Warning>

**Status.** Under active development. The training semantics below are in place and a multi policy
launch runs end to end - the two gpu e2e tests for it are enabled. Deploying each policy's trainer as
a release of its own is described in
[multi instance deployment](/advanced/multi-instance-deployment).

</Warning>

## The model id

- Every policy is named. That name is the model id, and it is the same string on both sides of the run:
    - `--megatron-config` declares it as a `name`, which is the source of truth.
    - `--sglang-config` must declare a model with the same `name` and `update_weights: true`.
    - the rollout function stamps it into `Sample.trainer_model_id`.
- A run without `--megatron-config` is a single policy run named `default`; nothing about it changes.

## `--megatron-config`

Symmetric to `--sglang-config`: a YAML file (or an inline `base64:<payload>`) listing one entry per
policy, each with the extra megatron CLI arguments that policy overrides.

```yaml
megatron:
  - name: policy_a
    args: --lr 1e-6 --tensor-model-parallel-size 2
  - name: policy_b
    args: --lr 5e-7
```

- The `args` string is overlaid on the parsed base arguments, per policy. An argument the base parser
  does not know is a startup error, not a silently dropped setting.
- The first entry is the **primary** policy. It owns the run's global checkpoint index.
- Internally a run is always a list of policy configs; omitting the flag builds a single-element list.
- A single policy config with a non-empty `args` is refused: nothing applies a per-policy overlay to a
  single policy run, so those arguments belong on the command line.

### Which arguments may differ between policies

Only the arguments a trainer reads for itself may be overridden. Everything else — the rollout rhythm
(`--num-rollout`, `--save-interval`, `--update-weights-interval`, `--debug-exit-after-rollout`), the
shared data source and the shared rollout executor — is read from the base command line by
`train_multi_policy.py`, so accepting a per-policy value would silently do nothing. A key outside the
list below is a startup error naming that key.

| Group | Keys |
| --- | --- |
| model identity | `--hf-checkpoint`, `--ref-load`, `--megatron-to-hf-mode` |
| optimizer | `--optimizer`, `--lr`, `--min-lr`, `--lr-decay-style`, `--lr-warmup-iters`, `--lr-warmup-fraction`, `--weight-decay`, `--adam-beta1`, `--adam-beta2`, `--clip-grad` |
| megatron parallelism | `--tensor-model-parallel-size`, `--pipeline-model-parallel-size`, `--context-parallel-size`, `--expert-model-parallel-size`, `--expert-tensor-parallel-size`, `--sequence-parallel` |
| batching | `--global-batch-size`, `--micro-batch-size`, `--max-tokens-per-gpu`, `--use-dynamic-batch-size` |
| loss | `--advantage-estimator`, `--use-kl-loss`, `--kl-loss-coef`, `--kl-loss-type`, `--entropy-coef`, `--eps-clip`, `--eps-clip-high` |

`--save` and `--load` are not in the list: they are derived per policy, see below.

## `Sample.trainer_model_id`

- The field is written by the custom rollout (generate) function, when it builds the sample.
- `None` means "the only policy". In a single policy run that resolves to it; in a multi policy run
  `None` (or an unknown id) fails loudly before the sample is enqueued, rather than training some
  policy on another policy's data.
- All samples of one prompt group must name the same policy: group-relative advantages are meaningless
  once a group is split across policies.
- See `examples/multi_policy/round_robin_generate.py` for the smallest possible implementation.

## Data flow

- The fully-async producer keeps one output queue per model id.
- Finished groups are grouped by `trainer_model_id` before being enqueued, and dispatched into the
  queue of their own policy.
- A training step drains only its own policy's queue: `RolloutExecutor.get(rollout_id,
  trainer_model_id=...)`.
- Staleness is per policy too: each policy publishes its own engine weight version.
- Backpressure is per policy. A single policy run enqueues inline, so a full buffer stops the producer
  exactly as it always did and the amount of generated-but-unconsumed data does not change. A multi
  policy run stages a finished group per policy instead, bounded by the same in-flight group budget the
  producer generates under, so a policy that stopped consuming holds up only its own queue and cannot
  grow an unbounded backlog of fully generated groups.
- A group that arrives for a policy whose staging area is already full goes to
  `--async-unused-samples-handler` (recycled or dropped) and is counted in
  `<model_id>/rollout/fully_async/staged_put_overflow_groups`. A steadily rising counter means that
  policy consumes slower than the run generates for it.

## Weight updates

- `InferenceController.start_update_weights(model_id=...)` returns exactly the engines of the named
  model, so a policy's broadcast never reaches another policy's engines.
- `check_weights(..., model_id=...)` is scoped the same way.
- Without a `model_id` the controller keeps the old single-updatable-model behaviour, so existing runs
  are unaffected.
- `updatable_model_ids()` lists the model ids the orchestration script must drive.

## Save and load

Each policy gets a checkpoint directory of its own, derived from the run's `--save` / `--load`:

- `--save <dir>` becomes `<dir>/policies/<model_id>` for every policy, and `--load` likewise. A model id
  must match `[A-Za-z0-9_-]+`, so it always names exactly one directory under them.
- Global state (the shared sidecar below, the rollout executor) stays at the root `<dir>`.
- **A single policy run derives nothing**: its checkpoints keep exactly the paths they had before, so
  existing checkpoints and existing resume commands are unaffected.

Global state (data source, rollout executor) is shared, so it is saved once, indexed by the **primary**
policy's `rollout_id`:

- when the primary reaches a save point, every other policy finishes its current round and parks;
- each policy saves its own model checkpoint at its own `rollout_id`, with the primary's `force_sync`;
- the primary then saves the global state and writes the other policies' `rollout_id`s next to its own
  checkpoint (`<save>/multi_policy_state/rollout_ids_<primary_rollout_id>.json`), recording only the
  policies that actually parked, plus the ones that had already finished;
- on load, every recorded policy's restored `rollout_id` is asserted against what the primary recorded,
  so a run can never resume from an inconsistent mixture of checkpoints. The record is read from
  `--load` when it is given, and from `--save` otherwise.
- the primary does not wait forever: if a policy never parks the save fails loudly with each policy's
  parking state rather than hanging the run.

Every policy also saves its own last round. The policies share one `--num-rollout` but not one speed,
so the primary usually exits while the others are still training, and from then on nobody drives a
global save:

- when a policy finishes its rounds it takes a final checkpoint of its own, unless the round it just
  finished is already the one it saved at;
- the final save does not need the primary to still be running, but it is still coordinated: it never
  overlaps a global save or another policy's final save, and it rewrites the record so the record names
  the last position of every policy;
- a resume therefore finds each policy exactly where the record says, including the policies that
  finished after the primary did.

`rollout_id` keeps its name; its meaning becomes *one `rollout_id` per policy model*.

<Warning>

**Known limitation.** The other policies stop and wait at a save point, so a global checkpoint costs a
bubble that grows with how far apart the policies are running.

</Warning>

## Restrictions

- Multi policy requires `--fully-async`. Every other rollout mode drives one policy per rollout round;
  the combination is rejected during argument validation.
- Multi policy requires `--sglang-config`, with one updatable inference model per policy.
- `--use-critic` and `--colocate` are not supported.
- The orchestration script is `train_multi_policy.py`.
- Every policy's trainer is addressed by its model id, and may be installed as a release of its own;
  see [multi instance deployment](/advanced/multi-instance-deployment).
- Evaluation is not supported. `train_multi_policy.py` has no eval dispatcher, so `--eval-interval`
  is rejected rather than accepted and ignored; convergence is read from the per policy training
  curves.
- All policies share one tokenizer: the rollout process builds it once from the run's
  `--hf-checkpoint`. When the policies do not all use that same checkpoint, every checkpoint involved -
  the per policy ones and the run's own `--hf-checkpoint` - is checked at startup to agree on the
  vocabulary size and then on a fingerprint of the tokenizer itself (vocabulary contents and special
  tokens), because two tokenizers of the same size can still map the same text to different ids.
- A per policy argument that other arguments are derived from is re-derived for that policy:
  `--advantage-estimator` re-derives `use_critic` (so a policy asking for PPO is refused like the run
  would be), and an overridden `--global-batch-size` is checked against `--num-steps-per-rollout`
  instead of silently changing that policy's gradient steps per round.

## Metrics

- In a multi policy run every metric is prefixed with the model id, on the rollout side
  (`<model_id>/rollout/...`, `<model_id>/perf/...`, `<model_id>/passrate/...`) and on the trainer side
  (`<model_id>/train/...`, and the trainer's own `<model_id>/rollout/...` and `<model_id>/perf/...`),
  so two policies writing into one tracking run do not interleave into a single unreadable curve.
- Each prefix is logged against its own step key: `<model_id>/rollout/step` for the rollout-indexed
  metrics and `<model_id>/train/step` for the train-step ones. `train_multi_policy.py` declares both
  axes for every policy at startup, otherwise wandb plots those curves against its internal counter.
- A single policy run logs exactly the keys it logged before: no prefix, one `rollout/step` and one
  `train/step`.
