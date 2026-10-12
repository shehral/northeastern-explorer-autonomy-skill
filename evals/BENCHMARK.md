# Benchmark — does the skill actually help?

A small, honest evaluation of whether a Claude Code session does better **with** this
skill than without it, on the failure modes the skill is meant to prevent.

## Method

- **Prompts:** the six scenarios in `evals.json` (an overnight self-managing run, a
  heartbeat/Monitor setup, a garbage-output + silent-no-op debugging task, a
  first-real-run field test covering key-auth connection, a node launch-kill diagnosis,
  and short-partition timeout recovery, a GPU-type placement task, and a job pending
  forever on the per-job GPU cap plus a resumed session that lost its watchers).
- **Two configurations per prompt:** one session **with the skill loaded**, one
  **baseline** with no skill. Same prompt, same model class, isolated subagents.
- **Grading:** each scenario has an assertion checklist. Assertions are checked by
  regex against the saved response, so grading is objective, not eyeballed.
- **Model:** Claude (Opus-class) general-purpose subagents.
- **Scale:** one trial per cell (3 graded prompts x 2 configs = 6 runs so far). This is **directional
  evidence**, not a tight statistical estimate. Re-run before trusting the magnitude.

## Result

| Metric | With skill | Without skill | Delta |
|--------|-----------:|--------------:|------:|
| Assertion pass rate | **100%** | ~73% | **+0.27** |

Per scenario (assertions passed / total):

| Scenario | With skill | Baseline |
|---|:--:|:--:|
| overnight self-managing run | 6 / 6 | 5 / 6 |
| heartbeat + Monitor setup | 5 / 5 | 3 / 5 |
| garbage-output + no-op debug | 4 / 4 | 3 / 4 |
| first-real-run field test (connect + launch-kill + timeout) | — | — |
| GPU-type placement (2.0.0) | — | — |
| pending forever + session resume (2.0.0) | — | — |

The with-skill responses also ran modestly longer (a small latency cost from the added
discipline).

**Regrade pending (2.0.0).** The three graded rows were scored against the v1.0
expectations. Version 2.0.0 corrected several of those expectations (GPU jobs offline
with a CPU-job pre-cache and the inherited proxy unset, a Monitor that polls `sacct`,
a per-job GPU type, dtype from compute capability, Desktop scheduled tasks rather than
cloud routines for a durable tick), so the numbers above describe the old skill and the
old checklists. The fourth, fifth and sixth scenarios have never been graded. Re-run
both configs on all six prompts and refill the table.

The fourth (first-real-run) and sixth (pending forever + resume) scenarios should be the
most discriminating. The baseline has no way to know that a key-auth probe via the
SSH-config alias skips MFA, that a seconds-fast signal-style launch-kill is a *node*
fault rather than a code bug or preemption, that a short-partition timeout of working
code is recoverable by CPU-job re-aggregation, that `QOSMaxGRESPerJob` never clears,
that raising a TimeLimit is admin-only, or that watchers are not restored on resume. All
of that is tacit cluster knowledge, so they should widen the delta, not narrow it.

## What the baseline missed

The assertions the no-skill baseline dropped were exactly the non-obvious,
cluster-specific footguns the skill encodes:

- **Compute-node offline mode** (`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` /
  `local_files_only`). The baseline wrote a reasonable sbatch but omitted it — the #1
  cause of silent meta-init-zeros and a whole walltime burned on noise.
- **Pre-cache + force offline** as the paired fix for the same bug (2.0.0 moves the
  pre-cache into a CPU job; the login node kills heavy work).
- **Persistent Monitor + bounded tick.** The baseline reached for `--dependency`
  chaining and missed the `persistent: true` rule for >1h jobs and the per-tick caps.
- **Key-auth-via-alias connection.** The skill knows that a non-interactive
  `ssh -o BatchMode=yes <alias>` against an SSH-config block with an authorized
  `IdentityFile` + `ControlMaster`/`ControlPersist` authenticates by key with no MFA
  prompt (and that the raw hostname bypasses the config and fails). The baseline defaults
  to an interactive login and eats the MFA round-trip.
- **Node launch-kill vs. code bug vs. preemption.** A seconds-fast `State=FAILED` with a
  `.batch` CANCELLED, a signal-style ExitCode, a COMPLETED `.extern`, and no log written
  is a node-level launch-kill — the skill diagnoses it with the canary ladder, recovers
  with `--exclude`, and crucially does *not* increment the G5 failure streak. The
  baseline misreads it as its own bug and burns ticks "fixing" working code (the priciest
  failure mode G5 exists to stop).
- **Timeout recovery by CPU-job aggregation.** When a short partition times out a job
  that already finished its real per-item work, the skill re-aggregates CPU-only in an
  `srun -p short` job from the idempotent per-item outputs instead of re-running on the
  GPU (1.x said "on the login node"; 2.0.0 corrects that).

Assertions that **both** configurations passed (gpu partition selection, budget caps,
smoke-testing) are non-discriminating: they confirm the skill isn't wrong, but the
skill's real value is the tacit cluster knowledge that the delta isolates.

## Reproduce it

Use any harness (for example the Claude Code skill-creator eval loop) that can:
1. run each `evals.json` prompt once with the skill loaded and once with no skill,
2. grade each response against its `expected_output` assertion checklist,
3. compare pass rates.

If you change the skill, re-run this and update the numbers above.
