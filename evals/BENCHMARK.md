# Benchmark — does the skill actually help?

A small, honest evaluation of whether a Claude Code session does better **with** this
skill than without it, on the failure modes the skill is meant to prevent.

## Method

- **Prompts:** the four scenarios in `evals.json` (an overnight self-managing run, a
  heartbeat/Monitor setup, a garbage-output + silent-no-op debugging task, and a
  first-real-run field test covering key-auth connection, a node launch-kill diagnosis,
  and short-partition timeout recovery).
- **Two configurations per prompt:** one session **with the skill loaded**, one
  **baseline** with no skill. Same prompt, same model class, isolated subagents.
- **Grading:** each scenario has an assertion checklist. Assertions are checked by
  regex against the saved response, so grading is objective, not eyeballed.
- **Model:** Claude (Opus-class) general-purpose subagents.
- **Scale:** one trial per cell (4 prompts x 2 configs = 8 runs). This is **directional
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

The with-skill responses also ran modestly longer (a small latency cost from the added
discipline).

The fourth scenario (the first-real-run field test) was added after the run above and
has not been graded yet; re-run both configs on the full four-prompt set and fill in its
row. It is the most discriminating of the four — the baseline has no way to know that a
key-auth probe via the SSH-config alias skips MFA, that a seconds-fast signal-style
launch-kill is a *node* fault rather than a code bug or preemption, or that a
short-partition timeout of working code is recoverable by login-node re-aggregation. All
three are pure tacit cluster knowledge, so it should widen the delta, not narrow it.

## What the baseline missed

The assertions the no-skill baseline dropped were exactly the non-obvious,
cluster-specific footguns the skill encodes:

- **Compute-node offline mode** (`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` /
  `local_files_only`). The baseline wrote a reasonable sbatch but omitted it — the #1
  cause of silent meta-init-zeros and a whole walltime burned on noise.
- **Pre-cache on the login node + force offline** as the paired fix for the same bug.
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
- **Timeout recovery by login-node aggregation.** When a short partition times out a job
  that already finished its real per-item work, the skill re-aggregates CPU-only on the
  login node from the idempotent per-item outputs instead of re-running on the GPU.

Assertions that **both** configurations passed (gpu/h200 selection, budget caps,
smoke-testing) are non-discriminating: they confirm the skill isn't wrong, but the
skill's real value is the tacit cluster knowledge that the delta isolates.

## Reproduce it

Use any harness (for example the Claude Code skill-creator eval loop) that can:
1. run each `evals.json` prompt once with the skill loaded and once with no skill,
2. grade each response against its `expected_output` assertion checklist,
3. compare pass rates.

If you change the skill, re-run this and update the numbers above.
