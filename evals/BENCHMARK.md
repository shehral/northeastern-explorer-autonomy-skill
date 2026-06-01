# Benchmark — does the skill actually help?

A small, honest evaluation of whether a Claude Code session does better **with** this
skill than without it, on the failure modes the skill is meant to prevent.

## Method

- **Prompts:** the three scenarios in `evals.json` (an overnight self-managing run, a
  heartbeat/Monitor setup, and a garbage-output + silent-no-op debugging task).
- **Two configurations per prompt:** one session **with the skill loaded**, one
  **baseline** with no skill. Same prompt, same model class, isolated subagents.
- **Grading:** each scenario has an assertion checklist. Assertions are checked by
  regex against the saved response, so grading is objective, not eyeballed.
- **Model:** Claude (Opus-class) general-purpose subagents.
- **Scale:** one trial per cell (3 prompts x 2 configs = 6 runs). This is **directional
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

The with-skill responses also ran modestly longer (a small latency cost from the added
discipline).

## What the baseline missed

The assertions the no-skill baseline dropped were exactly the non-obvious,
cluster-specific footguns the skill encodes:

- **Compute-node offline mode** (`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` /
  `local_files_only`). The baseline wrote a reasonable sbatch but omitted it — the #1
  cause of silent meta-init-zeros and a whole walltime burned on noise.
- **Pre-cache on the login node + force offline** as the paired fix for the same bug.
- **Persistent Monitor + bounded tick.** The baseline reached for `--dependency`
  chaining and missed the `persistent: true` rule for >1h jobs and the per-tick caps.

Assertions that **both** configurations passed (gpu/h200 selection, budget caps,
smoke-testing) are non-discriminating: they confirm the skill isn't wrong, but the
skill's real value is the tacit cluster knowledge that the delta isolates.

## Reproduce it

Use any harness (for example the Claude Code skill-creator eval loop) that can:
1. run each `evals.json` prompt once with the skill loaded and once with no skill,
2. grade each response against its `expected_output` assertion checklist,
3. compare pass rates.

If you change the skill, re-run this and update the numbers above.
