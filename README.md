# northeastern-explorer-autonomy

A Claude Code skill that lets a session operate the Northeastern Explorer HPC cluster (SLURM; H200, A100, V100 and T4 GPUs) autonomously, with guardrails. It teaches Claude how to connect through an SSH alias, set up the pinned environment, choose the GPU type each job fits and can start on soonest, submit only smoke-tested SLURM jobs through a verified hook layer, watch long jobs to completion, integrate results, and keep itself alive over hours via a self-rescheduling heartbeat. The point is hands-off overnight operation that is disciplined rather than reckless: the session can make hundreds of cheap moves and a handful of expensive ones without a human in the loop, and it never modifies its own rules unilaterally.

## Why it exists

Ad-hoc autonomous cluster use hits a small set of failure modes that are expensive in real money and wasted nights. This skill encodes the rules that prevent them:

- **Silent model collapse and mid-run downloads.** A checkpoint whose weights are missing or mis-tied can load "successfully" with freshly initialized tensors and generate noise for the entire walltime. Separately, a GPU job that can reach the hub may hang, retry, or resolve a different revision mid-run. The skill pre-caches the pinned revision in a CPU job (never on the login node), proves an offline load works there, runs every GPU job with mandatory offline flags (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `local_files_only=True`) and with the proxy it would otherwise inherit from the login shell unset, so a missing file fails loudly, and asserts an invariant only the real weights satisfy.
- **Runaway GPU spend and long queues.** One bad decision in a heartbeat tick could queue six 8-hour jobs (48 GPU-hours) before anyone notices. A per-day GPU-hour budget ledger, recorded per GPU type (guardrail G4), caps spend and forces pacing. A GPU-type selection procedure sends each job to the type that fits its measured memory, that the env supports, and that has free GPUs within the per-user QOS caps, instead of queueing everything behind the most contended type.
- **Autonomous bug-loops.** The priciest unattended failure is a loop of broken code, failed job, broken fix, failed job. A consecutive-failure rollback (guardrail G5) trips after N agent-code bugs in a row and downgrades all auto-submits to human approval. Crucially, infrastructure failures (HF timeout, scp race, node OOM, a walltime-kill of code that was working) do not count toward the streak, so the agent is not punished for a flaky cluster.
- **Silent no-ops masquerading as results.** An intervention bake-off can run for hours and produce identical outputs across every condition because the hook was registered but never actually fired. The verifier layer asserts a with-intervention run produces different output than without before any GPU-hour burns. A 4.5-hour bake-off was lost to exactly this before the check existed.

## Evidence it helps

The `evals/` folder contains five scenario prompts (an overnight self-managing run, a heartbeat/monitor setup, a garbage-output plus no-op debugging task, a first-real-run field test, and a GPU-type placement task) with assertion checklists, plus the measured results in `evals/BENCHMARK.md`. Only the first three have been graded, and against the v1.0 expectations; see BENCHMARK.md.

| Metric | With skill | Without skill | Delta |
|--------|-----------|---------------|-------|
| Assertion pass rate | 100% | ~73% | +0.27 |

Across the three graded scenarios (v1.0 skill and checklists), the session given the skill satisfied all of the graded assertions, while the no-skill baseline averaged roughly 73%. Be honest about what this is and is not: it is a single trial per cell (one with-skill and one without-skill response per scenario), so the numbers are **directional evidence that the skill helps on these failure modes**, not a tight statistical estimate. The benchmark also showed the with-skill responses ran modestly longer (a small added latency from the extra discipline). Re-run the evals yourself before trusting the magnitude.

## Requirements

- **Claude Code.** The skill is plain Markdown plus a few asset files. There is **no build step**, no dependencies to install, and nothing to compile.
- **An Explorer account with a project allocation.** You need SSH key access to `login.explorer.northeastern.edu` (through a `Host` alias in `~/.ssh/config`), a persistent `/projects/<group>/<user>` allocation (not `/scratch`, not `/home`), and a SLURM account to bill GPU time to. Without an allocation the skill has nothing to drive.

## Install

Clone the repo, then copy the **inner skill folder** into your personal skills directory:

```bash
git clone <repo-url> northeastern-explorer-autonomy-skill
mkdir -p ~/.claude/skills
cp -R northeastern-explorer-autonomy-skill/northeastern-explorer-autonomy ~/.claude/skills/
```

The folder you copy is `northeastern-explorer-autonomy/` (the one containing `SKILL.md`), not the repo root. After copying, restart Claude Code so it re-scans the skills directory, then verify it loaded:

```bash
ls ~/.claude/skills/northeastern-explorer-autonomy/SKILL.md
```

In a fresh Claude Code session you can confirm by typing `/` and looking for `northeastern-explorer-autonomy` in the skill list, or by asking Claude to "submit this to the cluster" and watching it reach for the skill.

**Project-scoped alternative.** If you only want the skill available inside one repo (for example, to keep its hooks and ledgers next to that project), copy it into that project's `.claude/skills/` instead of `~/.claude/skills/`:

```bash
mkdir -p /path/to/your/project/.claude/skills
cp -R northeastern-explorer-autonomy-skill/northeastern-explorer-autonomy /path/to/your/project/.claude/skills/
```

## Customize for your account (required)

The skill ships with `<angle-bracket>` placeholders for every account-specific value. **It will not work until you replace them.** Open `northeastern-explorer-autonomy/SKILL.md` (the "Customization Points" block), `references/cluster-setup.md`, and `assets/job_template.sbatch`, and set each one to your own values.

| Placeholder | Set it to | What it is |
|---|---|---|
| `<alias>` / `<user>` | Your SSH-config `Host` alias and Explorer username | Every ssh/scp/rsync goes through the alias (`HostName login.explorer.northeastern.edu`, key, ControlMaster) |
| `/projects/<group>/<user>` | Your project allocation path | Persistent storage for everything that must survive (not `/scratch`, not `/home`) |
| `<slurm-account>` | Your SLURM account | The `--account` your GPU time bills to |
| `$ALLOC/conda_envs/<conda-env>` | Your pinned conda env path | The environment activated via PATH-export in jobs. Write `<env>/.built-ok` once a build finishes: the template exits if the marker is missing |
| `$ALLOC/hf_cache` | Your model cache path | Pre-populated by a CPU pre-cache job (`srun -p short`) before any GPU job |
| `<squid-proxy-ip>` | The cluster's squid proxy address (from the research-computing docs) | Jobs inherit it from the login shell; exported explicitly in CPU pre-cache jobs and unset in GPU jobs |
| `<type>` | A GPU type per job (`h200`, `a100`, `v100-sxm2`, `v100-pcie`, `t4`) | Chosen per job with `references/cluster-setup.md` §5.1 |
| `<org>/<model>` @ `<revision>` | Your pinned model id and commit | If you load from a model hub; the pre-flight gate checks this snapshot exists |
| `<proj>` | Your repo name on the cluster | The working directory the job `cd`s into |
| `<jobname>` / `<script>.py` | Your job name and entry script | Per-job fields in the sbatch template |
| `DAILY_GPU_BUDGET` (default `24`) | Your daily GPU-hour soft cap (all types; per-type caps optional) | G4 budget threshold; tune per project |
| `FAIL_STREAK_MAX` (default `3`) | Your consecutive-agent-bug rollback threshold | G5 auto-rollback trigger; tune per project |

**Find every placeholder before your first run:**

```bash
grep -rn '<[^>]*>' ~/.claude/skills/northeastern-explorer-autonomy/
```

What you do **not** need to change: the **Explorer-wide values are correct as-is** and should usually be left alone. These include the pinned module (`miniconda3/24.11.1`; no `cuda/*` module for pip-installed torch, since the wheel carries its own CUDA runtime), the partition names (`gpu` for the 8h queue, `gpu-short` for smoke tests), and the GPU type names. The partition table in `references/cluster-setup.md` §5 (default and maximum times, per-job and per-user GPU caps, access) was recorded on 2026-10-12; re-check yours with `sacctmgr` and `scontrol show partition`.

**Delete the model-specific loader fix if it does not apply to you.** `references/cluster-setup.md` §7 contains an `lm_head` tie workaround that is specific to one Qwen-VL-family checkpoint (the checkpoint omits `lm_head.weight` and relies on `tie_word_embeddings`, which a newer transformers version reads off the wrong config level, leaving `lm_head` at meta-init zeros). If you are not on that model family, this fix is unnecessary or different. Remove or replace it so it does not silently change your loader.

## Repository layout / file-by-file guide

```
README.md, CHANGELOG.md, LICENSE, CONTRIBUTING.md, .gitignore
evals/
  evals.json
  BENCHMARK.md
northeastern-explorer-autonomy/        <- THE SKILL; copy this folder into ~/.claude/skills/
  SKILL.md
  references/
    cluster-setup.md
    autonomy-model.md
    heartbeat-loop.md
    safety-hooks.md
    long-job-patterns.md
  assets/
    job_template.sbatch
    compute_budget.json
    failure_streak.json
```

| File | What it is and when you touch it |
|---|---|
| `northeastern-explorer-autonomy/SKILL.md` | The skill entry point. The mental model (three nested loops), the Customization Points table, the one offline rule, the tier/guardrail summary, and pointers into the reference files. **Edit first** to fill placeholders; Claude reads it whenever the skill triggers. |
| `references/cluster-setup.md` | Connection (SSH alias, ControlMaster, Duo, the silent login-node kill), allocation and storage, CPU-job pre-caching with an offline load test, environment activation via PATH-export, the offline rule, the partition table (default and maximum times, per-job and per-user GPU caps, access), IdleBot and dependency rules, **choosing a GPU type per job (§5.1)**, the canonical sbatch anatomy and its silent-failure traps, loader gotchas (dtype per GPU type, attention impl, the model-specific `lm_head` fix), data movement, and field-tested troubleshooting. **Read first when setting up a session or writing a new job; edit the placeholders and delete the loader fix if it does not apply.** |
| `references/autonomy-model.md` | The T0–T3 action tiers, the five guardrails G1–G5, the agent-code-vs-infrastructure failure distinction, and the exact JSON shapes and semantics of the two ledgers. **Read when deciding what the session may auto-do; edit the tunable thresholds (walltime cap, budget, fail-streak, info-gain).** |
| `references/heartbeat-loop.md` | The bounded recurring tick (step sequence, call and word caps), the scheduling mechanisms (`/loop`, `CronCreate`, the `/schedule` skill) and their durability tradeoffs, the cache-window delay rule (never 300s), persistent-Monitor wiring for long jobs, how heartbeat plus Monitor compose, and how to recover a stalled session. **Read when configuring overnight or recurring operation.** |
| `references/long-job-patterns.md` | Patterns for multi-hour and multi-day GPU campaigns: interleaved sharding with checked merges, resumable shard jobs, budget estimation from a smoke shard, memory-adaptive batch and dtype, saturation checks, artifact stamping, fp16-store/fp32-load, and the loader trust check. **Read before designing any fit or sweep expected to exceed a few GPU-hours.** |
| `references/safety-hooks.md` | The PreToolUse hook wiring (settings.json snippet, exit 0 allow / 2 block), the per-job verifier pattern with its reusable `_lib.sh`, the verifier checklist (existence, static structure, smoke integrity, end-to-end wiring, norm-matched controls), and the graceful-degrade rule for dev boxes. **Read when wiring the verified submit path; adapt the verifier checks to your experiment.** |
| `assets/job_template.sbatch` | The canonical fill-in-the-blanks SLURM job. Encodes the pinned module, PATH-export activation, a per-job GPU type, mandatory offline flags, cache paths on the allocation, cuBLAS determinism, a one-op CUDA sanity check, a GPU-utilisation logger (IdleBot), a pinned-snapshot pre-flight gate, and an early-signal checkpoint hook. **Copy and fill the `<…>` / `CUSTOMIZE` fields for each new job.** |
| `assets/compute_budget.json` | Starter G4 ledger with the exact schema the budget guardrail expects (daily GPU-hour cap, hours per GPU type, per-day job list, atomic-write note). **Copy into your project and let the session update it; edit the soft cap to your budget.** |
| `assets/failure_streak.json` | Starter G5 ledger with the schema for the consecutive-failure rollback (current/max streak, rollback flags, history with the `counted_in_streak` agent-vs-infrastructure tag). **Copy into your project; the human resets it after a rollback, never the agent.** |
| `evals/evals.json` | Five scenario prompts with expected-output assertion checklists used to grade the skill. **Read or run when you want to verify or extend the quality evidence.** |
| `evals/BENCHMARK.md` | The measured with-skill vs no-skill results summarized above. **Read for the evidence; regenerate when you change the skill.** |

## How to use it

The skill is written to trigger on natural cluster phrasing. Once installed and customized, just ask in plain language and Claude will load it:

- "Submit this to the cluster."
- "Kick off an overnight run that submits follow-up jobs as results land."
- "Watch this SLURM job until it finishes."
- "Set up my Explorer environment."
- "Let Claude run experiments on its own tonight."
- "My model loads fine on the compute node but generates garbage. What's wrong?"
- "Recover my stalled autonomous session."

You can also invoke it explicitly:

```
/northeastern-explorer-autonomy
```

A typical first session: connect through the alias and verify the session is warm, pre-cache the model in a CPU job and test an offline load, smoke-test on `gpu-short`, choose the GPU type, compose the sbatch from the template, verify and submit, arm a persistent Monitor that polls `sacct`, and (if running unattended) end with a self-reschedule.

## Adapting to a different cluster

Most of this skill is portable; a smaller part is Explorer-specific.

**Portable as-is** (the conceptual core): the tiered autonomy model (T0–T3), the five-guardrail shape, the agent-code-vs-infrastructure failure distinction, the append-only budget and failure-streak ledgers, the bounded self-rescheduling heartbeat, the cache-window delay rule, the persistent-Monitor pattern, and the PreToolUse-hook-plus-verifier safety layer. These describe disciplined autonomous operation, not Explorer in particular.

**Explorer-specific** (rewrite for your site): the SSH target, allocation and storage paths, the SLURM account, partition and QOS names, the pinned module, the proxy address, the GPU type names and the IdleBot policy, and the model-specific loader fix. The offline rule for GPU jobs is portable: it makes a missing input fail loudly wherever you run. If your jobs do not pull from a model hub, replace it with an "inputs are present and local" check.

## Safety note

Do not flip everything on at once. Start a new project at **T0 + T2 only**: let the session observe, edit, smoke-test, and **recommend** the next job rather than firing it. Promote to T1 auto-submit only after you have watched the verifier, the budget guardrail, and the failure-streak rollback behave correctly across several manual cycles and you trust the smoke-test gate. The ledgers and the append-only auto-submit log are exactly the evidence that lets you (and your collaborators) make that promotion deliberately rather than on faith. The agent never clears its own rollback; a human resets the failure streak.

## License

This project is released under the MIT License. See `LICENSE`.
