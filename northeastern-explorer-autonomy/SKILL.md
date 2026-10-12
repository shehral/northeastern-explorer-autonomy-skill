---
name: northeastern-explorer-autonomy
description: >-
  Operate the Northeastern Explorer HPC cluster (SLURM; H200, A100, V100 and T4 GPUs)
  autonomously from a Claude Code session — connect, set up the pinned environment,
  pick the GPU type each job fits and can start on soonest, submit smoke-tested SLURM
  jobs through a safety-verified hook layer, and run a
  self-sustaining heartbeat loop with compute-budget and failure-streak guardrails
  so experiments run hands-off overnight. Use this whenever the user wants to run,
  submit, monitor, or schedule jobs on Explorer (login.explorer.northeastern.edu),
  set up autonomous or overnight HPC runs, spread jobs across GPU types to avoid long
  queues, configure a research heartbeat / cron /
  loop on a cluster, recover a stalled autonomous session, or onboard a collaborator
  to autonomous cluster operation. Trigger even when the user just says "submit this
  to the cluster", "kick off an overnight run", "watch this SLURM job", "set up my
  Explorer environment", or "let Claude run experiments on its own" — the guardrails
  and connection mechanics in here prevent the expensive footguns (silent meta-init
  model collapse, runaway GPU spend, bug-loops) that ad-hoc cluster use hits.
metadata:
  version: 2.0.0
  tags: [hpc, slurm, explorer, northeastern, autonomy, gpu-selection, heartbeat, guardrails]
---

# Operating Northeastern Explorer Autonomously with Claude Code

This skill turns a Claude Code session into a disciplined autonomous operator on the
**Northeastern Explorer** cluster: it connects, runs the right environment, submits
**only smoke-tested, verified** SLURM jobs, watches them to completion, integrates
results, and decides what to run next — all under guardrails that make unattended
operation safe rather than reckless.

It is written to be **reused by anyone** on Explorer. The cluster-specific values
(username, allocation, conda env, model) live in one **Customization Points** block;
everything else — the autonomy model, the heartbeat, the safety layer — is portable.

## When to reach for this

Use it when the work is "run things on the cluster, possibly while the human is away."
Submitting a job, watching a long job, scheduling a recurring research tick, recovering
a session that died overnight, or handing the pattern to a teammate. If the task is a
one-off local script with no cluster involved, you don't need this.

## The mental model: three nested loops

```
Session setup    →  connect, activate env, pre-cache model   (once per session)
  └─ Decision loop  →  for each candidate action, classify Tier T0-T3,
  │                    check guardrails G1-G5, act or escalate
  │     └─ Job loop  →  smoke-test → verify → sbatch → persistent Monitor →
  │                     intake result → update state → pick next
  └─ Heartbeat       →  every ~30 min, re-anchor state, run the decision loop,
                        emit a short report, reschedule itself
```

The four ideas working together — **tiered autonomy**, **guardrails**, **a verified
submit path**, and **a self-rescheduling heartbeat** — are what let a session make
hundreds of cheap moves and a handful of expensive ones without a human in the loop,
and never modify its own rules unilaterally.

## Customization Points — EDIT THESE FIRST for your account

Everything below assumes these values. **Replace every `<…>` placeholder with your own
account's values before first use** — the rest of the skill and the bundled
`assets/job_template.sbatch` reference them. Values shown without brackets
(module versions, partition names) are Explorer-wide and usually correct as-is.

| Point | Placeholder → set to yours | What it is |
|---|---|---|
| `SSH_ALIAS` | `<alias>` | a `Host <alias>` block in `~/.ssh/config` with `HostName login.explorer.northeastern.edu`, `User <user>`, an authorized `IdentityFile`, `ControlMaster auto`, `ControlPath`, `ControlPersist 8h`. Every ssh/scp/rsync goes through the alias. |
| `ALLOC` | `/projects/<group>/<user>` | your persistent project allocation (NOT `/scratch`, NOT `/home`) |
| `SLURM_ACCOUNT` | `<slurm-account>` | the `--account` your allocation bills to |
| `CONDA_ENV` | `$ALLOC/conda_envs/<conda-env>` | your pinned conda env path; write `<env>/.built-ok` when a build finishes (the template refuses an unmarked env) |
| `MODULES` | `miniconda3/24.11.1` | pinned module versions (Explorer-wide). No `cuda/*` module for pip-installed torch: the wheel carries its own CUDA runtime (`references/cluster-setup.md` §3) |
| `HF_CACHE` | `$ALLOC/hf_cache` | model cache (pre-populated by a CPU pre-cache job, never on the login node) |
| `PROXY` | `<squid-proxy-ip>:3128` | the cluster's squid proxy (from the research-computing docs). Jobs inherit it from the login shell: export it explicitly in CPU pre-cache jobs, **unset** it in GPU jobs |
| `MODEL_ID` / `REVISION` | `<org>/<model>` @ `<revision>` | your pinned model + commit (if you use a model hub) |
| `DAILY_GPU_BUDGET` | `24` GPU-hours | G4 soft cap per day, all types; the ledger records hours per GPU type (tune; add per-type caps if you need them) |
| `FAIL_STREAK_MAX` | `3` | G5 consecutive-agent-bug auto-rollback threshold (tune) |

If you are a **new user adapting this**, also rewrite any model-specific loader fix.
The `lm_head` tie workaround in `references/cluster-setup.md` §7 is a model-specific
*example* (one Qwen-VL-family checkpoint) — yours may differ or be unnecessary, so
delete or replace it. And if you are on a **different cluster**, see the README's
"Adapting to a different cluster" section: the autonomy model, heartbeat, and
guardrails are portable; only the connection/env/partition details change.

## Quick start (a single autonomous run, start to finish)

1. **Connect & verify the session is warm — try key-auth via the alias first.** If you
   have an SSH-config `Host <alias>` block (authorized `IdentityFile` +
   `ControlMaster auto` + `ControlPersist`), a non-interactive
   `ssh -o BatchMode=yes <alias> 'squeue -u <user>; sinfo -p gpu'` authenticates **by
   key with no MFA prompt** and opens the persistent master socket in the same call —
   so this single probe both connects and confirms warmth. Only if the key is **not**
   authorized do you fall back to an interactive `ssh <alias>` to clear Duo (reused for
   8h). **Always go through the configured `<alias>`, never the raw hostname** —
   connecting by hostname bypasses the `IdentityFile`/`ControlMaster` config and fails
   with a GSSAPI `Permission denied`. See `references/cluster-setup.md` §1.
2. **Pre-cache the model in a CPU job (`srun -p short`), never on the login node**, then
   prove an offline load works in that same CPU job before any GPU time is spent (GPU
   jobs run offline — see the rule below). `references/cluster-setup.md` §2. The login
   node has a ~8 GB effective RAM cap and kills heavy work silently.
3. **Smoke-test the experiment locally or on `gpu-short`** for ~60s on one input
   **before** any `gpu` submission (Guardrail G1; ~5000:1 cost asymmetry). Validate the
   **actual artifact you'll submit**: include at least one real `sbatch` smoke, not only
   an interactive `srun` — `srun` streams stdout over SSH and writes no output file, so
   it silently skips the failures (output-file creation, node prolog/launch) that only
   appear under `sbatch`. For anything expected to run more than a few GPU-hours, shape
   it with `references/long-job-patterns.md` first — interleaved shards, resumable
   checkpoints, and a saturation check routinely halve real spend.
4. **Choose the GPU type, then compose the sbatch.** Never hard-code one type: pick the
   one that fits the smoke's measured memory, that your env's torch build supports, and
   that can start soonest given free GPUs and your QOS caps
   (`references/cluster-setup.md` §5.1). Sweeps can split across several free types.
   Compose from `assets/job_template.sbatch` — it already encodes the
   pinned module, the PATH-export env activation, mandatory offline flags, cache
   paths, determinism, a pre-flight model-cache check, and an early-signal checkpoint hook.
5. **Verify, then submit.** A PreToolUse hook should run a per-job verifier before the
   `sbatch` lands (see `references/safety-hooks.md`). Submitting fires a job ID.
6. **Arm a persistent Monitor on the scheduler, not the log**: poll
   `sacct -j <id> -X -n -o State` every ~120 s until a terminal state. Slurm never
   writes COMPLETED or FAILED into the job log, and a launch-kill writes no log at all.
   For any job >1h, `persistent: true` (the persistent-Monitor rule). See
   `references/heartbeat-loop.md`.
7. **On wakeup, intake the result**, update your state ledgers, record success/failure
   in the failure streak, and let the decision loop pick the next action.
8. **If running unattended,** keep the loop alive with `/loop 30m ...` (self-paced) or
   a `CronCreate` cadence, or `/schedule` for a durable cloud agent. Use a 1200–1800s
   interval, **never 300s** (cache-window thrash). See `references/heartbeat-loop.md`.

## The one rule that prevents the most expensive failure

**GPU jobs run offline.** Compute nodes reach the internet only through the squid
proxy (`PROXY`), and jobs inherit it from the login shell by default. Export it
explicitly only in CPU pre-cache jobs; **unset it in every GPU job** and set the
offline flags, so a missing file fails loudly at load time instead of the job hanging,
retrying, or silently resolving a different revision mid-run:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   # in the sbatch
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy=localhost,127.0.0.1
```
```python
AutoModel...from_pretrained(MODEL_ID, revision=REV, local_files_only=True)  # in Python
```

Pre-caching (step 2) and offline-mode (this rule) are an inseparable pair — neither
works without the other. Some libraries call the hub API even for cached files and
crash under offline mode, which is why step 2 ends with an offline load test on a CPU
node. Full reasoning in `references/cluster-setup.md` §2 and §4.

### Field-tested gotchas

The first two are **infrastructure** failures — neither increments the G5 streak.
Diagnose before "fixing" the wrong layer; full recipes in `references/cluster-setup.md`
§6 and §9.

- **Node launch-kills.** Symptom: an `sbatch` job dies in 1–5s with `State=FAILED`, the
  `.batch` step `CANCELLED`, a signal-style `ExitCode` (e.g. `0:53`), the `.extern` step
  `COMPLETED`, and **no output log** (killed before the script's first line). This is a
  bad **node**, not a code bug or preemption. Diagnostic ladder: get the node from
  `sacct` `NodeList`; run a **minimal canary** (no model, tiny mem, short time, trivial
  payload, **same GPU type** as the failed job) — if it succeeds on a *different* node,
  the original node(s) are bad; rule out systemic causes (the job's fail reason, QOS
  limits, output-dir writability, maintenance reservations) so you don't mis-fix. Fix:
  add `#SBATCH --exclude=<bad-nodes>` and resubmit.
- **Short-partition timeout of finished work.** A short-walltime partition can
  **TIMEOUT a job that finished its real work** but not the final aggregation/write.
  Defend by writing per-item outputs as you go (idempotent skip-existing) and aggregating
  from them; if you time out near the end, re-aggregate CPU-only from the per-item files
  inside `srun -p short`, **never on the login node** — no GPU re-run. Size `--time` from
  the smoke's per-item time and don't undercut it.
- **Submission plumbing that fails silently.** `sbatch --export=ALL,IDX=1,2,3` splits on
  the commas, so the job sees `IDX=1` and still exits 0 — pass lists through the
  environment (`IDX=1,2,3 sbatch ...`) and assert the logged list equals the requested
  one. `ssh <alias> 'nohup srun ... &'` holds the ssh call open until srun ends; use
  `sbatch` for anything longer than a few seconds.

## The autonomy model (how to decide what to do without asking)

Classify every candidate action into a **tier**, and gate the auto-acting tier behind
**five guardrails**. This is the heart of safe unattended operation — full detail in
`references/autonomy-model.md`.

- **T0 — always allowed:** reads, greps, audits, local edits/commits, smoke tests,
  state-file updates, memory writes, dispatching read-only subagents.
- **T1 — auto-submit allowed (logged; surfaced only if anomalous):** an HPC job
  ≤ a walltime cap, **only if all five guardrails pass**.
- **T2 — do it, but surface in the next heartbeat:** new experiment designs, large
  code changes, new-checkpoint/cross-family runs.
- **T3 — never auto; escalate to the human:** destructive ops (`rm -rf`, force-push,
  cancelling others' jobs), external comms, **modifying your own protocol or the
  user's CLAUDE.md**, touching restricted/production artifacts.

The **five guardrails** that gate T1:

| | Guardrail | Threshold |
|---|---|---|
| **G1** | Smoke-test gate | 60s single-input run passed **via at least one real `sbatch`** (not only `srun` — `srun` writes no output file and skips launch/prolog failures), or script unchanged from a prior success |
| **G2** | Pre-registration | falsification criteria written **before** the run |
| **G3** | Info-gain | experiment discriminates between live hypotheses (≥ 0.3 if you score it) |
| **G4** | Daily compute budget | under `DAILY_GPU_BUDGET` GPU-h today, recorded per GPU type (`.compute_budget.json`) |
| **G5** | Failure-streak rollback | not under auto-rollback (`< FAIL_STREAK_MAX` consecutive **agent-code** bugs) |

G5's subtlety: only new agent-written-code bugs increment the streak. **Infrastructure
failures don't count:** HF timeouts, scp races, connection/auth failures, node
launch-kills, NODE_FAIL, a walltime timeout of code whose checkpoints were advancing,
and pending time. **Counts as agent code:** an IdleBot cancellation of a job the agent
designed with a long CPU-only phase inside a GPU allocation (a job-design bug). This
prevents bug-loops — the most expensive autonomous failure mode — while not punishing
the agent for the cluster being flaky. See the field-tested gotchas above and the
triage table in `references/autonomy-model.md`.

## The safety layer (make the verified path the only path)

Autonomy is only safe if the *submit* action is hard to get wrong. Wire two things
(details + copy-paste configs in `references/safety-hooks.md`):

1. **PreToolUse hooks** that intercept `sbatch`/`Edit`/`commit` and run a check
   *before* the call lands — blocking on failure so you see it inline.
2. **Per-job verifiers** (`verify_<job>.sh`) that enforce, before any GPU-hour burns:
   the script + sbatch exist, `--smoke` ran clean, offline flags are set,
   pre-registration constant present, and — critically (the end-to-end wiring check) —
   that a with-intervention run produces **different output** than without. A registered
   hook + a real bake-off both confirmed this catches the silent "the hook never
   actually fired" bug that otherwise looks like a real (null) result.

## The heartbeat (keep the session alive and disciplined over hours)

For multi-hour / overnight operation, run a recurring **tick** (~30 min). Each tick is
bounded (**≤25 tool calls, ≤300-word report**) and walks a fixed step sequence:
re-anchor state → audit consistency → check for landed jobs → integrate results →
update tournament/claims → run guardrail checks → maybe submit the next job → log →
emit report → **reschedule itself**. Full step list, cron setup, the `/loop` /
`CronCreate` / `/schedule` mechanisms, durability caveats, and the cache-window delay
rule are in `references/heartbeat-loop.md`.

Compose it with the Monitor: **cron-fired heartbeat** (periodic, broad) + **persistent
Monitor** (event-driven, fires ~1 min after a job hits a terminal state) = jobs get
integrated promptly without polling, and the loop survives even if one mechanism dies.

## Reference files (read the one you need)

- `references/cluster-setup.md` — connection, allocation/storage, env activation,
  offline rules, the canonical sbatch anatomy, model-loader gotchas, and the
  laptop↔cluster data-movement rules (never sync by deletion). **Read first
  when setting up a session or writing a new job.**
- `references/autonomy-model.md` — T0–T3 tiers, G1–G5 guardrails, the
  sacct-signature failure-triage table (agent-code vs infrastructure), and the
  `.compute_budget.json` / `.failure_streak.json` ledger shapes + semantics.
- `references/heartbeat-loop.md` — the tick protocol, cron / `/loop` / `/schedule`,
  persistent-Monitor wiring, and how they compose into hands-off overnight runs.
- `references/safety-hooks.md` — PreToolUse hook wiring and the per-job verifier
  pattern (with the reusable `_lib.sh` helper set).
- `references/long-job-patterns.md` — field-tested patterns for multi-day GPU
  campaigns: interleaved sharding with checked
  merges, resumable/idempotent shard jobs, budget estimation from a smoke shard,
  memory-adaptive batching, saturation-before-spending, artifact stamping,
  fp16-store/fp32-load, and the generalized loader trust check. **Read before
  designing any fit or sweep expected to exceed a few GPU-hours.**

## Bundled assets

- `assets/job_template.sbatch` — the canonical, fill-in-the-blanks SLURM job.
- `assets/compute_budget.json` / `assets/failure_streak.json` — starter ledgers with
  the exact schema the guardrails expect.

## Staging into autonomy (don't flip everything on at once)

A new project should **start at T0 + T2 only** — observe, edit, and *recommend* the
next job rather than firing it. Promote to T1 auto-submit once you've watched the
verifier + budget + streak machinery behave correctly across several manual cycles and
you trust the smoke-test gate. The ledgers and the overnight-fires log are what give
you (and the human) the evidence to make that promotion deliberately rather than on
faith.
