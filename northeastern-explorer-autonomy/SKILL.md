---
name: northeastern-explorer-autonomy
description: >-
  Operate the Northeastern Explorer HPC cluster (SLURM, login.explorer.northeastern.edu)
  autonomously from a Claude Code session: connect, set up the pinned environment, pick
  the GPU type each job fits and can start on soonest, submit smoke-tested jobs through
  a verified hook layer, wait on them event-driven, and run unattended overnight under
  compute-budget and failure-streak guardrails. Use whenever the user wants to run,
  submit, monitor, wait on or schedule jobs on Explorer, set up autonomous or overnight
  HPC runs, spread jobs across GPU types to avoid long queues, recover a stalled
  autonomous session, or onboard a collaborator. Trigger even on "submit this to the
  cluster", "kick off an overnight run", "watch this SLURM job", "set up my Explorer
  environment" or "let Claude run experiments on its own": the guardrails and
  connection mechanics here prevent the expensive footguns (silent meta-init model
  collapse, runaway GPU spend, bug-loops).
metadata:
  version: 2.1.0
  tags: [hpc, slurm, explorer, northeastern, autonomy, gpu-selection, event-driven, guardrails]
---

# Operating Northeastern Explorer Autonomously with Claude Code

This skill turns a Claude Code session into a disciplined autonomous operator on the
**Northeastern Explorer** cluster: it connects, runs the right environment, submits
**only smoke-tested, verified** SLURM jobs, waits for them to finish, integrates
results, and decides what to run next, all under guardrails that make unattended
operation safe rather than reckless.

It is written to be **reused by anyone** on Explorer. The cluster-specific values
(username, allocation, conda env, model) live in one **Customization Points** block;
everything else (the autonomy model, the wait-and-wake pattern, the safety layer) is
portable.

## When to reach for this

Use it when the work is "run things on the cluster, possibly while the human is away."
Submitting a job, waiting on a long job, running unattended overnight, recovering a
session that died, or handing the pattern to a teammate. If the task is a one-off local
script with no cluster involved, you don't need this.

## The mental model: a job loop with event-driven waits

```
Session start  →  connect, check the harness assumptions table, re-anchor from the
                  continuity docs (RESUME file, ledgers)
  └─ Decision  →  for each candidate action, classify Tier T0-T3,
                  check guardrails G1-G5, act or escalate
       └─ Job loop → smoke-test → verify → sbatch → background wait on sacct
                     (one wake at a terminal state) → intake result →
                     update ledgers → pick next
Fallback heartbeat (optional) → a slow clock tick, only for unattended runs and for
                     things nothing can notify on; a Desktop task when it must be durable
```

The harness wakes the session when a background wait exits, so the default is to wait
on events, not on a clock. Every wake (and every context summary) runs the same bounded
checklist: re-anchor from the continuity docs, intake, guardrails, maybe submit, log.
**Tiered autonomy**, **guardrails**, **a verified submit path** and **event-driven
waits** together let a session make hundreds of cheap moves and a handful of expensive
ones without a human in the loop, and never modify its own rules unilaterally.

## Customization Points — EDIT THESE FIRST for your account

Everything below assumes these values. **Replace every `<…>` placeholder with your own
account's values before first use**; the rest of the skill and the bundled
`assets/job_template.sbatch` reference them. Values shown without brackets
(module versions, partition names) are Explorer-wide and usually correct as-is.

| Point | Placeholder → set to yours | What it is |
|---|---|---|
| `SSH_ALIAS` | `<alias>` | a `Host <alias>` block in `~/.ssh/config` with `HostName login.explorer.northeastern.edu`, `User <user>`, an authorized `IdentityFile`, `ControlMaster auto`, `ControlPath`, `ControlPersist 8h`. Every ssh/scp/rsync goes through the alias. |
| `ALLOC` | `/projects/<group>/<user>` | your persistent project allocation (NOT `/scratch`, NOT `/home`) |
| `SLURM_ACCOUNT` | `<slurm-account>` | the `--account` your allocation bills to. Keep it, and every personal path, out of shared job files (below) |
| `CONDA_ENV` | `$ALLOC/conda_envs/<conda-env>` | your pinned conda env path; write `<env>/.built-ok` when a build finishes (the template refuses an unmarked env) |
| `MODULES` | `miniconda3/24.11.1` | pinned module versions (Explorer-wide). No `cuda/*` module for pip-installed torch: the wheel carries its own CUDA runtime (`references/cluster-setup.md` §3) |
| `HF_CACHE` | `$ALLOC/hf_cache` | model cache (pre-populated by a CPU pre-cache job, never on the login node) |
| `PROXY` | `<squid-proxy-ip>:3128` | the cluster's squid proxy (from the research-computing docs). Jobs inherit it from the login shell: export it explicitly in CPU pre-cache jobs, **unset** it in GPU jobs |
| `MODEL_ID` / `REVISION` | `<org>/<model>` @ `<revision>` | your pinned model + commit (if you use a model hub) |
| `DAILY_GPU_BUDGET` | `24` GPU-hours | G4 soft cap per day, all types; the ledger records hours per GPU type (tune). Per-user QOS caps bound what you can run at all: `gpu` + `gpu-short` allow 144 GPU-h/day (4 × 24 + 2 × 24), and `sharing` (2 GPUs) or, for course accounts, `courses-gpu` add more; check `sacctmgr show qos` (MaxTRESPU). Queue slots set the real pace (`references/autonomy-model.md` G4) |
| `FAIL_STREAK_MAX` | `3` | G5 consecutive-agent-bug auto-rollback threshold (tune) |

**Keep site values out of shared job files.** For a job file you commit or share, put
`--account` and personal paths in a gitignored `local.env` as `export` lines
(`export SBATCH_ACCOUNT=<slurm-account>`; or wrap a plain file in `set -a`), commit a
`local.env.example`, and submit with `. ./local.env && sbatch job.sbatch`: `sbatch`
reads exported `SBATCH_ACCOUNT`, `SBATCH_OUTPUT` and `SBATCH_ERROR`, and the job sources
`${SLURM_SUBMIT_DIR}/local.env` and stops on a missing value
(`${VAR:?set VAR in local.env}`). Slurm does not create a missing log directory, so
create it first. A gitignored file never travels with a code sync: create `local.env`
on the cluster before the first sync that drops the hard-coded values, or jobs silently
lose the account and proxy. The template's header comment shows the lines.

If you are a **new user adapting this**, also rewrite any model-specific loader fix.
The `lm_head` tie workaround in `references/cluster-setup.md` §7 is a model-specific
*example* (one checkpoint that omits `lm_head.weight` and relies on `tie_word_embeddings`); yours may differ or be unnecessary, so
delete or replace it. And if you are on a **different cluster**, see the README's
"Adapting to a different cluster" section: the autonomy model, waits and guardrails are
portable; only the connection/env/partition details change.

## Quick start (a single autonomous run, start to finish)

1. **Start the session.** Check the harness assumptions table
   (`references/heartbeat-loop.md`) against your tool list and log any drift to the
   field log (Self-evolution, below). Re-anchor from the project's continuity docs
   (RESUME file, ledgers) and `squeue -u <user>` before acting.
2. **Connect & verify the session is warm: try key-auth via the alias first.** With an
   SSH-config `Host <alias>` block (authorized `IdentityFile` + `ControlMaster auto` +
   `ControlPersist`), a non-interactive
   `ssh -o BatchMode=yes <alias> 'squeue -u <user>; sinfo -p gpu'` authenticates **by
   key with no MFA prompt** and opens the persistent master socket in the same call.
   Only if the key is **not** authorized fall back to an interactive `ssh <alias>` to
   clear Duo (reused for 8h). **Always go through the `<alias>`, never the raw
   hostname**: that bypasses the `IdentityFile`/`ControlMaster` config and fails with a
   GSSAPI `Permission denied`. See `references/cluster-setup.md` §1.
3. **Pre-cache the model in a CPU job via `sbatch` (`sharing` if it fits in 1 h,
   `short` beyond), never on the login node**, then prove an offline load works in that
   same CPU job before any GPU time is spent (GPU jobs run offline, see the rule below;
   `references/cluster-setup.md` §2). The login node SIGKILLs work silently, even light
   Python, imports and small rsyncs: keep it to `squeue`/`sacct`/`sinfo`/`cat`/`grep`/
   light `git`, and run anything else as a short CPU job on `sharing`
   (`references/cluster-setup.md` §1, §5 sharing partition).
4. **Smoke-test the experiment locally or on `gpu-short`** on one input **before** any
   `gpu` submission (Guardrail G1; ~5000:1 cost asymmetry). Validate the **actual
   artifact you'll submit**: include at least one real `sbatch` smoke, not only an
   interactive `srun`, which streams stdout over SSH and writes no output file, so it
   skips the failures (output-file creation, node prolog/launch) that only appear under
   `sbatch`. For anything expected to run more than a few GPU-hours, shape it with
   `references/long-job-patterns.md` first: interleaved shards, resumable checkpoints
   and a saturation check routinely halve real spend.
5. **Choose the partition and GPU type, then compose the sbatch.** Never hard-code one
   type: pick the one that fits the smoke's measured memory, that your env's torch build
   and the node's driver support, and that finishes soonest given free GPUs, partition
   walls and your QOS caps (`references/cluster-setup.md` §5.1 queue cascade). Split a
   sweep across types only when no measurement compares outputs across jobs; a
   fitted-on-one-set, read-on-another contrast runs entirely on one type (fallback: move
   the whole set to a same-architecture type after a smoke there, never half of it).
   Compose from `assets/job_template.sbatch`: it encodes the pinned module, the
   PATH-export env activation, the driver gate (§3), mandatory offline flags, cache
   paths, determinism, a pre-flight model-cache check, a stop-file checkpoint signal,
   the exit-code contract below, and a `NODE_DRIVER` log line. Serving-engine jobs
   (vLLM and friends) have their own start-up, port and memory rules
   (`references/cluster-setup.md` §7 Serving engines in jobs).
6. **Verify, then submit.** A PreToolUse hook should run a per-job verifier before the
   `sbatch` lands (`references/safety-hooks.md`). Submitting fires a job ID.
7. **Wait on the scheduler, not the log, with one background wait.** Run this
   until-loop through the Bash tool with `run_in_background` (an exact copy of the
   tested loop in `references/heartbeat-loop.md`; keep the two identical). The harness
   re-invokes the session once, when it exits: when `sacct` lists the job and every row
   (array tasks too) is in a terminal state, or with `SSH_DOWN` after 5 failed ssh polls
   in a row. Slurm never writes COMPLETED or FAILED into the job log, and a launch-kill
   writes no log. Foreground `sleep` is blocked (Bash tool description), and a chained
   `sleep N; cmd` was refused when tried (observed in 2.1.296).

   ```bash
   J=<jobid>; fails=0
   T='^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)'
   rsh() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 \
               -o ServerAliveCountMax=2 <alias> "$@"; }
   while :; do
     if s=$(rsh "sacct -j $J -X -n -P -o JobID,State"); then
       fails=0   # done when sacct lists the job and every row (array tasks too) is terminal
       printf '%s\n' "$s" | awk -F'|' -v t="$T" 'NF {n++; if ($2 !~ t) busy = 1}
         END {exit !(n && !busy)}' && break
     else
       fails=$((fails + 1))
       [ "$fails" -ge 5 ] && { echo "SSH_DOWN: job $J unknown after 5 polls"; exit 3; }
     fi
     sleep 120
   done
   printf '%s\n' "$s"
   rsh "tail -n 30 <log-dir>/<jobname>-$J.out <log-dir>/<jobname>-$J.err"
   ```

   Keep `-P`: without it `sacct` truncates `OUT_OF_MEMORY` to `OUT_OF_ME+` and an OOM
   job never ends the wait. Its exit 3 is a watcher code, not a job exit code.

   Use the Monitor tool only for streams (progress lines, several events); it ends
   after at most 30 min and must be re-armed. Many jobs: one watcher, one waker
   (`references/heartbeat-loop.md`).
8. **On wake, intake the result**, update your state ledgers, record success/failure in
   the failure streak, and let the decision loop pick the next action.
9. **If running unattended,** add a fallback heartbeat for what nothing can notify on
   (a queue estimate improving, a submit slot freeing) and in case a wait dies: a
   self-paced `/loop` (`ScheduleWakeup`, main session only, at 1200 s or more as a
   fallback) or a `CronCreate` cadence off minute :00/:30. Both are session-only and
   recurring crons expire after 7 days; a Desktop scheduled task is the durable local tick (cloud
   routines cannot reach the cluster). Pre-queue dependent work on the cluster, and send
   a `PushNotification` only when the owner has likely walked away and something needs
   them. See `references/heartbeat-loop.md`.

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

Pre-caching (step 3) and offline-mode (this rule) are an inseparable pair: neither
works without the other. Some libraries call the hub API even for cached files and
crash under offline mode, which is why step 3 ends with an offline load test on a CPU
node. Full reasoning in `references/cluster-setup.md` §2 and §4.

### Field-tested gotchas

The first two are **infrastructure** failures: neither increments the G5 streak.
Diagnose before "fixing" the wrong layer; full recipes in `references/cluster-setup.md`
§6 and §9.

- **Node launch-kills.** Symptom: an `sbatch` job dies in 1–5s with `State=FAILED`, the
  `.batch` step `CANCELLED`, a signal-style `ExitCode` (e.g. `0:53`), the `.extern` step
  `COMPLETED`, and **no output log** (killed before the script's first line). This is a
  bad **node**, not a code bug or preemption. Diagnostic ladder: get the node from
  `sacct` `NodeList`; run a **minimal canary** (no model, tiny mem, short time, trivial
  payload, **same GPU type** as the failed job); if it succeeds on a *different* node,
  the original node(s) are bad. Rule out systemic causes (the job's fail reason, QOS
  limits, output-dir writability, maintenance reservations) so you don't mis-fix. Fix:
  add `#SBATCH --exclude=<bad-nodes>` and resubmit.
- **Short-partition timeout of finished work.** A short-walltime partition can
  **TIMEOUT a job that finished its real work** but not the final aggregation/write.
  Defend by writing per-item outputs as you go (idempotent skip-existing) and aggregating
  from them; if you time out near the end, re-aggregate CPU-only from the per-item files
  in a CPU job (`sharing` for login-node-sized work, `short` for longer; `short` has
  queued for days), **never on the login node**, with no GPU re-run. Size `--time` from
  the smoke's max per-item time and don't undercut it.
- **Submission plumbing that fails silently.** `sbatch --export=ALL,IDX=1,2,3` splits on
  the commas, so the job sees `IDX=1` and still exits 0: pass lists through the
  environment (`IDX=1,2,3 sbatch ...`) and assert the logged list equals the requested
  one. `ssh <alias> 'nohup srun ... &'` holds the ssh call open until srun ends; use
  `sbatch` for anything longer than a few seconds. An `sbatch` error or hang does not
  mean no job: check by a unique name before resubmitting (`references/cluster-setup.md`
  §6).
- **Transfers from a Mac.** `/usr/bin/rsync` on current macOS is openrsync and fails
  against Explorer with `unexpected end of file`; stream tar over ssh instead
  (`references/cluster-setup.md` §8 transfer defaults).

## The autonomy model (how to decide what to do without asking)

Classify every candidate action into a **tier**, and gate the auto-acting tier behind
**five guardrails**. This is the heart of safe unattended operation; full detail in
`references/autonomy-model.md`.

- **T0 — always allowed:** reads, greps, audits, local edits/commits, smoke tests,
  state-file updates, memory and field-log writes, dispatching read-only subagents.
- **T1 — auto-submit allowed (logged; surfaced only if anomalous):** an HPC job
  ≤ a walltime cap, **only if all five guardrails pass**.
- **T2 — do it, but surface in the next report:** new experiment designs, large
  code changes, new-checkpoint/cross-family runs, a proposed skill change.
- **T3 — never auto; escalate to the human:** destructive ops (`rm -rf`, force-push;
  cancelling, holding or modifying any job outside your project's job-name prefix,
  even under your own user), external comms, **modifying your own protocol, this skill
  or the user's CLAUDE.md**, touching restricted/production artifacts.

The **five guardrails** that gate T1:

| | Guardrail | Threshold |
|---|---|---|
| **G1** | Smoke-test gate | a short single-input run passed **via at least one real `sbatch`** (not only `srun`, which writes no output file and skips launch/prolog failures), or script unchanged from a prior success |
| **G2** | Pre-registration | falsification criteria written **before** the run |
| **G3** | Info-gain | experiment discriminates between live hypotheses (≥ 0.3 if you score it) |
| **G4** | Daily compute budget | under `DAILY_GPU_BUDGET` GPU-h today, recorded per GPU type (`.compute_budget.json`) |
| **G5** | Failure-streak rollback | not under auto-rollback (`< FAIL_STREAK_MAX` consecutive **agent-code** bugs) |

**Job exit codes** (the template's contract): `0` done (or already done, skipped);
`75` stopped early on the wall signal or stop file after a checkpoint, resubmit to
resume (`afternotok` insurance fires on it; resume chains use `afterany`); `4` node or
driver incompatible, caught before any work: infrastructure, exclude the node; `2` any
other pre-flight refusal (missing `--gres` or cache, unmarked env, a guard): agent
design unless shown to be infrastructure; `3` a self-check failed after work started;
anything else is the program's own status.

G5's subtlety: only new agent-written-code bugs increment the streak. **Infrastructure
failures don't count:** HF timeouts, scp races, connection/auth failures, node
launch-kills, NODE_FAIL, exit 4 from the driver gate, a walltime timeout (or exit 75) of
resumable code whose checkpoints were advancing, a smoke TIMEOUT during a cold model
load when its wall met the §7 minimum, a submit refused by the per-user cap, and pending
time. **Counts as agent code:** an IdleBot cancellation of a job the agent designed with
a long CPU-only phase inside a GPU allocation, a TIMEOUT from an agent-chosen wall too
short for a non-resumable job, and a GPU job submitted without `--gres` (all job-design
bugs). Keep a second counter for hand-submitted or exploratory jobs, so one pipeline's
success cannot hide another's failures. The human resets a rollback, or the agent does
under a written delegation it quotes in the ledger. This prevents bug-loops, the most
expensive autonomous failure mode, without punishing the agent for a flaky cluster.
See the triage table in `references/autonomy-model.md`.

## The safety layer (make the verified path the only path)

Autonomy is only safe if the *submit* action is hard to get wrong. Wire two things
(details + copy-paste configs in `references/safety-hooks.md`):

1. **PreToolUse hooks** that intercept `sbatch`/`Edit`/`commit` and run a check
   *before* the call lands, blocking on failure so you see it inline.
2. **Per-job verifiers** (`verify_<job>.sh`) that enforce, before any GPU-hour burns:
   the script + sbatch exist, `--smoke` ran clean, offline flags are set,
   pre-registration constant present, and (the end-to-end wiring check) that a
   with-intervention run produces **different output** than without. A registered
   hook + a real bake-off both confirmed this catches the silent "the hook never
   actually fired" bug that otherwise looks like a real (null) result.

## Waiting and waking (hours-long and overnight runs)

Events by default (step 7), a clock only as a fallback (step 9). Background tasks,
Monitors and session crons die with the session; cluster jobs do not, so pre-queue
dependent work on the cluster. Every wake runs a bounded checklist (**≤25 tool calls,
≤300-word report**): re-anchor → audit consistency → intake landed jobs → update
ledgers → guardrails → maybe submit → log → report. Long sessions get summarised and
only the continuity docs survive, so re-anchor after every wake and every summary. Let
background subagents read large logs and return conclusions. Details, including the
harness assumptions table, are in `references/heartbeat-loop.md`.

## Self-evolution (human-gated)

When the cluster or the harness disagrees with or adds to this skill (an untriaged
failure, a node driver or probe result, a queue trick that worked, a harness tool
behaving differently from the assumptions table), append one JSON line to a private,
project-local field log (default `.explorer_field_log.jsonl`, gitignored). Never edit
the skill mid-task: proposing a change is T0/T2, applying it is T3. The template's
`NODE_DRIVER node=... gpu=... driver=...` line feeds `assets/fold_drivers.py`, which
folds job logs into a driver table. At about 15 new entries or monthly, run the evolve
procedure in `references/evolution.md` (runnable as `assets/evolve_workflow.js` with the
Workflow tool, in a main session on the owner's opt-in): harvest, diff against the skill, edit on a new
branch, review, fix, summarise. It never pushes, merges or syncs the installed copy.

## Reference files (read the one you need)

- `references/cluster-setup.md`: connection, allocation/storage, env activation and
  the per-node driver gate, offline rules, partitions (including `sharing`) and the
  queue cascade, the canonical sbatch anatomy, model-loader gotchas, serving engines in
  jobs, and the laptop↔cluster data-movement rules (tar streams; never sync by
  deletion). **Read first when setting up a session or writing a new job.**
- `references/autonomy-model.md`: T0–T3 tiers, G1–G5 guardrails, the sacct-signature
  failure-triage table (agent-code vs infrastructure), and the `.compute_budget.json` /
  `.failure_streak.json` ledger shapes + semantics.
- `references/heartbeat-loop.md` ("Waiting and waking"): the harness assumptions table,
  background waits, Monitor, `/loop`/`ScheduleWakeup`, cron and Desktop tasks,
  notifications, the bounded wake checklist, overnight auth and session recovery.
- `references/safety-hooks.md`: PreToolUse hook wiring and the per-job verifier
  pattern (with the reusable `_lib.sh` helper set).
- `references/long-job-patterns.md`: field-tested patterns for multi-day GPU
  campaigns: interleaved sharding with checked merges, resumable/idempotent shard jobs,
  budget estimation from a smoke shard, memory-adaptive batching,
  saturation-before-spending, artifact stamping, fp16-store/fp32-load, and the
  generalized loader trust check. **Read before designing any fit or sweep expected to
  exceed a few GPU-hours.**
- `references/evolution.md`: the field-log schema and the human-gated evolve
  procedure.

## Bundled assets

- `assets/job_template.sbatch`: the canonical, fill-in-the-blanks SLURM job.
- `assets/compute_budget.json` / `assets/failure_streak.json`: starter ledgers with
  the exact schema the guardrails expect.
- `assets/fold_drivers.py`: folds `NODE_DRIVER` lines from job logs into a driver
  table JSON.
- `assets/field_log.schema.json`: the JSON schema of one field-log line.
- `assets/evolve_workflow.js`: the evolve procedure as a Workflow script.

## Staging into autonomy (don't flip everything on at once)

A new project should **start at T0 + T2 only**: observe, edit, and *recommend* the
next job rather than firing it. Promote to T1 auto-submit once you've watched the
verifier + budget + streak machinery behave correctly across several manual cycles and
you trust the smoke-test gate. The ledgers and the overnight-fires log are what give
you (and the human) the evidence to make that promotion deliberately rather than on
faith.
