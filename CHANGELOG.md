# Changelog

## 2.1.0 — 2026-10-24

Field learnings from October 2026, a harness modernization, human-gated self-evolution
and a job exit-code contract. A minor version: no ledger field is removed, but these
defaults change. Waits are event-driven and the heartbeat is an optional fallback. The
CUDA 13 forward-compat libcuda loads only on nodes whose driver needs it. The template
forwards USR1 to python only on request, exits 75 when it stops early and 4 on an
unusable node, and its traps can no longer turn a successful job into a failure.
Transfers from a Mac default to tar streams.

### Changed: harness modernization

Checked against the tool schemas of a Claude Code 2.1.296 session on 2026-10-24; these
are version-dependent.

- Mental model: a job loop with event-driven waits replaces "three nested loops". A
  clock-driven heartbeat is a fallback for unattended runs and for things nothing can
  notify on (a queue estimate improving, a submit slot freeing). The bounded tick (call
  and word caps, fixed checklist) is now the checklist every wake runs.
- The core wait is a background Bash until-loop (`run_in_background`) that polls
  `sacct` for every terminal state, `PREEMPTED` included, then prints the state and the
  log tail: one notification at the right time. Foreground `sleep` and chained
  `sleep N; cmd` are refused by the harness.
- Monitor is for streams. It times out after 5 min by default and 30 min at most, then
  must be re-armed; there is no `persistent` flag, so the 2.0.0 "persistent-Monitor
  rule" is gone. Filters must match failure states too; `grep` needs `--line-buffered`.
- `ScheduleWakeup` (self-paced `/loop` only, 60-3600 s): this build caches prompts for
  an hour, so the "never 300 s" cache-window rule is obsolete. Choose the delay by what
  you wait for; 1200 s or more for a fallback heartbeat; never wake only to keep the
  cache warm.
- `CronCreate` is session-only (the `durable` flag has no effect), fires only while the
  session is idle, with jitter up to 10% of the period (max 15 min); recurring jobs
  expire after 7 days. Not for live watching.
- `PushNotification` only when the owner has likely walked away and something needs
  them. Workflow-tool orchestration only on the owner's opt-in, with the agent count and
  cost stated first; background subagents read large logs. Re-anchor from the
  continuity docs after every wake and every context summary, not on a clock.
- `references/heartbeat-loop.md` is retitled "Waiting and waking" (filename kept) and
  holds a "Harness assumptions (check at session start)" table. SKILL.md adds the
  session-start check.

### Added: self-evolution (human-gated)

- A private, project-local field log (default `.explorer_field_log.jsonl`): one JSON
  object per observation (date, project_tag, category, observation, evidence,
  confirmation observed|probed|inferred, generalizes, skill_ref, proposed_change).
  Agents append to it and never edit the skill mid-task. Schema in
  `assets/field_log.schema.json`.
- The template prints `NODE_DRIVER node=... gpu=... driver=...`;
  `assets/fold_drivers.py` folds those lines into a driver table JSON.
- `references/evolution.md` and `assets/evolve_workflow.js`: at about 15 new entries or
  monthly, harvest the field logs and continuity docs, diff against the skill (NEW,
  REFINES, CONTRADICTS, outdated), one editor per file on a new branch, a sanitization
  review and a refutation review that runs every snippet, a fix pass, an owner summary.
  It never pushes, merges or syncs the installed copy (proposing is T0/T2, applying T3).

### Changed: exit-code contract

- `0` done (or already done, skipped); `75` stopped early on the wall signal or stop
  file after a checkpoint, resubmit to resume; `4` node or driver incompatible, caught
  before any work (infrastructure: exclude the node); `2` any other pre-flight refusal
  (missing `--gres` or cache, unmarked env, a guard: agent design unless shown to be
  infrastructure); `3` a self-check failed after work started.
- `afternotok` insurance fires on 75, and resume chains use `afterany`. Before, a job
  that stopped cleanly on the stop file exited 0, so its insurance chunk was cancelled
  and the resume never ran.

### Added: field learnings

- **Driver gate (`cluster-setup.md` §3 driver gate).** Drivers differ by node, not
  only by GPU type. Observed: 570.86 on the H200 nodes d4052-d4055 (cu13x works with
  the compat libcuda), 595.58 on the H200 NVL d4077 (breaks with it, works without),
  545.23 on one A100 80 GB node and a V100 node (cu13x fails). The 580 cut-off is
  inferred from NVIDIA's forward-compat rule and the 570 floor is a choice. A per-node
  driver table fed from every job's log, steering with `--exclude`/`--nodelist`, and
  cu12x wheels preferred where packages allow.
- **`sharing` partition (§5 sharing partition):** 1 h wall, 2 running / 4 submitted / 2
  GPUs per user, heterogeneous GPUs (including the only H100 and an H200 NVL), CPU
  starts within about a minute. The default for login-node-sized work and 1-h resumable
  GPU pieces. `courses` and `courses-gpu` notes (V100/P100 only; bill course work to its
  own account).
- **Queue cascade (§5.1 queue cascade):** when a start is more than about an hour away,
  rank every partition x GPU type by estimated finish, filtered on memory, bf16,
  partition wall, node driver and a free running slot. `--test-only` estimates ignore
  your own QOS caps, so check headroom separately.
- GPU inventory: H200 pool size, maintenance drains, H100, 48 GB L40S/L40/A6000 (listed,
  not probed), P100 (sm_60, no bf16), and slowdowns when two GPU jobs share a node (one
  observation).
- Fairshare: one score per user and account across all sessions; how to read `sshare`
  and `sprio`; decay half-life (reported, verify with `scontrol show config`).
- Dependencies: the `afternotok` insurance chunk; every held link spends a submit slot.
- **Serving engines in jobs (§7):** budget model-load start-up (2-18 min observed) and
  load each model once per job; derive a per-job port and verify the server is yours;
  set `gpu_memory_utilization` and `max_model_len` for co-resident engines;
  `VLLM_USE_FLASHINFER_SAMPLER=0`, because the sampler JIT needs `ninja`/`nvcc`.
- Loader gotchas for transformers 5.x saves and training arguments; check argument
  names against the installed source.
- Hub tokens: read-only, fine-grained, mode 600, never in a transcript, none in GPU jobs.
- Allocation space is shared by the group; keep intermediate checkpoints on `/scratch`.
- **Transfer defaults (§8 transfer defaults):** macOS `/usr/bin/rsync` is openrsync and
  fails against Explorer; stream tar. Push into a staging dir and build the tree, and
  verify big files by size and md5, in a CPU job; single-file copies into a tree with
  pending jobs invalidate hash-based smoke stamps.
- Submission plumbing: an `sbatch` error or hang can still create the job (seen up to
  8 min later), so check `squeue` and `sacct` by a name reused on every retry before
  resubmitting (full text in `cluster-setup.md` §6 only); `--export=ALL` hands control
  flags to in-job tests; signals and pipe wrappers; a child sbatch run with `bash`
  ignores its `#SBATCH` lines.
- Keep site values (account, personal paths) in a gitignored `local.env` of `export`
  lines, read by `sbatch` through `SBATCH_ACCOUNT`/`SBATCH_OUTPUT`/`SBATCH_ERROR`.
- Template: the driver gate before any Python, `driver_version` in the `nvidia-smi`
  query, and a stop file as the default checkpoint channel (in 2 of 3 GPU training runs
  python never saw a forwarded USR1). The training loop polls `$STOP_FILE`.
- Verifier checklist: a GPU job must request a GPU; run the test suite on a cluster CPU
  node before the smokes (BLAS kernels, bf16 tolerances, signal masks differ).
- G1: smokes cover production shapes and every expensive path, a dry-run mode and
  pre-GPU guards; smokes get their own output roots.
- Long-job patterns: write required outputs before any optional phase; batch-and-replay
  instead of a live server; per-stage array index maps; smoke per-item rates usually
  overstate time, but a warm-cache smoke understates a cold start-up.
- Ledgers: one ledger of record with one writer, book hand-submitted and
  cancelled-while-pending jobs, timestamps from `date` with the clock labelled.
- G5: a separate `exploratory_agent_code_streak`; a written delegation lets the agent
  reset the counter that tripped after a verified fix (quoted in the ledger). New
  triage rows (driver gate exit 4, cold-load smoke timeout, per-user cap refusal,
  own-uid cancels; too-short agent-chosen walls and missing `--gres` count as agent
  code).
- The laptop is the weak link (sleep, scratchpad pruning); pre-queue dependent work on
  the cluster. zsh history modifiers after a colon.
- Baseline software on the H200 nodes: Slurm 23.11.6, bash 5.1.8, kernel 5.14.0-362.13.1.

### Changed: field learnings

- The CUDA 13 forward-compat export is conditional (driver gate above); the gate exits
  4 (infrastructure) on a driver below the floor and 2 when no driver is visible.
- Template traps end in commands that cannot fail, and the GPU-utilisation sampler runs
  with `set +e`. Before, a dead sampler made the EXIT trap's `kill` fail, turning a
  successful job into exit 1 (FAILED, `afterok` children never fire); reproduced locally.
- The login node SIGKILLs even light work (a few-MB rsync, a JSON read, one import), not
  only memory-heavy work. Keep it to `squeue`/`sacct`/`sinfo`/`cat`/`grep`/light `git`;
  run the rest as a short CPU job on `sharing`. CPU jobs (pre-cache, re-aggregation,
  tests) go through `sbatch` on `sharing` (within 1 h) or `short`, not `srun -p short`:
  `short` has queued for days.
- MaxSubmitPU counts dependency-held jobs and nested submissions; a job submitting its
  own successor can be refused after its GPU work is done. A `--test-only` refused with
  `QOSMaxSubmitJobPerUserLimit` means "retry when a slot frees", not NOT READY.
- IdleBot: `gpu-logs` failed in one October check; check with
  `srun --jobid=<id> --overlap nvidia-smi` and the email. A CPU-heavy middle phase costs
  submit slots when split into chained jobs.
- Sweeps split across GPU types only when no measurement compares outputs across jobs.
- G4: `gpu` + `gpu-short` allow 144 GPU-h/day per user, and `sharing` and
  `courses-gpu` add more (check `sacctmgr show qos`); slots and wall waves set the pace.
- Layered venvs: `--no-deps` plus a constraints file when the base env is already
  inconsistent; exempt `pip`/`setuptools`/`wheel` from the shadowing guard.
- The pin-rationale example no longer assumes every node runs a 12.8 driver.
- Idempotent submission keys the duplicate check on the code tree as well.
- The frontmatter description is under 1000 characters; the full GPU list moved to the
  README and the template's `--gres` comment.

### Fixed (review of this release)

- USR1 forwarding killed scripts that only poll the stop file (python's default SIGUSR1
  action terminates the process); forwarding is now opt-in (`FORWARD_USR1=1`), also for
  libraries that trap only SIGINT/SIGTERM.
- The §3 driver-gate snippet aborted silently under `set -e -o pipefail` when no GPU
  was visible; there is now one copy of the gate, in the template, reading the driver
  with `|| true`.
- Exit 2 meant both infrastructure (driver) and agent design (missing `--gres`); split
  into exit 4 and exit 2 (contract above).
- Overlapping triage rows: a cold-load smoke TIMEOUT is infrastructure only when its
  wall met the §7 minimum. A delegated rollback reset clears the counter that tripped.
- Snippets: the many-jobs watcher matched array ids as a regex and had no ssh timeout;
  `find | grep -q` misfired under `pipefail`; the dedupe tag must be reused on every
  retry of a job; the queue-cascade loop is bash-only, takes gres names from `sinfo`,
  filters nodes by driver and guards an unset account; the `--gres` verifier accepts
  `--gres X` and `--gpus*` and runs under bash 3.2; `sacct -D` budget rows are keyed by
  `(JobIDRaw, Start)`; template placeholders are quoted so `bash -n` passes.
- Overclaims corrected: when daily caps bound, the count behind one G5 rollback, the
  scratch-checkpoint totals, where measured model loads read from, the two-job slowdown
  (one observation), the decay half-life (reported), and the vLLM memory example.
- G1 smoke length is "about 60 s of work after the model loads", not a 60 s wall.
- Sanitization: figures that described one group or one study (allocation size, a
  fairshare example, model sizes) are generic; the loader example names no model
  family.
- README synced (GPU list, pre-cache wording, version, file list); the `sbatch`-error
  guidance is kept in full only in `cluster-setup.md` §6.
- The template remaps a program's own exits 2, 3 and 4 to 1 (log line `reserved
  contract code, remapped to 1`) unless `PROGRAM_FOLLOWS_EXIT_CONTRACT=1`, so `sacct`
  never shows a contract code the pre-flight did not raise.
- New exit-4 branch: a GPU allocated (the job's AllocTRES, then Slurm's GPU variables)
  but the driver unreadable is a node problem, not a missing `--gres` (exit 2).
- The template writes the stop file from a USR1 trap set right after `STOP_FILE`, so a
  signal that lands before the program starts still ends in a clean 75.
- The waiting recipes: `sacct -P` (an unparsed State column truncates OUT_OF_MEMORY),
  every terminal state, a cursor so re-armed wakers miss no event, and requeued jobs
  deduplicated on (id, state).
- `fold_drivers.py` orders readings by full timestamp and is idempotent on reruns; the
  evolve workflow's Fix and refutation agents act on every editor's cross-file list.
- Harness facts are labelled by build and by where a tool exists: `ScheduleWakeup` only
  in a main session running a self-paced `/loop`, the Workflow tool only in a main
  session with the owner's opt-in; the chained-`sleep` refusal is observed in 2.1.296.

### Not yet done

- The evals have not been regraded against 2.0.0 or 2.1.0, and eval ids 3, 4 and 5 have
  never been graded.
- Untested on the cluster: the exit-4 branch of the driver gate and drivers 546-569 and
  580-594 (unobserved). A 2-min probe on any such node settles the cut-offs; log it in
  the field log.
- Open: why the forwarded USR1 was lost on GPU runs; whether `sbatch --test-only` rejects
  a request over the per-job GPU cap; whether site policy allows `gpu-interactive`
  through `sbatch`; drivers, bf16 and gres names of the H100, L40S, L40 and A6000 nodes;
  the driver of the A100 node where the compat recipe was reported working.
- Answered from 2.0.0 (field report): Slurm does not create a missing `--output` directory.

## 2.0.0 — 2026-10-11

A major version because these behaviors change: GPU jobs now pick a GPU type per job
instead of always requesting an H200, pre-caching moves from the login node into a CPU
job, GPU jobs unset the proxy they inherit, the job Monitor polls the scheduler instead
of the log, and a durable tick runs as a Desktop scheduled task rather than a cloud
routine.

### Merged: the installed 1.2.0 line and the repo's 1.0.0 line

The installed copy (1.2.0) had been forked from the initial commit, so it lacked the
June field learnings (L1-L5), and the repo lacked its September additions. Both are
now merged three-way with the initial commit as the base:

- New `references/long-job-patterns.md` (interleaved sharding, resumable shard jobs,
  budget estimation, memory-adaptive batch, saturation check, artifact stamping,
  fp16-store/fp32-load, loader trust check), with private study details and numbers
  removed.
- `cluster-setup.md`: the silent login-node kill, QOS caps and `afterany` chaining,
  pin-rationale notes, the engine model-registry check, and a new §8 on data
  movement. The troubleshooting section is now §9.
- `autonomy-model.md`: sacct-signature triage table. `heartbeat-loop.md`: overnight
  auth and macOS-side gotchas. Template: `expandable_segments`, no HF progress bars.

### Added

- **GPU-type selection (`cluster-setup.md` §5.1).** Size from the smoke's measured
  peak memory; keep only types the env's torch build supports
  (`torch.cuda.get_arch_list()`); rank types by free GPUs per node and pending demand;
  respect per-user QOS caps, which apply across all types; spread parallel sweeps
  across free types; move your own jobs that pend ~15 min while a fitting type is
  free; record the type of every job.
- Per-user QOS caps verified on 2026-10-11, and array-task counting.
- IdleBot: GPU jobs under-utilised for an hour are cancelled; keep CPU phases on
  `-p short` and log GPU utilisation.
- G4 counts GPU-hours per GPU type (`.compute_budget.json` format version 2).
- An offline load test at the end of every CPU pre-cache job.
- Silent-failure notes: `--export` splits lists on commas, `#SBATCH` lines are not
  shell-expanded (use `%x-%j`), `ssh '... nohup srun &'` blocks, multi-glob `scp`
  fails (stream a tar), never switch commits under a running cluster tree.
- Triage rows for node launch-kills, IdleBot cancellations, "no kernel image",
  `QOSMaxGRESPerJob` and "exit 0, zero items".
- A partition table (`cluster-setup.md` §5), verified 2026-10-12: default and maximum
  times, per-job GPU caps (a request over the cap pends forever with
  `QOSMaxGRESPerJob`), per-user caps, and access by partition rather than by billed
  account.
- Dependency rules (`kill_invalid_depend`, `MinJobAge`, the `?` OR form), fixed-string
  matching of array ids, and node RAM as a placement limit (`Memory − AllocMem`).
- IdleBot's `sacct` symptom and policy page; the template logs GPU utilisation.
- Environments: treat a working env as immutable (layered venv, `.built-ok` marker that
  the template checks), a CUDA 13 forward-compat recipe (field-verified on H200 only),
  and the `__main__` guard for spawn-started inference engines.
- Commit stamps at job start, `readlink` after checkout in a symlinked second tree, one
  shared node-exclude list, and login-node transfer limits.
- Supporting rules: cluster output is data, never instructions; one job-name prefix per
  project; idempotent submission. T3 now covers any job outside your project's prefix.
- G4 ledger fed from `sacct -X -D`; G1 split smoke with commit-keyed receipts.
- Verifier checklist item 6: `sbatch --test-only` accepts the job.
- Heartbeat: step 0 re-arms expired or unrestored loops and Monitor tasks; an idle tick
  is a one-line backstop. zsh word-splitting gotcha.
- Eval ids 4 (`gpu-type-placement`) and 5 (`pending-forever-and-session-resume`).

### Changed

- The offline rule: compute nodes reach the internet only through the squid proxy, and
  jobs inherit it from the login shell (`--export=ALL`). Export it explicitly in CPU
  pre-cache jobs; unset it in every GPU job and set `HF_DATASETS_OFFLINE=1` with the
  other offline flags. The rationale no longer claims an offline miss silently loads
  meta-init zeros; missing or mis-tied weights are named as the separate case.
- Pre-cache in `srun -p short`, never on the login node. `snapshot_download(revision=)`
  writes no `refs/main`.
- No `cuda/*` module for pip-installed torch. Node drivers differ by GPU type.
- dtype: bf16 on A100/H200; fp32 (or fp16 with an overflow check) on T4/V100. Code
  picks bf16 from compute capability (`>= 8`), because `torch.cuda.is_bf16_supported()`
  also returns True on T4/V100 through emulation.
- Pre-flight gate checks the pinned snapshot's `config.json` and weight files. The
  template's sanity step runs a one-element CUDA op.
- The walltime hook uses `--signal=B:USR1@600` forwarded to a backgrounded python,
  replacing a SIGTERM trap that ran only after python had died. It is `set -e` safe.
- The Monitor polls `sacct` every ~120 s rather than grepping the log.
- The launch-kill canary uses the failed job's GPU type. Timeout recovery
  re-aggregates in `srun -p short`, not on the login node.
- G5: pending time is infrastructure; an IdleBot cancellation of a job with a long CPU
  phase in a GPU allocation counts as agent design. An OOM follows the triage row, so
  "node OOM" is no longer listed as infrastructure.
- A pending job's TimeLimit can be raised by its owner up to the partition maximum; a
  running job's only by an admin. Size `--time` at submit.
- Output dir on resubmit: resume in place after a timeout or node failure; use a fresh
  dir after a code fix.
- Connection goes through an `SSH_ALIAS` Customization Point. The proxy address is a
  `PROXY` placeholder.
- Scheduling: cloud `/schedule` routines cannot reach the cluster; a durable local tick
  is a Desktop scheduled task. Recurring session tasks expire after 7 days, and Monitor
  tasks and self-paced loops are not restored on resume.
- Evals 0-4 updated to the corrected behavior. BENCHMARK marks every row for
  regrading.

### Removed

- The claim that short-partition slots are reserved for a higher-priority partition.
  On 2026-10-11 `PreemptMode=OFF`, there were no reservations, and `gpu` and
  `gpu-short` had equal `PriorityTier`.

### Not yet done

- The evals have not been regraded against 2.0.0, and eval ids 3, 4 and 5 have never
  been graded.
- Open: whether `sbatch` (verified only for `srun`) creates a missing `--output`
  directory; whether `sbatch --test-only` rejects a request over the per-job GPU cap;
  whether site policy allows `gpu-interactive` through `sbatch`; the CUDA 13
  forward-compat recipe on A100 and T4 nodes.
