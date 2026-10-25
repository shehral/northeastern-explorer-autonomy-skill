# Changelog

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
