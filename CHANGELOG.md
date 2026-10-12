# Changelog

## 2.0.0 — 2026-10-11

A major version because three behaviors change: GPU jobs now pick a GPU type per job
instead of always requesting an H200, pre-caching moves from the login node into a CPU
job, and the job Monitor polls the scheduler instead of the log.

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
- Triage rows for node launch-kills, IdleBot cancellations and "no kernel image".
- Eval id 4 (`gpu-type-placement`).

### Changed

- The offline rule: compute nodes reach the internet only through a proxy the job
  exports itself. Use it only in CPU pre-cache jobs and keep every GPU job offline. The
  rationale no longer claims an offline miss silently loads meta-init zeros; missing
  or mis-tied weights are named as the separate case.
- Pre-cache in `srun -p short`, never on the login node. `snapshot_download(revision=)`
  writes no `refs/main`.
- No `cuda/*` module for pip-installed torch. Node drivers differ by GPU type.
- dtype: bf16 on A100/H200; fp32 (or fp16 with an overflow check) on T4/V100.
- Pre-flight gate checks the pinned snapshot's `config.json` and weight files. The
  template's sanity step runs a one-element CUDA op.
- The walltime hook uses `--signal=B:USR1@600` forwarded to a backgrounded python,
  replacing a SIGTERM trap that ran only after python had died. It is `set -e` safe.
- The Monitor polls `sacct` every ~120 s rather than grepping the log.
- The launch-kill canary uses the failed job's GPU type. Timeout recovery
  re-aggregates in `srun -p short`, not on the login node.
- G5: pending time is infrastructure; an IdleBot cancellation of a job with a long CPU
  phase in a GPU allocation counts as agent design.
- Output dir on resubmit: resume in place after a timeout or node failure; use a fresh
  dir after a code fix.
- Connection goes through an `SSH_ALIAS` Customization Point. The proxy address is a
  `PROXY` placeholder.
- Evals 0-3 updated to the corrected behavior. BENCHMARK marks every row for regrading.

### Removed

- The claim that short-partition slots are reserved for a higher-priority partition.
  On 2026-10-11 `PreemptMode=OFF`, there were no reservations, and `gpu` and
  `gpu-short` had equal `PriorityTier`.

### Not yet done

- The evals have not been regraded against 2.0.0, and eval ids 3 and 4 have never
  been graded.
- Not verified, so not adopted: that `scontrol update JobId=<id> TimeLimit=` can
  raise a pending job's limit.
