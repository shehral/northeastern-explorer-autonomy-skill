# Cluster setup — Northeastern Explorer

Connection, storage, environment activation, the offline rule, the canonical sbatch
anatomy, and loader gotchas. Replace the `Customization Points` values from `SKILL.md`
(`SSH_ALIAS`, `ALLOC`, `SLURM_ACCOUNT`, `CONDA_ENV`, `MODULES`, `HF_CACHE`,
`MODEL_ID`/`REVISION`) throughout.

## Table of contents
1. Connection (SSH, ControlMaster, Duo, login-node limits)
2. Allocation & storage (where files go, pre-caching)
3. Environment activation (modules, conda, the PATH-export pattern)
4. The offline rule (jobs inherit the proxy: export it in CPU pre-cache jobs, unset it in GPU jobs)
5. Partitions, GPUs, walltime, queue hygiene
6. Canonical sbatch anatomy
7. Loader gotchas (bf16, attn impl, the `lm_head` tie fix)
8. Data movement (laptop ↔ cluster) — authority split, never sync by deletion
9. Troubleshooting / field-tested failure modes (connection auth, smoke gate, node launch-kills, walltime/partition sizing)

---

## 1. Connection

```
# ~/.ssh/config                                    # SSH_ALIAS
Host <alias>
    HostName login.explorer.northeastern.edu
    User <user>
    IdentityFile ~/.ssh/<key>
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
    ControlPersist 8h
```

Every `ssh`/`scp`/`rsync` goes through `<alias>`.

- **ControlMaster** reuses one authenticated tunnel for ~8h across all `ssh`/`scp`/
  `rsync`. Set it in `~/.ssh/config` (a `Host <alias>` block is convenient). **Duo**
  two-factor is required only on the first cold connect or after the 8h window.
- **Key auth can bypass MFA — try the non-interactive probe FIRST (field-tested, L1).**
  If the user's `Host <alias>` block has an authorized `IdentityFile` plus
  `ControlMaster auto` + `ControlPersist`, a non-interactive
  `ssh -o BatchMode=yes -o ConnectTimeout=10 <alias> 'squeue -u <user>; sinfo -p gpu'`
  authenticates **by key with no Duo prompt**, and that first call also **opens the
  persistent master socket** that every later `ssh`/`scp`/`rsync` reuses. So the
  Quick-Start "ssh once to clear Duo" is the *fallback*: try the `BatchMode` key-auth
  probe via the alias first, and only drop to an interactive MFA login if the key turns
  out not to be authorized (the probe errors with a publickey/permission failure rather
  than succeeding). If it merely hangs, the human needs to re-clear Duo — surface it,
  don't retry blindly.
- **GOTCHA — always connect via the configured ALIAS, never the raw hostname (L1).**
  `ssh <user>@login.explorer.northeastern.edu` bypasses the `Host <alias>` block, so it
  picks up **none** of the `IdentityFile`/`ControlMaster`/`ControlPersist` config and
  fails with a `GSSAPI`/`Permission denied` error. The raw hostname is only the value of
  `HostName` *inside* the alias block; drive every connection (and every probe) through
  `<alias>`.
- **Dead control socket** (after a WiFi/IP change): remove the control socket
  `~/.ssh/control-*<alias>*` and reconnect once.

**Login-node limits:** the login node has a **~8 GB effective RAM
cap** (whatever `free` reports — the per-user cap is what bites) and will SIGKILL
(`exit 137`) heavy work. **Never** load the model, run a big `du`, or do real compute
there. Delegate to a compute node with `srun`/`sbatch`, including pre-caching (§2).

**The kill is SILENT — know the signature.** A login-node SIGKILL often produces
**no output at all**: no traceback, no "Killed" line, just a command that ran and
printed nothing. A ~200 MB parquet load or a few heavy imports back-to-back in a
shell loop is enough. This *reads exactly like an agent-code bug* and is not one —
classify it as **infrastructure** in the G5 streak (`autonomy-model.md`), move the
work to `srun --partition=short --mem=48G`, and run per-item checks one at a time
rather than in a login-node loop.

## 2. Allocation & storage

- **Persistent allocation:** `/projects/<group>/<user>` (`ALLOC`) — your group's
  persistent project space (typically ~1 TB). **Use this for everything that must
  survive.**
- **`/scratch/$USER` is purged monthly** (first Tuesday). **`/home/$USER`** has a low
  inode cap — do **not** put the HF cache there. Both are footguns; default to `ALLOC`.
- **HF cache:** `$ALLOC/hf_cache` (`HF_CACHE`). **Torch cache:** `$ALLOC/torch_cache`.
- **Pre-caching rule:** pre-cache in a CPU job, never on the login node, *before* the
  first GPU job (GPU jobs run offline, §4):

  ```bash
  ssh <alias> 'srun -p short --time=00:30:00 --mem=32G --cpus-per-task=4 \
      bash '"$ALLOC"'/scripts/preload_model.sh'
  ```

  The script exports the proxy (§4) and calls
  `snapshot_download(repo_id, revision=<sha>)`. `snapshot_download(revision=<sha>)`
  writes no `refs/main`, so offline loads must pass the same `revision=<sha>` (or you
  write `refs/main` yourself).
- **End every pre-cache with an offline load test in the same CPU job.** Unset the
  proxy, set `HF_HUB_OFFLINE=1`, and load every artifact the GPU job will load, through
  the same library calls. Some libraries list repo files through the hub API even when
  every file is cached, so offline mode crashes them although the cache is complete;
  this test catches that before any GPU time is spent. Fix it by recording what the
  call returns during pre-cache and serving it from a file in the GPU job.
- **Check the cache, not just the folder.** `ls -d models--<org>--<model>` passes on a
  half-downloaded or wrong-revision cache. Check for `snapshots/<rev>/config.json` plus
  at least one weight file. The same applies to environments: a half-built venv (for
  example from a cancelled install) has `bin/python` but nothing in site-packages, so
  check `pyvenv.cfg` **and** that the packages import before trusting it.

## 3. Environment activation

```bash
module load miniconda3/24.11.1                      # MODULES
```

Do not load a `cuda/*` module for pip-installed torch; the wheel carries its own
runtime. Load one only to compile CUDA extensions, matching the wheel's CUDA version.
The node's driver caps what runs, and drivers differ by node (field-observed: 570.86,
i.e. CUDA 12.8, on T4 and H200 nodes; 545.23 on a V100 node). cu12x wheels ran on
both; cu13x wheels need NVIDIA's forward-compat libs on `LD_LIBRARY_PATH` (recipe at the end
of this section). Print the
driver (`nvidia-smi`) in the smoke on each GPU type you route to (§5.1).

When you print package versions to verify an env, use
`importlib.metadata.version("<dist>")`: not every package defines `__version__`, and a
crashing check marks a good install as FAILED.

**Treat a working env as immutable.** Never force-reinstall into it: a forced torch
reinstall once broke an inference engine built against the old torch. For extra
packages:
1. Layer a venv: `<base-env>/bin/python -m venv --system-site-packages <new-env>`.
2. `pip install` the additions in a CPU job on `short` with the proxy.
3. Save `pip freeze` and print `module.__file__` for the key imports.

End every env build by writing `<env>/.built-ok`. Jobs check that marker (plus the §2
import check), not just `bin/python`; the template does.

Activate the conda env by **prepending its bin to PATH**, not `source activate` — in a
non-interactive sbatch shell `source activate` is fragile and ordering-dependent, and
plain `python` may resolve to system Python:

```bash
ENV=/projects/<group>/<user>/conda_envs/<conda-env>   # CONDA_ENV
export PATH=$ENV/bin:$PATH
PYTHON=$ENV/bin/python      # use $PYTHON explicitly everywhere, never bare `python`
```

Cache + determinism exports (caches must be under `ALLOC`, never `/tmp`):

```bash
export HF_HOME=$ALLOC/hf_cache TORCH_HOME=$ALLOC/torch_cache
mkdir -p "$HF_HOME" "$TORCH_HOME"
export CUBLAS_WORKSPACE_CONFIG=:4096:8              # deterministic cuBLAS
```

**CUDA 13 wheels on a 12.8 driver (forward compatibility).** If a package forces a
cu13x torch:
1. In a CPU job on `short` with the proxy, download NVIDIA's `cuda-compat-13-0` el9 RPM.
2. Unpack it with `rpm2cpio <rpm> | cpio -idm` under `$ALLOC/cuda-compat`.
3. In every GPU job, before any Python, run
   `export LD_LIBRARY_PATH=$ALLOC/cuda-compat/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH`.
   Load no `cuda/*` module.
4. In the smoke, check `torch.cuda.is_available()`, a matmul and the engine import, on
   each GPU type you route to. This has been field-verified on H200 only.

CUDA 13 dropped Volta, so cu13x stacks never run on V100. Prefer cu128 builds when your
packages allow, and keep a cu128 fallback env.

## 4. The offline rule (the single most expensive footgun)

Compute nodes have no direct internet; they reach it only through the squid proxy
(`PROXY`). Jobs **inherit** the proxy from the login shell, because the login node sets
it and `sbatch`/`srun` default to `--export=ALL` (verified 2026-10-12: a `short` job
reached the hub with the inherited variables and failed with them unset). So:

- **CPU pre-cache jobs:** export it explicitly:
  `export http_proxy=http://<squid-proxy-ip>:3128 https_proxy=... HTTP_PROXY=... HTTPS_PROXY=... no_proxy=localhost,127.0.0.1`.
  Do not rely on inheritance there.
- **Every GPU job:** unset it:
  `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1`.
  Also set the offline flags (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1`, `local_files_only=True`), so a missing file fails loudly at
  load time instead of the job hanging, retrying, or silently resolving a different
  revision. Unsetting it also keeps localhost servers (for example an inference
  engine) from being routed through the proxy.

Offline mode does not protect against a checkpoint whose weights are missing or
mis-tied: that loads "successfully" with freshly initialized tensors and generates
garbage (the `lm_head` case in §7; `long-job-patterns.md` §9). Assert an invariant only
the real weights satisfy.

Set all of these on every GPU job (the offline-mode rule):

```bash
# in the sbatch:
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY        # inherited from the login shell
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
```
```python
# in Python, before/at load:
import os; os.environ.setdefault("HF_HUB_OFFLINE", "1")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, revision=REVISION, local_files_only=True,   # <- mandatory
    dtype=torch.bfloat16, low_cpu_mem_usage=True)
proc  = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
```

`<squid-proxy-ip>` is the `PROXY` Customization Point (from the cluster's
research-computing docs). Export it explicitly in the pre-cache script, and unset it in
every GPU job.

## 5. Partitions, GPUs, walltime, queue hygiene

**Partitions and limits** (verified 2026-10-12 with
`sacctmgr -n -P show qos format=Name,MaxTRES,MaxTRESPU,MaxJobsPU,MaxSubmitPU`, where
MaxTRES is the per-job cap, and `scontrol show partition <p>`; re-check, they change):

| Partition | Use | Default / max time | GPUs per job | Per user: running / queued / GPUs | Access |
|---|---|---|---|---|---|
| `short` | CPU: pre-cache, env builds, statistics, aggregation | 4 h / 48 h | — | 50 jobs, 1024 CPUs | all |
| `gpu` | GPU work | 4 h / 8 h | 1 | 4 / 8 / 4 | all |
| `gpu-short` | smokes, short and 2-GPU jobs | 1 h / 2 h | 2 | 2 / 8 / 2 | all |
| `gpu-interactive` | interactive `srun` / OOD | 1 h / 2 h | 1 | 2 / 4 / — | all |
| `courses-gpu` | course work only | 4 h / 24 h | 1 | 2 / — / 2 | a course unix group |
| `multigpu` | more than 2 GPUs per job | — | — | — | restricted; ask Research Computing (T3) |

- Omitting `--time` gives the partition's **default**, not its maximum. Omitting `--mem`
  gives 2 GB per CPU (`DefMemPerCPU`).
- A job asking for more GPUs than the per-job cap is accepted and then **pends forever**
  with reason `QOSMaxGRESPerJob`. That is not queueing: cancel it and resize. DDP or
  tensor-parallel work must fit 2 GPUs × 2 h chunks on `gpu-short`, with checkpoints.
- A job over the running cap pends with `QOSMaxJobsPerUserLimit` until one of yours ends.
- `gpu`, `gpu-short` and `gpu-interactive` share the same nodes, so spreading across them
  adds job slots, not hardware.
- The caps apply across all GPU types and count every job under your user, including
  other projects and sessions (`squeue -u $USER`). Your account may map to a different
  QOS: `sacctmgr -n -P show assoc user=$USER format=account,partition,qos`.
- **Access is set by the partition, not by the account you bill.**
  `scontrol show partition <p>` shows AllowAccounts/AllowGroups/AllowQos; the course
  partition is gated by a unix group (`id -Gn`). Choose the GPU type per job (§5.1) and
  bill each project to its own `--account=<slurm-account>`; never borrow another
  project's account.
- Array tasks count individually toward MaxSubmitPU; `sacct -X` collapses a pending
  array into one row, so count queued tasks with `squeue -u $USER -h -r -o %i`. Stage
  submissions instead of bulk-submitting into a rejection. Array ids contain `[ ] % -`,
  so match them as fixed strings (`awk -F'|' -v j="$id" '$1==j'`, `grep -F`), never as
  regexes.
- **IdleBot cancels GPU jobs that sit under-utilised for an hour.** An email arrives at
  about 15 min, and `gpu-logs <jobid>` shows the record. CPU-heavy phases (statistics,
  probe fits, aggregation, data prep) go to `-p short`, never inside a GPU allocation.
  Log GPU utilisation every few minutes (the template logs `gpu-util:` lines every 5 min),
  and check in the smoke that the GPU is busy once loading ends. Symptom in `sacct`:
  `CANCELLED by 0`, with the batch step at exit 15. Policy:
  rc-docs.northeastern.edu/en/explorer-main/gpus/idlebot.html; `gpu-logs` keeps the
  record for 7 days. A pipeline with a CPU-heavy middle becomes three jobs: GPU, then CPU
  on `short`, then GPU, chained with `afterok`.
- **Job needs more than the 8h cap?** Don't fight the partition — design the job
  resumable (atomic per-item checkpoints, exit-0-if-done; `long-job-patterns.md` §2)
  and **chain it**: resubmit the same sbatch and it continues from checkpoints.
  Queue the successor at submit time with `--dependency=afterany:<jobid>` (not
  `afterok` — a TIMEOUT exits non-zero and `afterok` never fires, §5 below).
- On TIMEOUT, exit is non-zero and `afterok` deps won't fire (`afternotok` will). For
  resumable sweeps, write per-item outputs idempotently so a resume skips done work.
- **Dependencies.** Explorer runs with `kill_invalid_depend`, so a job whose dependency
  can no longer be met is cancelled, not left pending.
  1. Before you `scancel` a parent, including when rebalancing in §5.1, re-point or
     clear its children: `scontrol update JobId=<child> Dependency=afterany:<new-id>`
     (or `Dependency=` to clear).
  2. A dependency on a job that ended more than a few minutes ago (`MinJobAge`) is
     rejected with "Job dependency problem". Check `sacct -j <id>` and drop it.
  3. For two raced siblings, `--dependency=afterok:<a>?afterok:<b>` fires when either
     one succeeds.
- `cmd | head` under `set -o pipefail` can SIGPIPE-kill the job — use
  `cmd 2>&1 | head -n N || true`.

### 5.1 Choosing a GPU type (and spreading jobs across types)

Explorer's GPU partitions (`gpu` 8 h, `gpu-short` 2 h, `gpu-interactive`) share the
same nodes, so "idle in `sinfo`" is not "startable now". Types: `h200` (141 GB), `a100`
(40 GB; 80 GB on nodes with feature `a100@80g`, select with `--constraint=a100@80g`),
`v100-sxm2` / `v100-pcie` (16 or 32 GB; confirm with `nvidia-smi` in the smoke), `t4`
(16 GB). T4 and V100 have **no bf16**: run fp32, or fp16 with an overflow check. Before
every GPU submission:

1. **Need.** Take peak GPU memory from the smoke's `torch.cuda.max_memory_allocated()`
   and the runtime per item. Do not trust a hand estimate: logits, log-softmax and a
   previous batch's tensors still alive during the next forward can double it (one
   field run estimated ~5 GB and peaked at 12 GB). Keep at least 20% headroom.
2. **Types your env can run.** `$PYTHON -c "import torch; print(torch.cuda.get_arch_list())"`.
   sm_70 = V100, sm_75 = T4, sm_80 = A100, sm_90 = H200. Recent torch wheels have
   dropped sm_70 (cu130 builds, and also a torch 2.11 cu128 build in the field), so a
   V100 then fails at the first CUDA op with `no kernel image is available`. Check the
   arch list, not the CUDA version.
3. **Live availability.** Free GPUs per node, the best predictor of a fast start:
   `sinfo -p gpu-short -N -h -O NodeHost,Gres,GresUsed,StateLong` (use `-p gpu` for long
   jobs). Pending demand per type: `squeue -p gpu,gpu-short -h -t PD -o '%b' | sort | uniq -c`.
   Down or drained nodes and requests for types that no longer exist inflate the totals.
   Node RAM can block a free GPU: GPU nodes share their RAM across 3-8 GPUs, and a node
   with a free GPU may have almost no unallocated memory. Compare `Memory − AllocMem`
   from `sinfo -p <p> -N -h -O NodeHost,Gres,GresUsed,Memory,AllocMem` (MB) with your
   `--mem` (`FreeMem` is the OS's free RAM and does not drive scheduling). Request the
   RAM the smoke measured, not a padded default.
4. **Your caps** (§5): per user across all types, so splitting does not raise them,
   and per job (1 GPU on `gpu`, 2 on `gpu-short`).
5. **Start estimates are a weak hint.** `sbatch --test-only --gres=gpu:<type>:1 ...` gives
   backfill estimates. In one field run they said 28 min for a type with an idle node
   that then started in 28 s, and `squeue --start` returned N/A. Use them to break ties,
   not to decide.
6. **Submit to the type that fits and frees up soonest.** For short jobs, prefer an idle
   smaller GPU over a queued faster one; a small job on a big GPU also risks IdleBot
   under-utilisation. Never use bare `--gres=gpu:1` unless the code picks its dtype from
   the device's compute capability (`torch.cuda.get_device_capability()[0] >= 8` for
   bf16; `is_bf16_supported()` also returns True on T4/V100 through emulation) and the
   arch list covers every type.
7. **Spread embarrassingly parallel sweeps.** Split items across several single-GPU jobs
   on different free types, within the QOS caps, with interleaved shards
   (`long-job-patterns.md` §1) and per-item idempotent outputs. Overlapping one item
   across two types gives a cheap cross-hardware check; do not assume bit-identical
   results in general (TF32 or different kernels change them).
8. **Rebalance your own pending jobs.** If a job still pends after ~15 min while another
   fitting type shows free GPUs, `scancel` that pending job (your own only; re-point its
   dependents first, §5 Dependencies) and resubmit it on the free type. Pending time is infrastructure, not a G5 failure.
9. **Record** the chosen type, the alternatives and what you saw for them in the job's
   ledger entry, and log its GPU-hours under that type (G4).

## 6. Canonical sbatch anatomy

Use `assets/job_template.sbatch` — it encodes everything above. Its skeleton:

```
#SBATCH header   → job-name, --partition=gpu, --account, --gres=gpu:<type>:1 (§5.1),
                   --time (≤ partition cap + margin), --mem, --output/--error to
                   $ALLOC/.../logs/<job>-%j.{out,err}, --export=ALL
set -e -o pipefail
modules + conda  → module load $MODULES; PATH-export; $PYTHON
cache + offline  → HF_HOME/TORCH_HOME; HF_HUB/TRANSFORMERS/HF_DATASETS_OFFLINE=1; unset the inherited proxy
determinism      → CUBLAS_WORKSPACE_CONFIG
sanity           → nvidia-smi | head; torch.cuda.is_available() assert
pre-flight gate  → assert the model snapshot exists in HF_CACHE (else exit 2)  [G1 in spirit]
early signal     → --signal=B:USR1@600; python in background + wait; trap forwards USR1
the work         → $PYTHON code/<script>.py --args ... &   (checkpoints per item)
```

**Walltime and signals.** At the time limit Slurm sends SIGTERM to every process in the
job, then SIGKILL after `KillWait` (600 s on Explorer, 2026-10-11). A foreground
`python` dies from that same SIGTERM, so a SIGTERM trap in the batch shell runs only
after the Python state is gone. Per-item atomic checkpoints are the primary defence; for
an in-process save, request an early signal and forward it (the template does this):

```bash
#SBATCH --signal=B:USR1@600            # USR1 to the batch shell 10 min before the limit
$PYTHON code/<script>.py ... &         # background + wait, so the trap runs immediately
PY=$!
RC=0; SIG=0; TRC=0
trap 'SIG=1; kill -USR1 $PY; wait $PY && TRC=0 || TRC=$?' USR1   # script checkpoints on SIGUSR1
wait $PY || RC=$?                      # under set -e a bare `wait` would abort when the trap fires
if [ "$SIG" = 1 ]; then RC=$TRC; fi
[ "$RC" = 0 ] || exit "$RC"
```

(bash defers traps while a foreground child runs; hence `&` + `wait`. Tested locally:
the signalled, failing and clean paths exit 0, the script's code, and 0.)

**Submission plumbing that fails silently:**
- `#SBATCH` lines are not shell-expanded. Use Slurm patterns (`%x` = job name, `%j` = job
  id) in `--output`/`--error`, and set `--job-name` at submit time.
- `sbatch --export=ALL,IDX=9,10,11` splits on the commas, so `IDX` becomes `9` and the
  job still exits 0 with a plausible result. Pass lists through the environment
  (`IDX=9,10,11 sbatch ... job.sbatch`, with the default `--export=ALL`) or use another
  separator, and have the job log the list it parsed and assert it equals the request.
- `ssh <alias> 'nohup srun ... &'` keeps the ssh call open until srun ends, even with
  stdio redirected. `ssh <alias> '( srun ... > log 2>&1 < /dev/null & ); echo started'`
  returns at once, but for anything longer than a few seconds use `sbatch`.

## 7. Loader gotchas

- **bf16 on A100/H200; on T4/V100 use fp32 (or fp16 with an overflow check).** Never
  int4/int8 for diagnostics; never quantize the KV cache. `do_sample=False`
  (greedy) for diagnostics; pin and log every seed.
- **Attention impl:** `attn_implementation="eager"` when extracting attention or
  hooking internals; `"sdpa"` for routine generation.
- **`lm_head` tie fix — [EXAMPLE, delete if you adapt to another model].** This is
  specific to one Qwen-VL-family checkpoint: the
  checkpoint omits `lm_head.weight` and relies on `tie_word_embeddings`, but
  transformers 5.x reads that flag off the *outer* config (False) not `text_config`
  (True), so `lm_head` stays at meta-init zeros and generation collapses to `"!"`.
  Every loader must do:

  ```python
  model.config.tie_word_embeddings = True
  model.tie_weights()
  assert model.lm_head.weight.data_ptr() == \
         model.model.language_model.embed_tokens.weight.data_ptr()
  ```

  **If you adapt this skill for a different model, this fix is likely unnecessary or
  different — delete or replace it.** It's the clearest example of a Customization
  Point that is model-specific, not cluster-specific.

- **Record WHY each pin can't move.** Pins interlock: an inference engine's newer
  wheels may target a CUDA the driver doesn't have, while the older engine breaks on
  a newer `transformers` API — so *neither* pin can float, and a "harmless" upgrade
  of one bricks the other. A one-line comment per pin ("engine Y pinned because its
  newer wheels target CUDA 13 and the driver is 12.8") turns a future debugging
  session into a read.
- **New model on a pinned engine? Check the engine's model registry FIRST** — before
  designing any experiment around the model. A serving engine given an architecture
  it doesn't support may silently fall back to a related implementation and die deep
  inside its own startup profiling with a misleading shape error (cost: an hour of
  confusion, minimum). One import of the engine's model registry on a compute node
  answers it in seconds.
- **Inference-engine entry points.** Build the engine under `if __name__ == "__main__":`.
  When the engine uses the `spawn` start method (for example after a multimodal warm-up
  has initialised CUDA), the child re-imports your script, and an unguarded engine fails
  at startup.

## 8. Data movement (laptop ↔ cluster)

Autonomous loops shuttle code up and results down constantly. Three rules, each paid
for in a near-disaster or a false alarm:

- **Never sync by deletion.** `rsync --delete` once came within a prompt of erasing
  the cluster-side scripts that were the provenance for published numbers. The trees
  legitimately diverge in BOTH directions, so pick the authority per artifact class:
  **the cluster is authoritative for anything a cluster run produced** (result JSONs,
  receipts, acquisition scripts of record); **local is authoritative for source**.
  Sync additively in each direction; delete manually, deliberately, never as a flag.
- **Always exclude the junk:** `--exclude '.venv*' --exclude '__pycache__'
  --exclude '.git'` — one stray local venv once swept 3,000+ files into a transfer.
- **Verify what matters.** For any artifact a downstream number depends on, record an
  md5 at the source and check it after transfer (`long-job-patterns.md` §6 stamps
  them into result files for exactly this reason). scp races on files still being
  written are a known infrastructure failure — copy from checkpoints or after the
  job's terminal state, not from a live output file.
- **Many remote files: stream a tar.** `scp` with several remote globs can fail with
  "Connection closed"; `ssh <alias> 'tar cf - -C <dir> <files>' | tar xf -` works.
- **Never check out a different commit in a cluster tree** while any job from that tree
  is running or pending; the job reads whatever code is there when it starts.

## 9. Troubleshooting / field-tested failure modes

These are the failure modes the skill hit on its first real end-to-end run. Each is
named so you can match a symptom to a recovery without re-deriving it, and each notes
whether it is an **agent-code** bug (counts toward the G5 failure streak) or an
**infrastructure** failure (does **not** count — see `autonomy-model.md` §G5).

### 9.1 Connection auth fails or prompts unexpectedly (L1)
See §1 for the full mechanics. The two field-tested checks, in order:
1. **Probe with `BatchMode` via the ALIAS first** — key auth often clears with no Duo
   prompt and opens the persistent master socket in the same call. Interactive MFA is
   the fallback, not the default.
2. **If you get `GSSAPI`/`Permission denied`, you almost certainly used the raw
   hostname.** Re-run through `<alias>`, which is the only path that loads the
   `IdentityFile`/`ControlMaster` config.

Infrastructure, not agent code — a failed connect never increments the G5 streak.

### 9.2 Smoke gate (G1): smoke via `sbatch`, not only `srun` (L2)
**Symptom you want to avoid:** a job passes an interactive `srun` smoke and then **dies
at launch** the moment it's submitted as a real `sbatch`.

**Why `srun` alone is insufficient:** an interactive `srun` smoke streams stdout back
over your SSH session and writes **no `--output` file**. So it exercises neither
output-file creation nor the full `sbatch` node prolog/launch path — exactly the steps
that produce the launch failures in §9.3. A job can be green under `srun` and dead under
`sbatch`.

**Rule:** the **G1** smoke gate must validate the **actual artifact you will submit**.
Include at least one real `sbatch` submission (tiny inputs, `gpu-short`, short `--time`)
that writes to a real `--output` path — not only an `srun` stream. This is the cheap
form of the end-to-end wiring check, applied to the launch path itself.

### 9.3 Node launch-kill — sbatch dies in seconds with no log (L3)
A named, recurring **infrastructure** failure. Do **not** "fix" your code for it; if you
treat it as an agent-code bug you both waste time and wrongly increment the **G5**
streak.

**Symptom (the fingerprint):**
- The `sbatch` job dies in **~1–5 s** with `State=FAILED`.
- `sacct` shows the **`.batch` step `CANCELLED`** with a **signal-style `ExitCode`**
  (e.g. `0:53`), while the **`.extern` step is `COMPLETED`**.
- **No `--output`/`--error` log is written** — the job was killed *before your script's
  first line ran*, so there is nothing in the logs to debug.

This is a **node-level launch failure**: the node failed to bring the job up. It is not
a code bug and it is not preemption.

**Diagnostic ladder — run these before changing any code:**
```bash
# (a) Which node(s) did it land on?
ssh <alias> 'sacct -j <jobid> --format=JobID,State,ExitCode,NodeList -P'

# (b) Minimal canary on the SAME GPU type as the failed job (another type tests a
#     different node pool). Success on a different node means the original node is bad.
ssh <alias> 'srun -p gpu-short --time=00:02:00 --mem=2G --gres=gpu:<type>:1 \
    --account=<slurm-account> bash -lc "hostname; nvidia-smi -L; echo canary-ok"'

# (c) Rule out SYSTEMIC causes before blaming a node:
ssh <alias> 'sacct -j <jobid> -o JobID,State,ExitCode,Reason,NodeList -P'   # pending/fail reason
ssh <alias> 'sacctmgr -n -P show qos format=Name,MaxSubmitPU,MaxJobsPU,MaxTRESPU'  # QOS limits
ssh <alias> 'scontrol show reservations'      # maintenance reservations
#  + confirm the --output dir is writable.
```

On 2026-10-11 Explorer had `PreemptMode=OFF` and no reservations, so preemption is not a
cause; re-check if that changes.

**Fix:** exclude the bad node(s) and resubmit unchanged:
```
#SBATCH --exclude=<bad-nodes>
```
(`<bad-nodes>` = the `NodeList` from step (a), comma-separated.) If the canary in (b)
also fails on a fresh node, the cause is systemic (step (c)), not a single bad node —
escalate rather than excluding the whole partition.

**G5 classification:** infrastructure. Does **not** count toward the failure streak.

### 9.4 Walltime / partition sizing + short-partition timeout recovery (L4)
**(a) A short-walltime partition can TIMEOUT a job that actually finished its real
work** but ran out of clock during the final aggregation/write step. Defend against
this *before* it happens and recover cheaply *after*:
- **Write per-item outputs as you go, idempotently (skip-existing).** Then the
  expensive GPU work is durable the moment each item lands.
- **If a job times out after its per-item outputs landed, do not re-run on GPU.**
  Re-aggregate CPU-only from the per-item files inside
  `srun -p short --time=00:20:00 --mem=16G ...`, never on the login node (§1: a ~200 MB
  load can trigger the silent kill). (This is the same per-item / idempotent pattern as
  the resumable-sweep note in §5.)

**(b) "Idle" does not mean "startable now".** `gpu`, `gpu-short` and `gpu-interactive`
share the same nodes, and per-user QOS caps (§5) hold a job back even when GPUs are
free. Choose the type with §5.1, which reads free GPUs per node and your caps.

**(c) Size `--time` from the smoke, and don't undercut it.** Shorter walltime backfills
faster and schedules sooner, but estimate the real runtime from the **per-item time
measured in the G1 smoke** × item count, plus margin for load/aggregation. Undercutting
that estimate just buys you a §9.4(a) timeout. You can lower a job's limit
(`scontrol update JobId=<id> TimeLimit=<t>`), but raising it is admin-only, whether the
job is pending or running (`man scontrol`). So size `--time` at submit, and make long
jobs resumable rather than padding them to the cap.

**G5 classification:** a walltime-timeout of code that did its real work, and pending
time, are **infrastructure** — they do **not** count toward the failure streak.

### 9.5 What counts as a G5 streak failure (L5 — reinforced)
Infrastructure, not agent code (does not increment G5): connection/auth failures, node
launch-kills, NODE_FAIL, a walltime timeout of code whose checkpoints were advancing, and
pending time. **Counts as agent code:** an IdleBot cancellation of a job the agent
designed with a long CPU-only phase inside a GPU allocation, because that is a
job-design bug.

They must **not** increment the G5 consecutive-failure streak (`.failure_streak.json`). Record each explicitly with
`counted_in_streak: false`. Only a genuinely new bug in agent-written code (e.g. an OOM
you introduced in preprocessing, a wrong wrapper, a logic error) increments the streak.
When in doubt, ask: *"did my code have a bug, or did the cluster misbehave?"* See
`autonomy-model.md` §G5 for the canonical rule and ledger shape.
