# Cluster setup — Northeastern Explorer

Connection, storage, environment activation, the offline rule, the canonical sbatch
anatomy, and loader gotchas. Replace the `Customization Points` values from `SKILL.md`
(`SSH_ALIAS`, `ALLOC`, `SLURM_ACCOUNT`, `CONDA_ENV`, `MODULES`, `HF_CACHE`,
`MODEL_ID`/`REVISION`) throughout.

## Table of contents
1. Connection (SSH, ControlMaster, Duo, login-node limits)
2. Allocation & storage (where files go, disk budget, pre-caching, hub tokens)
3. Environment activation (modules, conda, the PATH-export pattern, per-node drivers, the driver gate)
4. The offline rule (jobs inherit the proxy: export it in CPU pre-cache jobs, unset it in GPU jobs)
5. Partitions, GPUs, walltime, queue hygiene (the `sharing` partition, submit slots, fairshare, the queue cascade)
6. Canonical sbatch anatomy (signals, traps, submission plumbing)
7. Loader gotchas (bf16, attn impl, the `lm_head` tie fix, transformers 5.x) and serving engines in jobs
8. Data movement (laptop ↔ cluster) — transfer defaults, authority split, never sync by deletion
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
  With an authorized `IdentityFile` plus `ControlMaster auto` + `ControlPersist` in the
  alias block,
  `ssh -o BatchMode=yes -o ConnectTimeout=10 <alias> 'squeue -u <user>; sinfo -p gpu'`
  authenticates **by key with no Duo prompt** and **opens the master socket** every later
  `ssh`/`scp`/`rsync` reuses. An interactive MFA login is the fallback, only when the
  probe fails with a publickey/permission error. If it merely hangs, the human needs to
  re-clear Duo: surface it, don't retry blindly.
- **GOTCHA — always connect via the configured ALIAS, never the raw hostname (L1).**
  `ssh <user>@login.explorer.northeastern.edu` skips the alias block's
  `IdentityFile`/`ControlMaster` config and fails with `GSSAPI`/`Permission denied`.
  Drive every connection and probe through `<alias>`.
- **Dead control socket** (after a WiFi/IP change): remove the control socket
  `~/.ssh/control-*<alias>*` and reconnect once.
- Single ssh calls to the login node have taken over 2 minutes (field-observed
  2026-10). Give every scripted call a timeout, so one slow call cannot stall a tick.

**Login-node limits:** keep the login node to `squeue`, `sacct`, `sinfo`, `cat`, `grep`
and light `git`. It has a **~8 GB effective RAM cap** (whatever `free` reports — the
per-user cap is what bites), but it also SIGKILLs (`exit 137`) work far below any
memory cap. Field-observed 2026-10: a few-MB `rsync` into a cluster tree at 0.05 GB RSS
(leaving a partial copy with files missing), a stdlib `python3` JSON read, a one-module
`python -c 'import <pkg>'`, and a tokenizer load. **Never** load a model, run `du`, or
run any Python there. Run tree builds, rsyncs into trees, `du` inventories, import
checks and every Python step as a short CPU job:
`sbatch --partition=sharing --time=00:15:00 --mem=4G --wrap '...'` starts within about a
minute (the rerun of the killed rsync took 6 s), while `short` has sometimes queued for
days (§5). After any copy, verify that every file arrived. There is no `nvidia-smi` on
the login node, so GPU checks belong in a job. Large single transfers through the login
node have also been killed silently: stage files and verify a checksum in a CPU job
(§8).

**The kill is SILENT — know the signature.** A login-node SIGKILL often produces
**no output at all**: no traceback, no "Killed" line, just a command that ran and
printed nothing. A single import is enough (above). This *reads exactly like an
agent-code bug* and is not one — classify it as **infrastructure** in the G5 streak
(`autonomy-model.md`), move the work to a CPU job (`sharing` when it fits in 1 h,
`short` beyond), and run per-item checks one at a time rather than in a login-node loop.

## 2. Allocation & storage

- **Persistent allocation:** `/projects/<group>/<user>` (`ALLOC`) — your group's
  persistent project space, **shared with the whole group**. **Use this for everything
  that must survive.** Its size varies by group, and free space can fall by more than
  100 GB a week when several members write to it. Check it with `df -h $ALLOC` from a
  CPU job, never `du` on the login node, and set a per-project disk budget.
- **`/scratch/$USER` is purged monthly** (first Tuesday) and not backed up.
  **`/home/$USER`** has a low inode cap — do **not** put the HF cache there. Both are
  footguns; default to `ALLOC`.
- **Checkpoints fill disks fast.** Libraries that keep every checkpoint can write
  hundreds of GB (optimizer state included). Keep intermediate checkpoints on
  `/scratch` and copy only the final artifact to `$ALLOC`.
- **HF cache:** `$ALLOC/hf_cache` (`HF_CACHE`). **Torch cache:** `$ALLOC/torch_cache`.
  If any shared model cache lives on `/scratch`, the purge breaks every job that reads it
  at once: keep the pre-flight cache check in every GPU job and re-run the pre-cache job
  after each purge.
- **Pre-caching rule:** pre-cache in a CPU job, never on the login node, *before* the
  first GPU job (GPU jobs run offline, §4). Submit it with `sbatch`, not an interactive
  `srun`, so a busy queue cannot block the session; watch it with `sacct`:

  ```bash
  ssh <alias> 'sbatch -p sharing --time=00:30:00 --mem=32G --cpus-per-task=4 \
      --account=<slurm-account> '"$ALLOC"'/scripts/preload_model.sh'
  ```

  `sharing` usually starts sooner (§5); use `short` for a download that may need more
  than 1 h. The script exports the proxy
  (§4) and calls `snapshot_download(repo_id, revision=<sha>)`.
  `snapshot_download(revision=<sha>)` writes no `refs/main`, so offline loads must pass
  the same `revision=<sha>` (or you write `refs/main` yourself).
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
- **Hub tokens on the cluster.** Use a read-only, fine-grained token that covers only
  the repos you pre-cache, keep it in a file with mode 600, and never paste it into a
  chat or session transcript. GPU jobs run offline and need no token, so do not
  export it there. Rotate it before any upload; rotation stays on the human's checklist
  (T3: credentials).

## 3. Environment activation

```bash
module load miniconda3/24.11.1                      # MODULES
```

Do not load a `cuda/*` module for pip-installed torch; the wheel carries its own
runtime. Load one only to compile CUDA extensions, matching the wheel's CUDA version.
Without one, a job has no `nvcc` or `ninja`, so anything that JIT-compiles on first use
fails there (§7 Serving engines in jobs).

**Drivers differ by node**, not only by GPU type, so a `--gres` type does not pin the
driver, and the node's driver caps what runs. Field-observed 2026-10: H200 d4052-d4055
on 570.86.15 (CUDA 12.8); H200 NVL d4077 (on `sharing`) on 595.58.03; one A100 node on
570.86.15, while A100 80GB PCIe d3233 runs 545.23.08; a V100 node on 545.23; T4 nodes on
570.86. Keep one per-node driver table (`node -> gpu, driver, source job`) for all your
projects: every template job prints `NODE_DRIVER node=<host> gpu=<name> driver=<version>`,
`assets/fold_drivers.py` folds those lines from job logs into the table, and 2-minute
probe jobs cover unprobed types. Log a new node or driver in the field log
(`evolution.md`). Steer with `--exclude=` for nodes your env cannot use (or
`--nodelist=` to restrict to known-good ones). cu12x (cu128) wheels ran on 570 and 545
nodes and need no compat lib on 570 nodes: prefer them when your packages allow, and
keep a cu128 fallback env. A cu13x env cannot run on 545 nodes and needs the driver gate
below everywhere else.

When you print package versions to verify an env, use
`importlib.metadata.version("<dist>")`: not every package defines `__version__`, and a
crashing check marks a good install as FAILED.

**Treat a working env as immutable.** Never force-reinstall into it: a forced torch
reinstall once broke an inference engine built against the old torch. Never
`pip install` into a shared base env. For extra packages:
1. Layer a venv: `<base-env>/bin/python -m venv --system-site-packages <new-env>`.
2. `pip install` the additions in a CPU job on `short` with the proxy. If the base is
   already inconsistent (`pip check` shows a conflict), the resolver re-validates base
   packages and stops with `ResolutionImpossible`: install the new package with
   `--no-deps`, then its missing requirements with `-c <constraints>`, a constraints
   file made from the base's `pip freeze`.
3. Save `pip freeze` and print `module.__file__` for the key imports. A guard that checks
   the venv does not shadow base packages must exempt `pip`, `setuptools` and `wheel`,
   which a venv carries at its own versions.

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

**CUDA 13 wheels (forward compatibility) and the driver gate.** If a package forces a
cu13x torch:
1. In a CPU job on `short` with the proxy, download NVIDIA's `cuda-compat-13-0` el9 RPM.
2. Unpack it with `rpm2cpio <rpm> | cpio -idm` under `$ALLOC/cuda-compat`.
3. **Driver gate.** In every GPU job, before any Python, load the compat path **only
   when the node's driver needs it**. Keep one copy of the gate: the one in
   `assets/job_template.sbatch` (set `CU13=1` there). It reads the driver safely under
   `set -e -o pipefail`, prints a `NODE_DRIVER node=... gpu=... driver=...` line, and
   decides:

   | Driver read | GPU allocated? | Action |
   |---|---|---|
   | none (nvidia-smi missing, failing or empty) | yes: `scontrol show job` AllocTRES has `gres/gpu`, falling back to `SLURM_JOB_GPUS`/`SLURM_STEP_GPUS` | exit 4: node problem, `--exclude` it and resubmit |
   | none | no | exit 2: missing `--gres?` |
   | below 570 (cu13x env) | yes | exit 4: below the floor, `--exclude` the node |
   | 570-579 | yes | load the cuda-13.0 compat libcuda |
   | 580 or newer | yes | skip the compat libcuda |

   It does not use `CUDA_VISIBLE_DEVICES` to decide, because `--export=ALL` can carry it
   over from the submit shell. Evidence, field-observed 2026-10: 570.86 works with the
   compat lib; 595.58 (an H200 NVL) reported `CUDA unavailable` with it and ran torch
   cu130 plus a recent inference engine without it; on 545.23 cu130 torch failed with or
   without it. The 580 cut-off is inferred from NVIDIA's forward-compat rule (the compat
   libcuda must be newer than the driver, and 580 already carries CUDA 13). The 570 floor
   is a choice, since no 546-569 driver was seen. Both exit-4 branches and the exit-2 branch
   are new and untested on the cluster; a 2-minute probe on a 550-, 560- or 580-class
   node would settle the range. Exit 4 is infrastructure; exit 2 is agent design unless
   shown otherwise (§6 exit codes). Put the gate in every script that touches the GPU,
   including scripts a wrapper job runs, or the wrapper inherits an old unconditional
   export. Load no `cuda/*` module.
4. In the smoke, check `torch.cuda.is_available()`, a matmul and the engine import on
   each GPU type **and driver** you route to. An A100 run with compat was reported
   working, but its node's driver was not recorded (inferred to be a 570 node).

CUDA 13 dropped Volta, so cu13x stacks never run on V100.

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
MaxTRES is the per-job cap, and `scontrol show partition <p>`; `gpu`/`gpu-short`
re-checked unchanged 2026-10-21; `sharing` and `courses` field-observed 2026-10;
re-check, they change):

| Partition | Use | Default / max time | GPUs per job | Per user: running / queued / GPUs | Access |
|---|---|---|---|---|---|
| `short` | CPU: pre-cache, env builds, statistics, aggregation | 4 h / 48 h | — | 50 jobs, 1024 CPUs | all |
| `sharing` | short CPU helpers (tests, tree builds, rsync into trees, inventories, small statistics) and <= 1 h GPU pieces | 1 h (jobs timed out at 1 h; check default vs max with `scontrol show partition sharing`) | unverified | 2 / 4 / 2 | all (observed under a lab account and a course account) |
| `gpu` | GPU work | 4 h / 8 h | 1 | 4 / 8 / 4 | all |
| `gpu-short` | smokes, short and 2-GPU jobs | 1 h / 2 h | 2 | 2 / 8 / 2 | all |
| `gpu-interactive` | interactive `srun` / OOD | 1 h / 2 h | 1 | 2 / 4 / — | all |
| `courses` | CPU work billed to a course account | accepted 4 h and 12 h walls (max unverified) | — | unverified | course account / unix group |
| `courses-gpu` | course work only | 4 h / 24 h | 1 | 2 / — / 2 | a course unix group |
| `multigpu` | more than 2 GPUs per job | — | — | — | restricted; ask Research Computing (T3) |

Baseline software on the H200 nodes (2026-10): Slurm 23.11.6, bash 5.1.8, kernel
5.14.0-362.13.1 on every H200 node checked. Record these with the driver table (§3): a
node that differs is the first suspect when signal, trap or kernel behaviour changes
between runs.

- Omitting `--time` gives the partition's **default**, not its maximum. Omitting `--mem`
  gives 2 GB per CPU (`DefMemPerCPU`).
- `short` is not always the fast CPU option: it has queued for days at least once
  (2026-10-12). Small CPU tasks go to `sharing`.
- **The `sharing` partition.** It is heterogeneous: P100, V100, an H100 80 GB (d4041)
  and an H200 NVL (d4077, driver 595.58, see the §3 driver gate) were all seen there, so
  always pass a typed `--gres=gpu:<type>:1`; a bare `--gres=gpu:1` got a V100. CPU jobs
  there started within about a minute while `short` estimates were hours to days out,
  which makes it the default for any login-node-sized task (§1):
  `sbatch --partition=sharing --time=00:15:00 --mem=4G --wrap '...'`. Anything that may
  need more than an hour goes elsewhere: a statistics job that needed more timed out
  there. The 1-h wall still suits resumable GPU work: 1-h resume pieces on its H200
  started the same day while `gpu` estimates were the next day. Chain the pieces with
  `afterany` (below) and count them against its 4-submitted cap.
- **Course partitions.** `courses-gpu` has only V100/P100, so cu13x stacks and anything
  that needs bf16 cannot run there. It also queued for hours (about 6 h, and overnight
  until morning) while `sharing` under the same course account started at once.
- A job asking for more GPUs than the per-job cap is accepted and then **pends forever**
  with reason `QOSMaxGRESPerJob`. That is not queueing: cancel it and resize. DDP or
  tensor-parallel work must fit 2 GPUs × 2 h chunks on `gpu-short`, with checkpoints.
- A job over the running cap pends with `QOSMaxJobsPerUserLimit` until one of yours ends.
- `gpu`, `gpu-short` and `gpu-interactive` share the same nodes, so spreading across them
  adds job slots, not hardware.
- The caps apply across all GPU types and count every job under your user, including
  other projects and sessions (`squeue -u $USER`). Your account may map to a different
  QOS: `sacctmgr -n -P show assoc user=$USER format=account,partition,qos`. Per-account
  job caps (about 250) never bound in practice; per-user QOS caps did.
- **Access is set by the partition, not by the account you bill.**
  `scontrol show partition <p>` shows AllowAccounts/AllowGroups/AllowQos; the course
  partitions are gated by a unix group (`id -Gn`). Choose the GPU type per job (§5.1) and
  bill each project to its own `--account=<slurm-account>`; never borrow another
  project's account, and bill each course's work only to that course's account, even
  when another course account shows better fairshare.
- **Fairshare sets your priority.** Every session and project billing one account under
  your user shares one fairshare score. One heavy study can push a user's FairShare far
  below 0.5 when its decayed usage is a large multiple of its share, and every job billed
  to that account then queues at low priority. When jobs pend on `Priority`, check
  `sshare -u $USER -l` (FairShare, EffectvUsage vs NormShares) and `sprio -u $USER`. Past
  usage decays with the configured half-life
  (`scontrol show config | grep PriorityDecayHalfLife`; reported as about 7 days, which is
  also Slurm's default: verify), so priority recovers by itself. A reservation or priority boost is a request to your PI
  or Research Computing (T3).
- **Everything queued counts toward MaxSubmitPU:** array tasks, **dependency-held jobs**
  (an `afterok`/`afterany` chain spends a slot per link) and every job of every project
  under your user. `sacct -X` collapses a pending array into one row, so count queued
  tasks with `squeue -u $USER -h -r -o %i`. Before submitting a chain or a wrapper, count
  the slots it will use, nested submissions included; to spend fewer, have one job loop
  over several units instead of one job per unit. Stage submissions instead of
  bulk-submitting into a rejection. Two traps when the cap is full:
  1. An `sbatch` that a running job makes for itself (a wrapper submitting its own
     successor or a stats step) is refused with `QOSMaxSubmitJobPerUserLimit`, often
     after the GPU work is done. Make such jobs print a clear "submission failed, submit
     by hand" line, or submit successors from the login node or laptop with a retrying
     feeder.
  2. `sbatch --test-only` is refused the same way, so a verifier that requires it reports
     NOT READY on a job the scheduler would accept a minute later. Treat that refusal as
     "retry when a slot frees", not a failed check.

  Array ids contain `[ ] % -`, so match them as fixed strings
  (`awk -F'|' -v j="$id" '$1==j'`, `grep -F`), never as regexes.
- **IdleBot cancels GPU jobs that sit under-utilised for an hour**, with an email at
  about 15 min. CPU-heavy phases (statistics, probe fits, aggregation, data prep) go to a
  CPU partition, never inside a GPU allocation. A cold model load from NFS can reach the
  15-min email by itself (§7 Serving engines in jobs). Log GPU utilisation every few
  minutes (the template logs `gpu-util:` lines every 5 min), and check in the smoke that
  the GPU is busy once loading ends. For live utilisation of a running job use
  `srun --jobid=<id> --overlap nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`
  plus its `gpu-util:` lines, a coarse view at the sampler interval. The policy page
  says `gpu-logs <jobid>` keeps the record for 7 days, but in one 2026-10 check it did
  not run from the login node (explorer-01) nor through `srun` on the job's node: treat it as optional, and use the IdleBot email for a
  cancelled job. Observed symptom in `sacct` (one field observation; the policy page
  does not list it): `CANCELLED by 0`, with the batch step at exit 15. Policy:
  rc-docs.northeastern.edu/en/explorer-main/gpus/idlebot.html. A pipeline with a
  CPU-heavy middle becomes three jobs: GPU, then CPU, then GPU, chained with `afterok`.
  Each held link spends a submit slot (above), so for a long chain submit each stage from
  a laptop feeder when the previous one ends.
- **Job needs more than the 8h cap?** Don't fight the partition — design the job
  resumable (atomic per-item checkpoints, per-item outputs written idempotently,
  exit-0-if-done; `long-job-patterns.md` §2) and **chain it**: resubmit the same sbatch
  and it continues from checkpoints. Queue the successor at submit time with
  `--dependency=afterany:<jobid>` (not `afterok`: a piece that stops early exits 75 and a
  TIMEOUT is non-zero, so `afterok` never fires; `afternotok` does), and make each piece
  exit 0 immediately when the work is already done.
- **Dependencies.** Explorer runs with `kill_invalid_depend`, so a job whose dependency
  can no longer be met is cancelled, not left pending.
  1. Before you `scancel` a parent, including when rebalancing in §5.1, re-point or
     clear its children: `scontrol update JobId=<child> Dependency=afterany:<new-id>`
     (or `Dependency=` to clear).
  2. A dependency on a job that ended more than a few minutes ago (`MinJobAge`) is
     rejected with "Job dependency problem". Check `sacct -j <id>` and drop it.
  3. For two raced siblings, `--dependency=afterok:<a>?afterok:<b>` fires when either
     one succeeds.
  4. **Insurance chunk.** For a long job that should finish inside one wall but might
     not, queue its resume at submit time with `--dependency=afternotok:<first>` (and
     `RESUME=auto`). This rests on the §6 exit codes: a job that finished exits 0 and
     `kill_invalid_depend` cancels the insurance chunk with no GPU time spent; a job that
     stopped early on the stop file exits 75, and a TIMEOUT or failure is non-zero, so
     the resume runs. A job that checkpoints on the stop file and exits 0 defeats it:
     the insurance is cancelled and the work stays unfinished. For chains that will need
     several pieces, use `afterany`. Every held link occupies a submit slot.
- `cmd | head` under `set -o pipefail` can SIGPIPE-kill the job — use
  `cmd 2>&1 | head -n N || true`.

### 5.1 Choosing a GPU type (and spreading jobs across types)

Explorer's GPU partitions (`gpu` 8 h, `gpu-short` 2 h, `gpu-interactive`) share the
same nodes, so "idle in `sinfo`" is not "startable now". GPU inventory (field-observed
2026-10; re-check with `sinfo -o '%P %N %G %f'`, which also gives each type's gres
name): `gpu` and `gpu-short` share four H200 nodes, d4052-d4055 with 8 × 141 GB each (32
H200s, gres `h200`). Excluding one of them removes a quarter of the pool, and nodes get
drained for maintenance: once, 15 of the 32 H200s were out while the A100s were full.
`sharing` adds an H200 NVL (d4077) and the only H100 80 GB (d4041), plus P100/V100.
Below the H200 come A100 (`a100`; 40 GB, or 80 GB with `--constraint=a100@80g`), then
48 GB cards (L40S, L40, A6000; listed in sinfo, not probed for bf16 or driver), then
V100 (`v100-sxm2` / `v100-pcie`, 16 or 32 GB; confirm with `nvidia-smi` in the smoke),
T4 (`t4`, 16 GB) and P100. T4, V100 and P100 have **no bf16**: run fp32, or fp16 with an
overflow check. GPU jobs sharing a node may run slower: one two-shard job on a node
shared with two trainings took 0.46 h against 0.27 h for its two-shard sibling (one
observation). Before every GPU submission:

1. **Need.** Take peak GPU memory from the smoke's `torch.cuda.max_memory_allocated()`
   and the runtime per item. Do not trust a hand estimate: logits, log-softmax and a
   previous batch's tensors still alive during the next forward can double it (one
   field run estimated ~5 GB and peaked at 12 GB). Keep at least 20% headroom.
2. **Types and nodes your env can run.** `$PYTHON -c "import torch; print(torch.cuda.get_arch_list())"`.
   sm_60 = P100, sm_70 = V100, sm_75 = T4, sm_80 = A100, sm_90 = H100/H200. Recent
   torch wheels have dropped sm_70 (cu130 builds, and also a torch 2.11 cu128 build in
   the field), so a V100 then fails at the first CUDA op with
   `no kernel image is available`; current cu12x/cu13x wheels may not include sm_60
   either. Check the arch list, not the CUDA version. Then check the node driver (§3):
   a cu13x env cannot use 545 nodes, so steer with `--exclude=`/`--nodelist=` from your
   driver table and keep the §3 driver gate in the job.
3. **Live availability.** Free GPUs per node, the best predictor of a fast start:
   `sinfo -p gpu-short,sharing -N -h -O NodeHost,Gres,GresUsed,StateLong` (add `-p gpu`
   for long jobs). Pending demand per type:
   `squeue -p gpu,gpu-short,sharing -h -t PD -o '%b' | sort | uniq -c`. Down or drained
   nodes and requests for types that no longer exist inflate the totals. Free GPUs on a
   node whose driver your env cannot use do not count. Node RAM can block a free GPU: GPU
   nodes share their RAM across 2-8 GPUs, and a node with a free GPU may have almost no
   unallocated memory. Compare `Memory − AllocMem` from
   `sinfo -p <p> -N -h -O NodeHost,Gres,GresUsed,Memory,AllocMem` (MB) with your `--mem`
   (`FreeMem` is the OS's free RAM and does not drive scheduling). Request the RAM the
   smoke measured, not a padded default.
4. **Your caps** (§5): per user across all types, so splitting does not raise them,
   and per job (1 GPU on `gpu`, 2 on `gpu-short`). Check headroom per partition yourself:
   `sacctmgr -n -P show qos format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU` against
   `squeue -u $USER -h -r -o '%P %T' | sort | uniq -c`.
5. **Start estimates ignore your own caps.** `sbatch --test-only --gres=gpu:<type>:1 ...`
   gives backfill estimates that do not apply your per-user QOS caps: a job can show an
   early estimate and then sit on `QOSMaxJobsPerUserLimit` because your slots are full.
   They also go to N/A when the queue jams, and are noisy: in one field run they said
   28 min for a type with an idle node that then started in 28 s, and `squeue --start`
   returned N/A.
6. **Queue cascade.** When a GPU job's estimate is more than about an hour away, rank
   every (partition × GPU type) candidate by **estimated finish** (start estimate plus
   the projected runtime on that type), not by start. Filter on memory, bf16, the
   partition wall (split a resumable job into pieces of wall minus start-up overhead),
   node driver (step 2) and a free running slot (step 4). Shorter walls backfill far
   sooner. Field-observed with about 100 H200 jobs pending for 32 H200s: an 8-h `gpu`
   job sat 9 h on Priority, and the same work resubmitted on `gpu-short` with
   `--time=01:30:00` started that afternoon; 1-h resume pieces on `sharing` beat a
   next-day `gpu` estimate. A reusable script is optional; the estimates come from a loop
   like this, run under bash on the login node (zsh does not word-split `$c`). Take the
   gres names from `sinfo -o '%P %N %G'`; a refusal while your submit slots are full
   means "retry later" (§5):
   ```bash
   # EXCL: your exclude list plus nodes whose driver your env cannot use (driver table), comma-separated
   for c in "gpu h200 08:00:00" "gpu-short h200 01:30:00" "gpu a100 08:00:00" "sharing <h200-nvl-gres> 01:00:00"; do
     set -- $c
     printf '%-10s %-16s %s  ' "$1" "$2" "$3"
     sbatch --test-only -p "$1" --gres=gpu:"$2":1 --time="$3" ${SLURM_ACCOUNT:+--account="$SLURM_ACCOUNT"} \
         ${EXCL:+--exclude="$EXCL"} job.sbatch 2>&1 \
       | grep -o 'to start at [^ ]*' || echo 'no estimate (refused or N/A)'
   done
   ```
   If a script prints a chaining command for the pieces, make it `afterany` (§5).
7. **Submit to the candidate that fits and finishes soonest.** For short jobs, prefer an
   idle smaller GPU over a queued faster one; a small job on a big GPU also risks IdleBot
   under-utilisation. Never use bare `--gres=gpu:1` unless the code picks its dtype from
   the device's compute capability (`torch.cuda.get_device_capability()[0] >= 8` for
   bf16; `is_bf16_supported()` also returns True on T4/V100 through emulation) and the
   arch list covers every type.
8. **Spread embarrassingly parallel sweeps, but never split a contrast.** Split items
   across several single-GPU jobs on different free types, within the QOS caps, with
   interleaved shards (`long-job-patterns.md` §1) and per-item idempotent outputs, only
   when no measurement compares outputs **across** jobs. If a quantity is fitted on one
   set of runs and read on another (a probe fitted on base states and read on treated
   states, a with-vs-without contrast), run the whole contrast on one GPU type. If you
   must fall back, move the entire set to a same-architecture type (H200 to H100, both
   Hopper) after a smoke there, never half of it. On one type at equal batch size,
   repeated captures were bit-identical (max abs diff 0 over about 800 rows): assert that
   as a cheap determinism check, with a cosine floor (for example 0.9999) as the fallback
   bound. Across types, do not assume bit-identical results (TF32 or different kernels
   change them); overlapping one item across two types gives a cheap cross-hardware
   check.
9. **Rebalance your own pending jobs.** If a job still pends after ~15 min while another
   fitting type shows free GPUs, `scancel` that pending job (only your project's, by
   job-name prefix; re-point its dependents first, §5 Dependencies) and resubmit it on
   the free type. Pending time is infrastructure, not a G5 failure.
10. **Record** the chosen type, the alternatives and what you saw for them in the job's
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
sanity           → §3 driver gate before any Python; a NODE_DRIVER node=... gpu=... driver=... line
                   (feeds the §3 driver table); torch.cuda.is_available() + one CUDA op
pre-flight gate  → assert the model snapshot exists in HF_CACHE (else exit 2)  [G1 in spirit]
early signal     → --signal=B:USR1@600; python in background + wait; the USR1 trap writes a stop file
                   (and forwards USR1 only when FORWARD_USR1=1)
the work         → $PYTHON code/<script>.py --args ... &   (checkpoints per item; exit codes below)
```

**Exit codes.** One contract for every GPU job script, which chains and the G5 triage
(§9.5) read:

| Code | Meaning | Next step |
|---|---|---|
| 0 | work complete, or already complete (skip) | none |
| 75 | stopped early on the wall signal or the stop file, after writing a checkpoint | resubmit to resume; `afternotok` insurance fires on it, resume chains use `afterany` |
| 4 | node or driver incompatible, or a GPU allocated with its driver unreadable, caught by the pre-flight before any work | infrastructure: exclude the node, resubmit |
| 2 | any other pre-flight refusal (missing `--gres`, missing cache, env not marked built, a guard) | agent design, unless the cause is shown to be infrastructure |
| 3 | a failed self-check after work started | read the log |
| other | the program's own status | triage table |

A program can exit 2, 3 or 4 for its own reasons (python itself exits 2 on a usage error
or a script it cannot open), so the code alone does not prove a contract meaning. Read 2
and 4 as pre-flight codes only when the log has the matching `FATAL:` line from the
pre-flight (`below the 570 floor`, `driver unreadable`, `missing --gres?`, `not marked
built`, ...); otherwise triage it as the program's own status.

**Walltime and signals.** At the time limit Slurm sends SIGTERM to every process in the
job, then SIGKILL after `KillWait` (600 s on Explorer, 2026-10-11). A foreground
`python` dies from that same SIGTERM, so a SIGTERM trap in the batch shell runs only
after the Python state is gone. Per-item atomic checkpoints are the primary defence;
checkpoint every 50-100 steps so a lost signal costs at most one interval. For an
in-process save, request an early signal: the template's USR1 trap writes a stop file,
which the training loop polls every step (`os.path.exists(os.environ['STOP_FILE'])`),
then checkpoints and exits 75 (a stop-file stop was verified on an H200; the code 75 is
the contract above). The trap forwards USR1 to python only when `FORWARD_USR1=1`, for a
program that installs a SIGUSR1 handler: python's default SIGUSR1 action terminates the
process, so forwarding to a poll-only script kills it before it checkpoints. A forwarded
USR1 is also unreliable on its own: in 2 of 3 GPU training runs the batch-shell trap
logged the signal but python's handler never saw it, so training ran into the wall and
TIMEOUT cancelled the chain (root cause unproven; the same path stopped correctly in a
CPU reproduction). (One training loop also called
`signal.pthread_sigmask(signal.SIG_BLOCK, [])` each poll to flush a pending signal, an
untested extra.)

**Manual early stop.** The stop file lives in the compute node's `/tmp`, so you cannot
create it from the login node. Send the same signal the wall sends:
`scancel --signal=USR1 --batch <jobid>` reaches the batch shell, whose trap writes the
stop file. Install a stop-file-only USR1 trap as soon as `STOP_FILE` is set (below): bash's
default action for USR1 kills the batch shell, so a signal that lands before the main
trap (a wall of 10 min or less with `@600`, or an early manual stop) would otherwise end
the job with no stop file and no exit 75.

```bash
#SBATCH --signal=B:USR1@600            # USR1 to the batch shell 10 min before the limit
( set +e; while sleep 300; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sed 's/^/gpu-util: /'; done ) &
WATCH=$!
trap 'kill "$WATCH" 2>/dev/null || true' EXIT
export STOP_FILE=/tmp/stop-${SLURM_JOB_ID:-$$}; rm -f "$STOP_FILE"
trap ': > "$STOP_FILE" || true' USR1   # early USR1 (before python starts): keep the shell alive
$PYTHON code/<script>.py ... &         # background + wait, so the trap runs immediately
PY=$!
RC=0; SIG=0; TRC=0
trap 'SIG=1; : > "$STOP_FILE" || true; if [ "${FORWARD_USR1:-0}" = 1 ]; then kill -USR1 "$PY" 2>/dev/null || true; fi; wait "$PY" && TRC=0 || TRC=$?' USR1
wait "$PY" || RC=$?                    # under set -e a bare `wait` would abort when the trap fires
if [ "$SIG" = 1 ]; then RC=$TRC; fi
[ "$RC" = 0 ] || exit "$RC"            # 75 (stopped early) reaches Slurm
```

(bash defers traps while a foreground child runs; hence `&` + `wait`. Tested locally on
bash 3.2: clean and failing runs exit 0 and the script's code; a signalled poll-only
script exits 75, as does one with a handler under `FORWARD_USR1=1`; a poll-only script
under `FORWARD_USR1=1` is killed by the signal. A USR1 sent before python starts leaves
the shell running with the stop file written, and the poll-only script then exits 75;
without the early trap the shell dies of the signal. This short snippet's final `exit "$RC"`
passes a program's own 2, 3 or 4 through unchanged; the template remaps them to 1 unless
`PROGRAM_FOLLOWS_EXIT_CONTRACT=1`. Either way, triage 2 and 4 by their `FATAL:` line, as
above.)

**Traps must not fail.** Under `set -e`, the status of the last command in an EXIT trap
becomes the script's exit status: `trap 'kill $WATCH 2>/dev/null' EXIT` turns a
successful job into exit 1 whenever the utilisation sampler has already died, and a
sampler subshell that inherits `set -e -o pipefail` dies on a single failed `nvidia-smi`
call (hence `set +e` inside it). The job is then marked FAILED, its `afterok` children
never fire, and the failure can be miscounted in G5. A failing `kill` inside the USR1
trap (python already gone) also aborts the script. End every trap with commands that
cannot fail (`|| true`), as above. Verified with local repros on bash 3.2: the
dead-watcher case gives rc 1 without the fix and rc 0 with it, and a USR1 trap whose
`kill` fails exits 1. Also reported on the cluster's bash 5.1.8.

**Check what your library traps, and where the signal lands.**
1. Some training libraries trap only SIGINT/SIGTERM and never poll a stop file
   (observed: SAELens 6.54). The early USR1 then changes nothing, and forwarding it
   (`FORWARD_USR1=1`) would kill them. Poll `$STOP_FILE` from a callback and exit 75 if
   the library allows; otherwise Slurm's SIGTERM at the wall makes them write a partial
   checkpoint that a resume has to recognise and skip: keep only checkpoints whose files
   are all present and size-consistent.
2. If python runs inside a pipe or `bash -c` wrapper (`python ... | tee log`), the
   batch-shell signal reaches the wrapper, not python. `exec` python, use process
   substitution for the log, or forward explicitly.
3. A child sbatch file run with `bash jobs/X.sbatch` inside a wrapper job ignores its
   `#SBATCH` lines (`--signal`, `--time`, `--gres`), because only `sbatch` parses them
   (inferred). The wrapper's own header governs.

**Submission plumbing that fails silently:**
- `#SBATCH` lines are not shell-expanded. Use Slurm patterns (`%x` = job name, `%j` = job
  id) in `--output`/`--error`, and set `--job-name` at submit time.
- `sbatch --export=ALL,IDX=9,10,11` splits on the commas, so `IDX` becomes `9` and the
  job still exits 0 with a plausible result. Pass lists through the environment
  (`IDX=9,10,11 sbatch ... job.sbatch`, with the default `--export=ALL`) or use another
  separator, and have the job log the list it parsed and assert it equals the request.
- `--export=ALL` also hands your submit-time **control flags** to everything the job
  runs, including its own self-tests: a skip flag set for one run reached an in-job
  pytest, and two end-to-end tests failed. Have test sessions clear job-control
  variables at start (for example in `conftest.py`), and deploy that isolation to every
  code tree, because a tree built before the fix fails the same way.
- **An sbatch error does not mean no job.** When slurmctld is overloaded, `sbatch` can
  print `Unexpected message received`, time out or hang and still create the job, which
  may show up in `squeue`/`sacct` many minutes later (field-observed: up to 8 min).
  Guarded resubmits that saw zero jobs a few seconds after the error produced duplicates
  more than once, and a batch reported as failed had created a duplicate chain.
  `scancel` and `scontrol update` can also time out and then take effect. Give every
  submission a unique `--job-name`/`--comment`. After an error or hang, wait at least
  30 s (10 min is safer while the controller struggles), query **both**
  `squeue -u $USER -h -n <job-name>` and `sacct -X -n -S <today> --name=<job-name>`, and
  resubmit only if both are empty. If a duplicate appears, cancel the later one before it
  writes outputs. Submit one job at a time until the controller recovers. Skipping this
  costs a per-user slot plus two jobs racing on one output dir. Classify the error as
  infrastructure.
- Array index maps are per stage. If two pipeline stages order their units differently,
  submitting stage B with stage A's indices computes the wrong units, and some tasks then
  die on missing inputs. Before each array submission, print the index-to-unit map from
  the manifest the job will read and check it against the intended units.
- `ssh <alias> 'nohup srun ... &'` keeps the ssh call open until srun ends, even with
  stdio redirected. `ssh <alias> '( srun ... > log 2>&1 < /dev/null & ); echo started'`
  returns at once, but for anything longer than a few seconds use `sbatch`.

## 7. Loader gotchas

- **bf16 on A100/H100/H200; on T4/V100/P100 use fp32 (or fp16 with an overflow check).**
  Never int4/int8 for diagnostics; never quantize the KV cache. `do_sample=False`
  (greedy) for diagnostics; pin and log every seed.
- **Attention impl:** `attn_implementation="eager"` when extracting attention or
  hooking internals; `"sdpa"` for routine generation.
- **`lm_head` tie fix — [EXAMPLE, delete if you adapt to another model].** This is
  specific to one checkpoint that omits `lm_head.weight` and relies on `tie_word_embeddings`, but
  transformers 5.x reads that flag off the *outer* config (False) not `text_config`
  (True), so `lm_head` stays at meta-init zeros and generation collapses to `"!"`.
  Every loader must do:

  ```python
  model.config.tie_word_embeddings = True
  model.tie_weights()
  assert model.lm_head.weight.data_ptr() == \
         model.model.language_model.embed_tokens.weight.data_ptr()
  ```

  **For a different model, delete or replace this fix**: it is model-specific, not
  cluster-specific.

- **transformers 5.x breaks old training and save code** (observed on 5.17).
  `TrainingArguments(warmup_ratio=...)` is gone (`warmup_steps` accepts a float ratio).
  `save_pretrained(safe_serialization=...)` is gone. `save_pretrained` now validates
  `GenerationConfig` on every save and refuses a checkpoint whose generation config sets
  `temperature`/`top_p` without `do_sample`. That breaks LoRA merges and full-model
  checkpoint saves of such models: set `model.generation_config.do_sample = True` before
  saving, then copy the original `generation_config.json` back into the output. Before a
  smoke, check argument names against the **installed** transformers/TRL source
  (`inspect.signature`), not docs or memory: two such smoke failures in one day once
  completed a three-failure streak and tripped a G5 rollback. Check optional dependencies too: if `bitsandbytes` is absent,
  8-bit optimizers are unavailable, so name the substitute in results.
- **Record WHY each pin can't move.** Pins interlock: an inference engine's newer
  wheels may need a CUDA some nodes' drivers don't have, while the older engine breaks
  on a newer `transformers` API — so *neither* pin can float, and a "harmless" upgrade
  of one bricks the other. Pins can be node-dependent: a cu13x engine runs on 570 nodes
  with the §3 driver gate but never on 545 nodes. A one-line comment per pin ("engine Y
  pinned because its newer wheels need cu13x torch and we route to 545 nodes", or
  "pinned to reproduce published numbers") turns a future debugging session into a
  read.
- **New model on a pinned engine? Check the engine's model registry FIRST** — before
  designing any experiment around the model. A serving engine given an architecture
  it doesn't support may silently fall back to a related implementation and die deep
  inside its own startup profiling with a misleading shape error (cost: an hour of
  confusion, minimum). One import of the engine's model registry on a compute node
  answers it in seconds.

### Serving engines in jobs

- **Budget start-up, then keep the GPU busy.** Field-measured with an env on `/projects`
  and model caches on `/scratch`: vLLM model loads took 2 to 18 min (load_seconds 132 to
  1109). Size did not predict it (a smaller model took longer than one more than twice
  its size), which points to cold NFS/scratch reads rather than `torch.compile`
  (inferred; `enforce_eager=True` did not remove it). Python plus engine import and engine spawn alone took 4 to 6 min cold and
  1.5 to 2.5 min warm. A training job's model read from NFS sat in D state for about
  14 min before the GPU did anything. That is right at IdleBot's ~15-min email (§5), which went
  out for a job that restarted vLLM six times in one allocation (each restart idled the
  GPU 4 to 6 min for 35 to 120 s of generation). Rules:
  1. Give smokes that load models a wall of at least 45 min. A smoke TIMEOUT during a
     cold load is infrastructure only when its wall met that minimum; below it, the wall
     is agent design.
  2. In a multi-model job, load each model once: one long-lived process that imports
     once, runs every stage for model A, frees it (`del llm`, `gc.collect()`,
     `torch.cuda.empty_cache()`), then loads B; or one job per model.
  3. Batch many items per engine start rather than one small job per item.
  4. Count load time in GPU-hour projections.
- **Local servers on shared nodes: derive the port, verify it is yours.** Two of your
  own jobs can land on one node, and both may start a server on a fixed port.
  Field-observed: the second job's health check accepted the first job's server, and it
  died with `Connection refused` when that job ended. Derive the port per job, fail if it
  already answers, and after `/health` passes require that **your** server PID is alive
  and that `/v1/models` lists the expected served name. This applies to any job-local
  server (judge, engine, TensorBoard):
  ```bash
  PORT=${PORT:-$((20000 + SLURM_JOB_ID % 20000))}          # an explicit PORT wins
  URL=http://127.0.0.1:$PORT
  if curl -s -o /dev/null --max-time 2 "$URL/health"; then echo "FATAL: port $PORT already answers"; exit 2; fi
  $PYTHON -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH" --served-model-name "$SERVED" --port "$PORT" &
  SRV=$!
  DEADLINE=$((SECONDS + 2400))
  until curl -sf -o /dev/null --max-time 2 "$URL/health"; do
    kill -0 "$SRV" 2>/dev/null || { echo "FATAL: server exited during start-up"; exit 3; }
    [ "$SECONDS" -lt "$DEADLINE" ] || { echo "FATAL: server not healthy in 40 min"; exit 3; }
    sleep 10
  done
  kill -0 "$SRV" 2>/dev/null && curl -sf --max-time 5 "$URL/v1/models" | grep -qF "\"$SERVED\"" \
    || { echo "FATAL: $URL is not this job's server"; exit 3; }
  ```
- **Co-resident engines need explicit memory shares.** vLLM reserves the
  `gpu_memory_utilization` share of the GPU at start (default 0.9). A large judge
  server, at a share that was not recorded, held over half of a 141 GB H200, and loading
  a second large model next to it OOMed 75 min into the job. Set
  `gpu_memory_utilization` and `max_model_len` explicitly for every server sharing a
  GPU, and size them from the smoke's `nvidia-smi` memory, not from parameter counts.
  Stop all servers before any large in-process load.
- **JIT paths need a toolchain jobs don't have.** With vLLM 0.31, the FlashInfer
  top-p/top-k sampler JIT-compiles on first use and shells out to `ninja` and `nvcc`.
  Neither is on a job's PATH when no `cuda/*` module is loaded (§3), so the engine fails
  at the first sampled request. Set `export VLLM_USE_FLASHINFER_SAMPLER=0`: attention
  keeps its prebuilt kernels, and at `top_p=1.0` the sampler adds nothing. In general,
  run a real generation, not only an import, in the smoke, so first-use JIT failures
  surface there.
- **Inference-engine entry points.** Build the engine under `if __name__ == "__main__":`.
  When the engine uses the `spawn` start method (for example after a multimodal warm-up
  has initialised CUDA), the child re-imports your script, and an unguarded engine fails
  at startup.

## 8. Data movement (laptop ↔ cluster)

Autonomous loops shuttle code up and results down constantly. The rules below were each
paid for in a near-disaster or a false alarm:

- **Transfer defaults: tar streams, not macOS `rsync`.** On current macOS,
  `/usr/bin/rsync` is openrsync (protocol 29) and fails against Explorer with
  `unexpected end of file`. That reads like a size or network problem but isn't one.
  Folder `scp` has also been cut off mid-copy, and `scp` with several remote globs can
  fail with "Connection closed". Defaults:
  - Push: never unpack straight into a tree from the login node (it SIGKILLs such
    copies and leaves them partial, §1). Pack once, copy the archive and its checksum
    into a fresh staging dir, then verify and unpack in a short CPU job and read its
    log (`push.tgz: OK`). `--no-xattrs --no-mac-metadata` keeps macOS xattrs and `._*`
    files out. Deploy only into a tree with no queued job (below).
    ```bash
    tar --no-xattrs --no-mac-metadata --exclude='.venv*' --exclude=__pycache__ --exclude=.git -czf push.tgz <paths>
    shasum -a 256 push.tgz > push.tgz.sha256
    STAGE=/scratch/<user>/stage/$(date +%Y%m%d-%H%M%S)
    ssh <alias> "mkdir -p $STAGE" && scp push.tgz push.tgz.sha256 <alias>:"$STAGE/"
    ssh <alias> "sbatch --partition=sharing --time=00:15:00 --mem=4G --chdir=$STAGE --wrap 'sha256sum -c push.tgz.sha256 && tar xzf push.tgz -C <tree>'"
    ```
  - Single big files take the same route (gzip large text first): nothing reaches the
    live path until its checksum passes, so a half-written file is never used.
  - Pull: `ssh <alias> 'tar -C <dir> -czf - <paths>' | tar xzf -`.

  Or install GNU rsync (Homebrew) and call it by absolute path. Observed rate: about
  8.9 GB in about 55 min by tar stream.
- **Always exclude the junk,** in tar as in rsync: `--exclude='.venv*'
  --exclude=__pycache__ --exclude=.git` — one stray local venv once swept 3,000+ files
  into a transfer.
- **Never sync by deletion.** `rsync --delete` once came within a prompt of erasing
  the cluster-side scripts that were the provenance for published numbers. The trees
  legitimately diverge in BOTH directions, so pick the authority per artifact class:
  **the cluster is authoritative for anything a cluster run produced** (result JSONs,
  receipts, acquisition scripts of record); **local is authoritative for source**.
  Sync additively in each direction; delete manually, deliberately, never as a flag.
- **Verify what matters.** For any artifact a downstream number depends on, record an
  md5 at the source and check it after transfer (`long-job-patterns.md` §6 stamps
  them into result files for exactly this reason). scp races on files still being
  written are a known infrastructure failure — copy from checkpoints or after the
  job's terminal state, not from a live output file.
- **Never change code in a cluster tree** while any job from that tree is running or
  pending; the job reads whatever code is there when it starts. That includes
  **single-file copies**: if your smoke stamps hash the code package (G1 by hash),
  copying one updated module, adding a diagnostic file or deploying a shared fix into a
  tree with pending jobs invalidates every stamp there, and the queued full runs refuse
  to start; copying code in while a job runs makes its provenance say "code changed
  mid-run". Before touching a tree, confirm `squeue` shows no job of its prefix.
  Patterns that worked: deploy into a new side tree (cloned from a `git bundle` of the
  full branch, detached at the reviewed commit, built as a CPU job), give each tree its
  own smoke, and make each job refuse to start unless the tree's `HEAD` equals the commit
  it was smoke-tested at. Keep tests and ledgers outside the hash. Scope dirty-tree
  checks to code paths (`git status --porcelain --untracked-files=no -- <code dirs>`),
  because symlinked results/data overlays otherwise report dirty.
- **Stamp the commit when the job starts,** not when it ends: an end-of-job stamp reads
  whatever tree is there by then. If a side tree shares data or results with the first
  by symlink, run `readlink` on them after every checkout, because `git checkout -f`
  turns tracked paths back into real files.

## 9. Troubleshooting / field-tested failure modes

These are the failure modes the skill hit on its first real end-to-end run. Each is
named so you can match a symptom to a recovery without re-deriving it, and each notes
whether it is an **agent-code** bug (counts toward the G5 failure streak) or an
**infrastructure** failure (does **not** count — see the G5 rule in `autonomy-model.md`).

### 9.1 Connection auth fails or prompts unexpectedly (L1)
Mechanics in §1. In order: (1) probe with `BatchMode` via the alias (key auth, no Duo,
opens the master socket); interactive MFA is the fallback. (2) `GSSAPI`/`Permission
denied` almost always means the raw hostname was used: re-run through `<alias>`.
Infrastructure, not agent code: a failed connect never increments the G5 streak.

### 9.2 Smoke gate (G1): smoke via `sbatch`, not only `srun` (L2)
**Symptom:** a job passes an interactive `srun` smoke, then **dies at launch** as a real
`sbatch`. An `srun` smoke streams stdout over SSH and writes **no `--output` file**, so
it skips output-file creation and the `sbatch` prolog/launch path, the steps behind
§9.3's failures.

**Rule:** the **G1** smoke validates the **artifact you will submit**: at least one real
`sbatch` (tiny inputs, `gpu-short`, a `--time` that covers model loading, §7) writing to
a real `--output` path.

### 9.3 Node launch-kill — sbatch dies in seconds with no log (L3)
A named, recurring **infrastructure** failure: do **not** "fix" your code for it or count
it in **G5**.

**Symptom (the fingerprint):**
- The `sbatch` job dies in **~1–5 s** with `State=FAILED`.
- `sacct` shows the **`.batch` step `CANCELLED`** with a **signal-style `ExitCode`**
  (e.g. `0:53`), while the **`.extern` step is `COMPLETED`**.
- **No `--output`/`--error` log is written** — the job was killed *before your script's
  first line ran*, so there is nothing in the logs to debug.

The node failed to bring the job up: not a code bug, not preemption.

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

Keep **one** exclude list for all your projects (for example
`~/.config/explorer/exclude_nodes`, one node per line with date and reason) and submit
with `--exclude=$(paste -sd, <file>)`. Re-run the canary on an entry before relying on it
again: excluded nodes get repaired, and an outdated list shrinks your pool.

**G5 classification:** infrastructure. Does **not** count toward the failure streak.

### 9.4 Walltime / partition sizing + short-partition timeout recovery (L4)
**(a) A short-walltime partition can TIMEOUT a job that actually finished its real
work** but ran out of clock during the final aggregation/write step. Defend against
this *before* it happens and recover cheaply *after*:
- **Write per-item outputs as you go, idempotently (skip-existing).** Then the
  expensive GPU work is durable the moment each item lands.
- **Write the required outputs before any optional phase.** Three CPU statistics jobs
  finished their required results within 40 min, then TIMED OUT in an unbounded
  exploratory step before writing anything, losing the results three times. Write and flush every required artifact first; put open-ended analysis
  after it, behind a skip flag and its own time budget (`long-job-patterns.md`).
- **If a job times out after its per-item outputs landed, do not re-run on GPU.**
  Re-aggregate CPU-only from the per-item files in a CPU job
  (`sbatch -p sharing --time=00:20:00 --mem=16G --wrap '...'`, or `short` if it may need
  more than 1 h), never on the login node (§1: even an import can trigger the silent
  kill). (This is the same per-item / idempotent pattern as the resumable-sweep note in
  §5.)

**(b) "Idle" does not mean "startable now".** `gpu`, `gpu-short` and `gpu-interactive`
share the same nodes, and per-user QOS caps (§5) hold a job back even when GPUs are
free. Choose the type with §5.1, which reads free GPUs per node and your caps.

**(c) Size `--time` from the smoke, and don't undercut it.** Shorter walltime backfills
faster and schedules sooner (an 8-h `gpu` job sat 9 h; the same work at 1:30 on
`gpu-short` started that afternoon, §5.1 queue cascade), but undercutting the real runtime just buys you a
§9.4(a) timeout. Smoke per-item rates usually overstate time: start-up sits inside the
smoke (a training smoke ran at 11-16k tokens/s against 29k in full runs), and small
batches understate engine throughput (a timing smoke measured about 8 s per item, the
full vLLM run about 1.2 s, because the engine batches more on a large set). So they are
a safe upper bound on time per item. Start-up itself is not: a warm-cache re-smoke
loaded in under 2 min where the cold load had taken 19 min, so add the cold-load figure
(§7) separately.
- Fit the wall from the smoke's **max** per-item time (not the median) × item count,
  plus the cold start-up (§7 Serving engines in jobs). Projections resting on 1-3 timed
  calls are thin.
- Choose `gpu-short` only when that projection × 1.25 fits its 2 h. Split any job
  projected above about 6 h of an 8 h wall into resumable pieces.
- CPU-only fits deserve a per-cell timing smoke too: one fit ran at about 12 min per
  cell on `short` (8 CPUs) against about 4 min for the whole fit on a laptop. Thread
  settings are a suspect, not a confirmed cause: set
  `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` to `$SLURM_CPUS_PER_TASK`
  and measure.

You can lower a job's limit, and raise a **pending** job's limit yourself up to the
partition maximum (`scontrol update JobId=<id> TimeLimit=<t>`; verified 2026-10-12).
Raising a **running** job's limit is admin-only (`man scontrol`). So size `--time` at
submit, and make long jobs resumable rather than padding them to the cap.

**G5 classification:** a walltime-timeout of resumable code that did its real work, and
pending time, are **infrastructure** — they do **not** count toward the failure streak.
An agent-chosen wall too short for a non-resumable job is agent code (§9.5).

### 9.5 What counts as a G5 streak failure (L5 — reinforced)
Read the job's exit code first (§6 exit codes): 75 is a planned stop, not a failure
(resubmit to resume); 4 is infrastructure; 2 is agent design unless the cause is shown
to be infrastructure. Read 2 and 4 that way only with the pre-flight's matching `FATAL:`
line in the log (§6); without it the code is the program's own.

Infrastructure, not agent code (does not increment G5): connection/auth failures, node
launch-kills, NODE_FAIL, a walltime timeout of resumable code whose checkpoints were
advancing, pending time, a node or driver mismatch caught by the pre-flight before any
work or a GPU allocated with its driver unreadable (exit 4 from the §3 driver gate), a
smoke TIMEOUT during a cold model load when the smoke's wall met the §7 minimum
(45 min), an sbatch error or hang from an overloaded
controller (§6), a submit refused by the per-user cap (`QOSMaxSubmitJobPerUserLimit`),
an operator or self cancel, and expected failures of a deliberate diagnostic
reproduction.

**Counts as agent code:** an IdleBot cancellation of a job the agent designed with a long
CPU-only phase inside a GPU allocation, because that is a job-design bug; a TIMEOUT from
an agent-chosen wall too short for a non-resumable job, whether it was progressing or
not (hung *or* under-sized are both agent design), including a model-loading smoke under
the 45-min minimum; a GPU job submitted without `--gres` (exit 2, `missing --gres?`);
test-isolation and tolerance bugs; a deployment that missed a tree; an array submitted
with another stage's index map (§6).

Map `CANCELLED by <uid>` with `id -u` before classifying: your own uid means you or a
delegated subagent cancelled it (check subagent transcripts), not IdleBot or an admin.

Infrastructure failures must **not** increment the G5 consecutive-failure streak
(`.failure_streak.json`). Record each explicitly with `counted_in_streak: false`. Only a
genuinely new bug in agent-written code (e.g. an OOM you introduced in preprocessing, a
wrong wrapper, a logic error) increments the streak.
When in doubt, ask: *"did my code have a bug, or did the cluster misbehave?"* See the G5
rule in `autonomy-model.md` for the canonical rule and ledger shape.
