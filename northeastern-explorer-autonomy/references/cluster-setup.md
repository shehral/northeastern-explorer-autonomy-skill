# Cluster setup — Northeastern Explorer

Connection, storage, environment activation, the offline rule, the canonical sbatch
anatomy, and loader gotchas. Replace the `Customization Points` values from `SKILL.md`
(`SSH_TARGET`, `ALLOC`, `SLURM_ACCOUNT`, `CONDA_ENV`, `MODULES`, `HF_CACHE`,
`MODEL_ID`/`REVISION`) throughout.

## Table of contents
1. Connection (SSH, ControlMaster, Duo, login-node limits)
2. Allocation & storage (where files go, pre-caching)
3. Environment activation (modules, conda, the PATH-export pattern)
4. The offline rule (why compute nodes need it; the proxy split)
5. Partitions, GPUs, walltime, queue hygiene
6. Canonical sbatch anatomy
7. Loader gotchas (bf16, attn impl, the `lm_head` tie fix)
8. Data movement (laptop ↔ cluster) — authority split, never sync by deletion
9. Troubleshooting / field-tested failure modes (connection auth, smoke gate, node launch-kills, walltime/partition sizing)

---

## 1. Connection

```bash
ssh <user>@login.explorer.northeastern.edu        # SSH_TARGET
```

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
- **Stale socket** (after a WiFi/IP change): remove the control socket
  `~/.ssh/control-*<alias>*` and reconnect once.

**Login-node limits:** the login node has a **~8 GB effective RAM
cap** (whatever `free` reports — the per-user cap is what bites) and will SIGKILL
(`exit 137`) heavy work. **Never** load the model, run a big `du`, or do real compute
there. Delegate to a compute node with `srun`/`sbatch`. The one allowed heavy-ish
login-node task is pre-caching downloads (§2), which is I/O not RAM.

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
- **Pre-caching rule:** models must be downloaded to `HF_CACHE` from the **login node**
  (which has proxy internet) *before* the first compute-node job, because compute nodes
  are offline (§4). A short pre-cache run:

  ```bash
  ssh explorer 'srun -p short --time=00:30:00 --mem=32G --cpus-per-task=4 \
      bash '"$ALLOC"'/scripts/preload_model.sh'
  ```

  The preload script sets the login-node proxy env, then does a normal (online)
  `from_pretrained(MODEL_ID, revision=REVISION)` so the snapshot lands in `HF_CACHE`.

## 3. Environment activation

Pinned modules (do **not** float CUDA — `cuda/12.8` is too new for torch 2.4.1 wheels
built against 12.4; 12.3 driver runs them fine):

```bash
module load miniconda3/24.11.1 cuda/12.3.0          # MODULES
```

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

## 4. The offline rule (the single most expensive footgun)

**Compute nodes cannot reach the internet.** The Squid proxy is reachable **only from
the login node**. `--export=ALL` propagates the proxy env vars to the job, but the
endpoint itself is unreachable from compute nodes, so HF API calls there silently 503.

If offline mode is **not** set, `from_pretrained` swallows the failure and falls back
to **meta-init zeros** — the model "loads," then generates pure noise for the entire
walltime. You lose the whole job to garbage and it looks like it ran.

Belt-and-suspenders — set all of these on every compute-node job (the offline-mode rule):

```bash
# in the sbatch:
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```
```python
# in Python, before/at load:
import os; os.environ.setdefault("HF_HUB_OFFLINE", "1")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, revision=REVISION, local_files_only=True,   # <- mandatory
    dtype=torch.bfloat16, low_cpu_mem_usage=True)
proc  = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
```

The login-node proxy (for pre-caching only) looks like:
`export http_proxy=http://<squid-proxy-ip>:3128 https_proxy=http://<squid-proxy-ip>:3128`
(+ upper-case variants, `no_proxy=localhost,127.0.0.1`). Set it explicitly in the
preload script; do not rely on inheritance.

## 5. Partitions, GPUs, walltime, queue hygiene

- **`gpu`** — 8h hard walltime cap. **`gpu-short`** — ≤2h, good for smoke/pre-flight.
- Request the H200 explicitly: `--gres=gpu:h200:1`. Bill to `--account=<slurm-account>`.
- **Concurrent-job limit varies by QOS/account** — ~4 running GPU jobs on some,
  a hard `QOSMaxSubmitJobPerUserLimit=2` on others. Don't assume: check with
  `sacctmgr show assoc user=$USER format=account,qos` and observe; stage
  submissions instead of bulk-submitting into a rejection.
- **Job needs more than the 8h cap?** Don't fight the partition — design the job
  resumable (atomic per-item checkpoints, exit-0-if-done; `long-job-patterns.md` §2)
  and **chain it**: resubmit the same sbatch and it continues from checkpoints.
  Queue the successor at submit time with `--dependency=afterany:<jobid>` (not
  `afterok` — a TIMEOUT exits non-zero and `afterok` never fires, §5 below).
- On TIMEOUT, exit is non-zero and `afterok` deps won't fire (`afternotok` will). For
  resumable sweeps, write per-item outputs idempotently so a resume skips done work.
- `cmd | head` under `set -o pipefail` can SIGPIPE-kill the job — use
  `cmd 2>&1 | head -n N || true`.

## 6. Canonical sbatch anatomy

Use `assets/job_template.sbatch` — it encodes everything above. Its skeleton:

```
#SBATCH header   → job-name, --partition=gpu, --account, --gres=gpu:h200:1,
                   --time (≤ partition cap + margin), --mem, --output/--error to
                   $ALLOC/.../logs/<job>-%j.{out,err}, --export=ALL
set -e -o pipefail
modules + conda  → module load $MODULES; PATH-export; $PYTHON
cache + offline  → HF_HOME/TORCH_HOME; HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1
determinism      → CUBLAS_WORKSPACE_CONFIG
sanity           → nvidia-smi | head; torch.cuda.is_available() assert
pre-flight gate  → assert the model snapshot exists in HF_CACHE (else exit 2)  [G1 in spirit]
SIGTERM trap     → run a POST_SAVE_HOOK to checkpoint before a walltime kill   [persistent-job rule]
the work         → $PYTHON code/<script>.py --args ...
```

## 7. Loader gotchas

- **bf16 only.** Never int4/int8; never quantize the KV cache. `do_sample=False`
  (greedy) for diagnostics; pin and log every seed.
- **Attention impl:** `attn_implementation="eager"` when extracting attention/doing
  interpretability; `"sdpa"` for routine generation.
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
  of one bricks the other. A one-line comment per pin ("cuda/12.3 because torch X
  wheels", "engine Y because driver Z") turns a future debugging session into a read.
- **New model on a pinned engine? Check the engine's model registry FIRST** — before
  designing any experiment around the model. A serving engine given an architecture
  it doesn't support may silently fall back to a related implementation and die deep
  inside its own startup profiling with a misleading shape error (cost: an hour of
  confusion, minimum). One import of the engine's model registry on a compute node
  answers it in seconds.

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

# (b) Minimal canary: no model, tiny --mem, short --time, trivial payload.
#     If the SAME script succeeds on a DIFFERENT node, the original node(s) are bad.
ssh <alias> 'srun -p gpu-short --time=00:02:00 --mem=2G --gres=gpu:h200:1 \
    --account=<slurm-account> bash -lc "hostname; nvidia-smi -L; echo canary-ok"'

# (c) Rule out SYSTEMIC causes so you don't mis-fix a node problem:
ssh <alias> 'scontrol show reservations'      # free slots reserved for another partition?
ssh <alias> 'scontrol show config | grep -i preempt'   # PreemptMode
ssh <alias> 'sacctmgr show qos -P'            # QOS limits
#  + confirm the --output dir is writable, and rule out any cron/watchdog killing jobs.
```

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
- **If you time out near the end, do NOT re-run on GPU.** Re-aggregate **CPU-only on the
  login node** from the per-item files — the per-item results already exist, so the
  final table costs seconds, not another H200-hour. (This is the same per-item /
  idempotent pattern as the resumable-sweep note in §5.)

**(b) "Idle" does not mean "available" on a short partition.** A `gpu-short`-style
partition often shares physical nodes with the main `gpu` partition, and free slots can
be **reserved for higher-priority partitions** — so even a tiny job **pends** or gets
**launch-killed** (§9.3) despite the node looking idle. Before assuming availability,
check:
```bash
ssh <alias> 'scontrol show reservations'
ssh <alias> 'scontrol show config | grep -i preempt'   # PreemptMode
```

**(c) Size `--time` from the smoke, and don't undercut it.** Shorter walltime backfills
faster and schedules sooner, but estimate the real runtime from the **per-item time
measured in the G1 smoke** × item count, plus margin for load/aggregation. Undercutting
that estimate just buys you a §9.4(a) timeout.

**G5 classification:** a walltime-timeout of code that did its real work, and
partition-contention pends/launch-kills, are **infrastructure** — they do **not** count
toward the failure streak.

### 9.5 What counts as a G5 streak failure (L5 — reinforced)
The failure modes in §9.1, §9.3, and §9.4 — **connection auth failures, node
launch-kills, walltime-timeouts of working code, and partition contention** — are
**infrastructure**, not agent-code bugs. They must **not** increment the G5
consecutive-failure streak (`.failure_streak.json`). Record each explicitly with
`counted_in_streak: false`. Only a genuinely new bug in agent-written code (e.g. an OOM
you introduced in preprocessing, a wrong wrapper, a logic error) increments the streak.
When in doubt, ask: *"did my code have a bug, or did the cluster misbehave?"* See
`autonomy-model.md` §G5 for the canonical rule and ledger shape.
