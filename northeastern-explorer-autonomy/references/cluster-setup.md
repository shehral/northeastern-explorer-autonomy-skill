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

---

## 1. Connection

```bash
ssh <user>@login.explorer.northeastern.edu        # SSH_TARGET
```

- **ControlMaster** reuses one authenticated tunnel for ~8h across all `ssh`/`scp`/
  `rsync`. Set it in `~/.ssh/config` (a `Host explorer` alias is convenient). **Duo**
  two-factor is required only on the first cold connect or after the 8h window.
- From an autonomous session, probe non-interactively so auth failures fail fast:
  `ssh -o BatchMode=yes -o ConnectTimeout=10 explorer 'squeue -u <user>'`. If this
  hangs or errors, the human needs to re-clear Duo — surface it, don't retry blindly.
- **Stale socket** (after a WiFi/IP change): remove the control socket
  `~/.ssh/control-*explorer*` and reconnect once.

**Login-node limits:** the login node has a **~8 GB effective RAM
cap** and will SIGKILL (`exit 137`) heavy work. **Never** load the model, run a big
`du`, or do real compute there. Delegate to a compute node with `srun`/`sbatch`. The
one allowed heavy-ish login-node task is pre-caching downloads (§2), which is I/O not
RAM.

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
`export http_proxy=http://10.99.0.130:3128 https_proxy=http://10.99.0.130:3128`
(+ upper-case variants, `no_proxy=localhost,127.0.0.1`). Set it explicitly in the
preload script; do not rely on inheritance.

## 5. Partitions, GPUs, walltime, queue hygiene

- **`gpu`** — 8h hard walltime cap. **`gpu-short`** — ≤2h, good for smoke/pre-flight.
- Request the H200 explicitly: `--gres=gpu:h200:1`. Bill to `--account=<slurm-account>`.
- **Concurrent-job limit** ~4 running GPU jobs/user; extras pend gracefully.
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
