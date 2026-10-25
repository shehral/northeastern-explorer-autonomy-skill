#!/bin/bash
# ============================================================================
# Canonical Northeastern Explorer GPU job template.
# Fill in the <ANGLE-BRACKET> placeholders and the Customization Points marked
# CUSTOMIZE. Encodes: pinned modules, PATH-export env activation, mandatory
# offline mode, cache paths on the persistent allocation, cuBLAS determinism,
# the per-node driver gate, a pre-flight model-cache gate, and an early stop
# signal (a stop file; USR1 forwarded only on opt-in).
# Submit lists via the environment (IDX=1,2,3 sbatch ...), not --export=ALL,IDX=1,2,3:
# --export splits on commas.
# Shared or committed job file? Keep site values out of it (SKILL.md Customization
# Points): delete the --account/--output/--error lines, put SBATCH_ACCOUNT,
# SBATCH_OUTPUT, SBATCH_ERROR, ALLOC (and the proxy) in a gitignored local.env as
# `export` lines, create the log dir (Slurm does not), submit with
# `. ./local.env && sbatch <file>`, and replace the ALLOC line below with:
#   . "${SLURM_SUBMIT_DIR}/local.env"; ALLOC=${ALLOC:?set ALLOC in local.env}
#
# Exit codes (the G5 triage table in references/autonomy-model.md reads them):
#   0  work complete, or already complete (skip)
#   75 stopped early on the wall signal or the stop file after writing a checkpoint:
#      resubmit to resume (afternotok insurance fires on it; resume chains use afterany)
#   4  node or driver problem, caught by the pre-flight before any work (driver below
#      the floor, or a GPU allocated but its driver unreadable): infrastructure;
#      --exclude the node and resubmit
#   2  any other pre-flight refusal (no GPU allocated: missing --gres?; missing cache,
#      env not marked built, a guard): agent design unless the cause is shown to be
#      infrastructure
#   3  a self-check failed after work started
#   other: the program's own status. The program's 2, 3 and 4 are reserved: they are
#      remapped to 1 (python itself exits 2 on a usage error or a missing script)
#      unless the program opts into this contract with PROGRAM_FOLLOWS_EXIT_CONTRACT=1.
# See references/cluster-setup.md for the reasoning behind every line.
# ============================================================================
#SBATCH --job-name=<jobname>              # or set at submit time; start it with your project's prefix (autonomy-model.md)
#SBATCH --partition=gpu                  # 8h cap; use gpu-short (<=2h) for smoke/pre-flight
#SBATCH --account=<slurm-account>                 # CUSTOMIZE: SLURM_ACCOUNT
#SBATCH --gres=gpu:<type>:1               # CUSTOMIZE per cluster-setup.md s5.1 (h200 | a100 | v100-sxm2 | v100-pcie | t4 | p100;
                                          #  H100, H200 NVL, L40S, L40, A6000: exact name from `sinfo -o '%P %N %G'`).
                                          #  Keep a default even if you pick the type at submit (a command-line --gres
                                          #  overrides it): a GPU job without one lands on a CPU node.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=48G                        # from the smoke's measured peak + margin; node RAM can bind before GPUs (cluster-setup.md s5.1)
#SBATCH --time=02:00:00                  # <= partition cap; include ~30% margin; omitted --time = partition default (gpu 4 h, gpu-short 1 h), not the max
#SBATCH --output=/projects/<group>/<user>/<proj>/logs/%x-%j.out   # CUSTOMIZE: ALLOC (#SBATCH lines are not
#SBATCH --error=/projects/<group>/<user>/<proj>/logs/%x-%j.err    #  shell-expanded; %x = job name, %j = job id)
#SBATCH --signal=B:USR1@600              # USR1 to the batch shell 10 min before the time limit
#SBATCH --export=ALL

set -e
set -o pipefail
# Early stop trap first: a USR1 that arrived before any trap would kill the batch shell
# (bash's default action) with no stop file and no exit 75. Until python starts, it only
# writes the stop file; the launch below then exits 75 without starting work.
# Manual early stop from the login node: scancel --signal=USR1 --batch <jobid>
export STOP_FILE=/tmp/stop-${SLURM_JOB_ID:-$$}; rm -f "$STOP_FILE"
trap ': > "$STOP_FILE" || true' USR1
echo "=== <jobname> start: $(date) on $(hostname) ==="

# --- Customization Points (edit once) --------------------------------------
ALLOC="/projects/<group>/<user>"                       # CUSTOMIZE
ENV="$ALLOC/conda_envs/<conda-env>"                    # CUSTOMIZE: CONDA_ENV
PROJ="$ALLOC/<proj>"                                   # CUSTOMIZE: repo on the cluster
MODEL_DIR_GLOB="$ALLOC/hf_cache/hub/models--<org>--<model>"  # CUSTOMIZE: pre-cached snapshot
                                                            # (HF Hub cache layout, e.g. models--meta--Llama-3.1-8B;
                                                            #  point elsewhere if you load from local files / another source)
DONE="$PROJ/results/<run-id>/DONE"                     # CUSTOMIZE: the marker your script writes last
FORWARD_USR1=${FORWARD_USR1:-0}                        # CUSTOMIZE: 1 only if the script installs a SIGUSR1 handler
PROGRAM_FOLLOWS_EXIT_CONTRACT=${PROGRAM_FOLLOWS_EXIT_CONTRACT:-0}  # CUSTOMIZE: 1 only if the script's own exits 2/3/4
                                                       # mean what the header says (else they are remapped to 1)

# Already complete: exit 0 before spending anything, so a resubmit or a fired
# insurance chunk is a no-op (long-job-patterns.md s2).
[ -f "$DONE" ] && { echo "=== already complete ($DONE): skip ==="; exit 0; }

# --- Modules + conda (PATH-export, NOT `source activate`) -------------------
source /etc/profile
module purge
module load miniconda3/24.11.1                        # Explorer-wide pin. No cuda/* module: pip torch wheels carry
                                                      # their own runtime (cluster-setup.md s3); load one only to
                                                      # compile CUDA extensions, matching the wheel's CUDA version.
export PATH=$ENV/bin:$PATH
PYTHON=$ENV/bin/python                                # always use $PYTHON, never bare `python`
[ -f "$ENV/.built-ok" ] || { echo "FATAL: env $ENV not marked built (.built-ok)"; exit 2; }   # cluster-setup.md s3

# --- Cache paths (on the persistent allocation, never /tmp) -----------------
export HF_HOME=$ALLOC/hf_cache
export TORCH_HOME=$ALLOC/torch_cache
mkdir -p "$HF_HOME" "$TORCH_HOME"

# --- Offline mode (MANDATORY on every GPU job; see cluster-setup.md s4) -----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
# Jobs inherit the login shell's proxy (--export=ALL). GPU jobs stay offline, and localhost servers must bypass it.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

# --- Deterministic cuBLAS ---------------------------------------------------
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- Long-fit hygiene (see long-job-patterns.md) -----------------------------
# expandable_segments prevents fragmentation OOM on multi-hour fits with
# many transient allocations; progress bars off keeps captured logs readable
# (a single scripted HF transfer can otherwise dump ~1 MB of \r-bars).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_PROGRESS_BARS=1

# --- Driver gate, before any Python (cluster-setup.md s3 driver gate) --------
# Drivers differ by node, not only by GPU type. The NODE_DRIVER line feeds your
# per-node driver table (assets/fold_drivers.py folds it from job logs).
# Evidence (field-observed 2026-10): 570.86 runs cu13x with the 13.0 compat libcuda;
# 595.58 breaks with it and works without; cu130 fails on 545.23. The 580 cut-off is
# inferred from NVIDIA's forward-compat rule, the 570 floor is a choice, and both
# exit-4 branches (below the floor; GPU allocated but driver unreadable) are new and
# untested on the cluster.
CU13=0                                                # CUSTOMIZE: 1 if the env's torch is a cu13x build
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)
DRV_MAJOR=${DRV%%.*}
case "$DRV_MAJOR" in ''|*[!0-9]*) DRV=none; GPU_NAME=none;; esac
echo "NODE_DRIVER node=$(hostname -s) gpu=${GPU_NAME:-none} driver=$DRV"
if [ "$DRV" = none ]; then
  # Was a GPU allocated? The job record is authoritative; Slurm's GPU variables are
  # the fallback when scontrol is unavailable. (Not CUDA_VISIBLE_DEVICES: --export=ALL
  # can carry it over from the submit shell.)
  TRES=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | grep -o 'AllocTRES=[^ ]*' | head -n 1 || true)
  case "$TRES" in
    *gres/gpu*) GPU_ALLOC=1;;
    AllocTRES=*) GPU_ALLOC=0;;
    *) GPU_ALLOC=0; [ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" ] || GPU_ALLOC=1;;
  esac
  if [ "$GPU_ALLOC" = 1 ]; then
    echo "FATAL: GPU allocated but driver unreadable on $(hostname -s) (node problem); --exclude=$(hostname -s) and resubmit"; exit 4
  fi
  echo "FATAL: no GPU allocated and no driver visible on $(hostname -s) (missing --gres?)"; exit 2
fi
if [ "$CU13" = 1 ]; then
  [ "$DRV_MAJOR" -ge 570 ] || { echo "FATAL: driver $DRV below the 570 floor for cu13x; --exclude=$(hostname -s) and resubmit"; exit 4; }
  if [ "$DRV_MAJOR" -lt 580 ]; then
    export LD_LIBRARY_PATH=$ALLOC/cuda-compat/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  fi
fi

# --- Sanity: GPU + torch ----------------------------------------------------
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv 2>&1 | head -5 || true
# A one-element CUDA op, not just is_available(): an env built without this GPU's
# architecture passes is_available() and then dies with "no kernel image".
# bf16 from compute capability (>= 8): is_bf16_supported() also says True on T4/V100 (emulation).
$PYTHON -c "import torch; assert torch.cuda.is_available(); x=torch.ones(1,device='cuda'); assert (x+1).item()==2; print('torch', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'bf16' if torch.cuda.get_device_capability(0)[0] >= 8 else 'no-bf16')"

# GPU utilisation every 5 min in the log, so an idle phase shows long before IdleBot's 1 h cancel (cluster-setup.md s5).
# set +e: the subshell inherits set -e, and one failed nvidia-smi would end the sampler.
# Every trap must end in a command that cannot fail: under set -e a failing kill in a
# trap turns a successful job into exit 1 (FAILED, afterok children never fire).
( set +e; while sleep 300; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sed "s/^/gpu-util: /"; done ) &
WATCH=$!
trap 'kill "$WATCH" 2>/dev/null || true' EXIT

# --- Pre-flight gate: the pinned snapshot must be complete in the cache ------
# A bare folder check passes on a half-downloaded or wrong-revision cache. (Not `ls a b`:
# it fails when either glob is empty, so a safetensors-only snapshot would be refused.)
SNAP="$MODEL_DIR_GLOB/snapshots/<revision>"                 # CUSTOMIZE: REVISION (commit sha)
[ -f "$SNAP/config.json" ] && { compgen -G "$SNAP/*.safetensors" || compgen -G "$SNAP/*.bin"; } >/dev/null || {
  echo "FATAL: pinned snapshot incomplete in HF cache. Pre-cache it in a CPU job via sbatch first (sharing if it fits in 1 h, short beyond)."
  exit 2
}

# --- The work, with an early stop signal --------------------------------------
# Per-item atomic checkpoints are the primary defence against the time limit; save
# every 50-100 steps so a lost signal costs at most one interval. At the limit Slurm
# SIGTERMs python too, so a SIGTERM trap here would run after python's state is gone.
# On the early USR1 (--signal above) the trap writes $STOP_FILE. The script polls
# os.path.exists(os.environ["STOP_FILE"]) every step; when it appears, it writes a
# checkpoint and exits 75 (0 if the work is in fact complete). Exit 0 after an early
# stop would let kill_invalid_depend cancel an afternotok insurance chunk and start an
# afterok successor on partial work.
# USR1 is forwarded to python only with FORWARD_USR1=1, for a script that installs a
# SIGUSR1 handler: python's default SIGUSR1 action terminates the process before it can
# checkpoint. Forwarding alone was not reliable either: in field runs (2026-10) python
# sometimes never saw it.
# Run python directly (no `| tee`, no `bash -c` wrapper). bash defers traps while a
# foreground child runs, hence & + wait. The RC/TRC bookkeeping keeps `set -e` from
# aborting when the trap interrupts `wait`, and preserves python's exit code.
# STOP_FILE and the early USR1 trap are set at the top of the script.
cd "$PROJ"
[ ! -f "$STOP_FILE" ] || { echo "=== USR1 before the work started: no work done, exit 75: resubmit ==="; exit 75; }
echo "=== running main script ==="
$PYTHON "code/<script>.py" --arg value &   # CUSTOMIZE
PY=$!
RC=0; SIG=0; TRC=0
trap 'echo "=== USR1 at $(date): stop file written ==="; SIG=1; : > "$STOP_FILE" || true; if [ "$FORWARD_USR1" = 1 ]; then kill -USR1 "$PY" 2>/dev/null || true; fi; wait "$PY" && TRC=0 || TRC=$?' USR1
wait "$PY" || RC=$?
if [ "$SIG" = 1 ]; then RC=$TRC; fi
case "$RC" in
  0) ;;
  75) echo "=== stopped early with a checkpoint (exit 75): resubmit to resume ==="; exit 75;;
  2|3|4)
    if [ "$PROGRAM_FOLLOWS_EXIT_CONTRACT" = 1 ]; then echo "=== python exited $RC (contract code) ==="; exit "$RC"; fi
    echo "=== python exited $RC (reserved contract code, remapped to 1; PROGRAM_FOLLOWS_EXIT_CONTRACT=0) ==="; exit 1;;
  *) echo "=== python exited $RC ==="; exit "$RC";;
esac

echo "=== <jobname> done: $(date) ==="
