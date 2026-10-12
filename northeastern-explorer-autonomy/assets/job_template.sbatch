#!/bin/bash
# ============================================================================
# Canonical Northeastern Explorer GPU job template.
# Fill in the <ANGLE-BRACKET> placeholders and the Customization Points marked
# CUSTOMIZE. Encodes: pinned modules, PATH-export env activation, mandatory
# offline mode, cache paths on the persistent allocation, cuBLAS determinism,
# a pre-flight model-cache gate, and an early-signal checkpoint hook.
# Submit lists via the environment (IDX=1,2,3 sbatch ...), not --export=ALL,IDX=1,2,3:
# --export splits on commas.
# See references/cluster-setup.md for the reasoning behind every line.
# ============================================================================
#SBATCH --job-name=<jobname>              # or set at submit time: sbatch --job-name=...
#SBATCH --partition=gpu                  # 8h cap; use gpu-short (<=2h) for smoke/pre-flight
#SBATCH --account=<slurm-account>                 # CUSTOMIZE: SLURM_ACCOUNT
#SBATCH --gres=gpu:<type>:1               # CUSTOMIZE per cluster-setup.md s5.1 (h200 | a100 | v100-sxm2 | v100-pcie | t4)
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=48G                        # from the smoke's measured peak + margin; node RAM can bind before GPUs (cluster-setup.md s5.1)
#SBATCH --time=02:00:00                  # <= partition cap; include ~30% margin; omitted --time = partition default (gpu 4 h, gpu-short 1 h), not the max
#SBATCH --output=/projects/<group>/<user>/<proj>/logs/%x-%j.out   # CUSTOMIZE: ALLOC (#SBATCH lines are not
#SBATCH --error=/projects/<group>/<user>/<proj>/logs/%x-%j.err    #  shell-expanded; %x = job name, %j = job id)
#SBATCH --signal=B:USR1@600              # USR1 to the batch shell 10 min before the time limit
#SBATCH --export=ALL

set -e
set -o pipefail
echo "=== <jobname> start: $(date) on $(hostname) ==="

# --- Customization Points (edit once) --------------------------------------
ALLOC=/projects/<group>/<user>            # CUSTOMIZE
ENV=$ALLOC/conda_envs/<conda-env>                       # CUSTOMIZE: CONDA_ENV
PROJ=$ALLOC/<proj>                                    # CUSTOMIZE: repo on the cluster
MODEL_DIR_GLOB="$ALLOC/hf_cache/hub/models--<org>--<model>"  # CUSTOMIZE: pre-cached snapshot
                                                            # (HF Hub cache layout, e.g. models--meta--Llama-3.1-8B;
                                                            #  point elsewhere if you load from local files / another source)

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

# --- Sanity: GPU + torch ----------------------------------------------------
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>&1 | head -5 || true
# A one-element CUDA op, not just is_available(): an env built without this GPU's
# architecture passes is_available() and then dies with "no kernel image".
# bf16 from compute capability (>= 8): is_bf16_supported() also says True on T4/V100 (emulation).
$PYTHON -c "import torch; assert torch.cuda.is_available(); x=torch.ones(1,device='cuda'); assert (x+1).item()==2; print('torch', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'bf16' if torch.cuda.get_device_capability(0)[0] >= 8 else 'no-bf16')"

# GPU utilisation every 5 min in the log, so an idle phase shows long before IdleBot's 1 h cancel (cluster-setup.md s5)
( while sleep 300; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | sed "s/^/gpu-util: /"; done ) &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

# --- Pre-flight gate: the pinned snapshot must be complete in the cache ------
# A bare folder check passes on a half-downloaded or wrong-revision cache.
SNAP="$MODEL_DIR_GLOB/snapshots/<revision>"                 # CUSTOMIZE: REVISION (commit sha)
[ -f "$SNAP/config.json" ] && ls "$SNAP"/*.safetensors "$SNAP"/*.bin >/dev/null 2>&1 || {
  echo "FATAL: pinned snapshot incomplete in HF cache. Pre-cache in a CPU job (srun -p short) first."
  exit 2
}

# --- The work, with an early checkpoint signal --------------------------------
# Per-item atomic checkpoints are the primary defence against the time limit. At the
# limit Slurm SIGTERMs python too, so a SIGTERM trap here would run after python's
# state is gone. Instead: run python in the background and forward the early USR1
# (from --signal above); the script must checkpoint on SIGUSR1. bash defers traps
# while a foreground child runs, hence & + wait. The RC/TRC bookkeeping keeps
# `set -e` from aborting when the trap interrupts `wait`, and preserves python's exit code.
cd "$PROJ"
echo "=== running main script ==="
$PYTHON code/<script>.py --arg value &   # CUSTOMIZE
PY=$!
RC=0; SIG=0; TRC=0
trap 'echo "=== USR1 at $(date): forwarding checkpoint signal ==="; SIG=1; kill -USR1 $PY; wait $PY && TRC=0 || TRC=$?' USR1
wait $PY || RC=$?
if [ "$SIG" = 1 ]; then RC=$TRC; fi
[ "$RC" = 0 ] || { echo "=== python exited $RC ==="; exit "$RC"; }

echo "=== <jobname> done: $(date) ==="
