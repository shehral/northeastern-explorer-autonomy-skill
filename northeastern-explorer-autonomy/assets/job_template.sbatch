#!/bin/bash
# ============================================================================
# Canonical Northeastern Explorer GPU job template.
# Fill in the <ANGLE-BRACKET> placeholders and the Customization Points marked
# CUSTOMIZE. Encodes: pinned modules, PATH-export env activation, mandatory
# offline mode, cache paths on the persistent allocation, cuBLAS determinism,
# a pre-flight model-cache gate, and a SIGTERM save-hook for walltime kills.
# See references/cluster-setup.md for the reasoning behind every line.
# ============================================================================
#SBATCH --job-name=<jobname>
#SBATCH --partition=gpu                  # 8h cap; use gpu-short (<=2h) for smoke/pre-flight
#SBATCH --account=<slurm-account>                 # CUSTOMIZE: SLURM_ACCOUNT
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=48G                        # pad for logit/activation buffers
#SBATCH --time=02:00:00                  # <= partition cap; include ~30% margin
#SBATCH --output=/projects/<group>/<user>/<proj>/logs/<jobname>-%j.out   # CUSTOMIZE: ALLOC
#SBATCH --error=/projects/<group>/<user>/<proj>/logs/<jobname>-%j.err
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

# --- Cache paths (on the persistent allocation, never /tmp) -----------------
export HF_HOME=$ALLOC/hf_cache
export TORCH_HOME=$ALLOC/torch_cache
mkdir -p "$HF_HOME" "$TORCH_HOME"

# --- Offline mode (MANDATORY on every GPU job; see cluster-setup.md s4) -----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
$PYTHON -c "import torch; assert torch.cuda.is_available(); x=torch.ones(1,device='cuda'); assert (x+1).item()==2; print('torch', torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'bf16' if torch.cuda.is_bf16_supported() else 'no-bf16')"

# --- Pre-flight gate: the pinned snapshot must be complete in the cache ------
# A bare folder check passes on a half-downloaded or wrong-revision cache.
SNAP="$MODEL_DIR_GLOB/snapshots/<revision>"                 # CUSTOMIZE: REVISION (commit sha)
[ -f "$SNAP/config.json" ] && ls "$SNAP"/*.safetensors "$SNAP"/*.bin >/dev/null 2>&1 || {
  echo "FATAL: pinned snapshot incomplete in HF cache. Pre-cache in a CPU job (srun -p short) first."
  exit 2
}

# --- SIGTERM save-hook: checkpoint before a walltime kill (persistent-job rule) ----
POST_SAVE_HOOK="${POST_SAVE_HOOK:-}"
_on_term() {
  echo "=== SIGTERM at $(date) — running POST_SAVE_HOOK if set ==="
  [ -n "$POST_SAVE_HOOK" ] && bash -c "$POST_SAVE_HOOK" || echo "(no/failed save hook — non-fatal)"
  exit 1
}
trap _on_term SIGTERM SIGINT

# --- The work ---------------------------------------------------------------
cd "$PROJ"
echo "=== running main script ==="
$PYTHON code/<script>.py --arg value     # CUSTOMIZE

echo "=== <jobname> done: $(date) ==="
