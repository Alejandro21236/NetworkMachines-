#!/bin/bash
#SBATCH --job-name=drt_surface
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=400G
#SBATCH --output=drt_surface.out
#SBATCH --error=drt_surface.err

set -euo pipefail

# =========================
# USER-CONFIGURABLE PATHS
# =========================
PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
WORKDIR=${WORKDIR:-${PROJECT_ROOT}/machine}
SCRIPT=${SCRIPT:-machine4.py}

PATCHES_DIR=${PATCHES_DIR:-/path/to/patches}
LABELS_CSV=${LABELS_CSV:-/path/to/labels.csv}
OUTROOT=${OUTROOT:-${PROJECT_ROOT}/outputs/drt_surface}

# Optional RT utilities
RT_ROOT=${RT_ROOT:-${PROJECT_ROOT}/RT3}
export RT3_PATH=${RT3_PATH:-${RT_ROOT}/RT3.py}

# =========================
# ENVIRONMENT
# =========================
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8.0 || true
fi

export CUDA_VISIBLE_DEVICES=0

# ---- memory / thread limits ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# =========================
# CHECKS
# =========================
need(){ [[ -e "$1" ]] || { echo "Missing: $1" >&2; exit 2; }; }
need "${WORKDIR}/${SCRIPT}"
need "${PATCHES_DIR}"
need "${LABELS_CSV}"

mkdir -p "${OUTROOT}"
cd "${WORKDIR}"

# =========================
# RUN
# =========================
srun -u -N1 -n1 --gpus-per-task=1 --kill-on-bad-exit=1 \
  python -u "${SCRIPT}" \
    --patches_dir "${PATCHES_DIR}" \
    --labels_csv "${LABELS_CSV}" \
    --out_dir "${OUTROOT}" \
    --epochs 20 \
    --batch_size 1 \
    --max_patches 16 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --freeze_blocks 9 \
    --seed 42 \
    --layers "3" \
    --graph_topk 15 \
    --ref_batches 1 \
    --cap_patches 2000 \
    --cap_features 512 \
    --save_spatial \
    --spatial_examples 1 \
    --attr_top_frac 0.10

echo "All done."

