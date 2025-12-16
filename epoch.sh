#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=job_epoch_video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --output=job_epoch_video.out
#SBATCH --error=job_epoch_video.err

set -euo pipefail

# ---- PATHS ----
WORKDIR="/path/to/your/workdir"                 # e.g., /fs/scratch/<ACCOUNT>/<USER>/<PROJECT>
SCRIPT="epoch_l3_param_video.py"
PATCHES_DIR="/path/to/patches"                  # e.g., /fs/scratch/<ACCOUNT>/<DATASET>/patches
LABELS_CSV="/path/to/labels.csv"                # e.g., /fs/scratch/<ACCOUNT>/<USER>/datasets/labels.csv
OUTROOT="/path/to/output_dir"                   # e.g., /fs/scratch/<ACCOUNT>/<USER>/<PROJECT>/out_epoch_video

# ---- ENV ----
module load cuda/11.8.0 || true
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# keep BLAS/OpenMP threads low
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=1

# make sure code can import RT3.py from WORKDIR
# supports both WORKDIR/RT3.py and WORKDIR/RT3/RT3.py layouts
export PYTHONPATH="${WORKDIR}:${WORKDIR}/RT3:${PYTHONPATH:-}"

mkdir -p "${OUTROOT}"
cd "${WORKDIR}"

# ---- RUN ----
srun -u -N1 -n1 --gpus-per-task=1 --kill-on-bad-exit=1 \
  python -u "${SCRIPT}" \
    --patches_dir "${PATCHES_DIR}" \
    --labels_csv "${LABELS_CSV}" \
    --out_dir "${OUTROOT}" \
    --batch_size 1 \
    --max_patches 64 \
    --lr 5e-4 \
    --seed 42 \
    --rank_cap 128 \
    --frameskip 1

echo "Done: video and diagnostics in ${OUTROOT}"
