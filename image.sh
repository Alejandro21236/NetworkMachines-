#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=job_imgfreq_entropy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=job_imgfreq_entropy.out
#SBATCH --error=job_imgfreq_entropy.err

set -euo pipefail

# ---- PATHS ----
WORKDIR="/path/to/your/workdir"              # e.g., /fs/scratch/<ACCOUNT>/<USER>/<PROJECT>
SCRIPT="epoch_imgfreq_entropy.py"            # located in WORKDIR
PATCHES_DIR="/path/to/patches"               # e.g., /fs/scratch/<ACCOUNT>/<DATASET>/patches
LABELS_CSV="/path/to/labels.csv"             # e.g., /fs/scratch/<ACCOUNT>/<USER>/labels.csv
OUTROOT="/path/to/output_dir"                # e.g., /fs/scratch/<ACCOUNT>/<USER>/<PROJECT>/out_imgfreq

# ---- MODULES ----
module load cuda/11.8.0 || true
module load python/3.10-conda || true

# Optional conda env
# source activate your_env

mkdir -p "${OUTROOT}"
cd "${WORKDIR}"

# ---- RUN (single epoch; entropy / Boltzmann-compatible) ----
srun -u -N1 -n1 --gpus-per-task=1 --kill-on-bad-exit=1 \
  python -u "${SCRIPT}" \
    --patches_dir "${PATCHES_DIR}" \
    --labels_csv "${LABELS_CSV}" \
    --out_dir "${OUTROOT}" \
    --lr 5e-4 \
    --batch_size 1 \
    --max_patches 64 \
    --ref_patches 64 \
    --frameskip 1 \
    --nbins 64 \
    --pad 8 \
    --seed 42

echo "Done."

