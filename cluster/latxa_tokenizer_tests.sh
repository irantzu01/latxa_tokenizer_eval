#!/bin/bash
#SBATCH --job-name=latxa-test
#SBATCH --output=logs/train/train_%A_%a.log
#SBATCH --error=logs/train/train_%A_%a.err
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:A100:1
#SBATCH --time=02:00:00
#SBATCH --partition=react
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -C inet

echo "Starting job $SLURM_JOB_ID on node $(hostname)"

# === 1. LOAD MODULES (MAY VARY BY CLUSTER) ===
module purge
module load cuda/12.1  # example, use what your cluster has
module load anaconda/2023  # if available

# === 2. ACTIVATE YOUR VIRTUAL ENVIRONMENT ===
source ~/MASTER/WiSe25/Lab\ Rotation/dynamic-tokenization/dynamic_tokenization_311/bin/activate

# Alternatively, if using conda:
# conda activate dynamic-tokenization

echo "Environment loaded."

# === 3. MOVE TO PROJECT DIRECTORY ===
cd ~/MASTER/WiSe25/Lab\ Rotation/latxa_tokenizer_eval

echo "Running python experiment..."

# === 4. RUN YOUR PYTHON SCRIPT ===
python scripts/test_cluster.py

echo "Job finished."
