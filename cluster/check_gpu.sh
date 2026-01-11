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

# Print GPU info
echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Check GPU
nvidia-smi

echo "========================================"
echo "Starting training..."
echo "========================================"

# Activate your conda/virtual environment if needed
# source activate your_env_name

# Run training
python diagnose_model.py

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"