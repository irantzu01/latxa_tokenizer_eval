#!/bin/bash
#SBATCH --job-name=latxa_500k_improved
#SBATCH --output=logs/train_500k_improved_%j.out
#SBATCH --error=logs/train_500k_improved_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

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
python train_500k_improved.py

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"