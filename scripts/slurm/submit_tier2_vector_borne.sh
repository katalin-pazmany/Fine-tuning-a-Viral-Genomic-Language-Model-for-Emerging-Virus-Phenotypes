#!/bin/bash
#SBATCH --job-name=viral_tier2_vb
#SBATCH --partition=gpu-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/tier2_vb_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/tier2_vb_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi
mkdir -p /users/sgkpazma/viral-phenotype-finetuning/logs
cd /users/sgkpazma/viral-phenotype-finetuning
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python -u pipeline/tier2_partial_unfreeze_vector_borne.py --epochs 20 --batch_size 2 --lr 2e-5
echo "Job finished: $(date)"
