#!/bin/bash
#SBATCH --job-name=viral_tier2_pu_dd
#SBATCH --partition=gpu-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/tier2_pu_dedup_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/tier2_pu_dedup_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
mkdir -p /users/sgkpazma/viral-phenotype-finetuning/logs
cd /users/sgkpazma/viral-phenotype-finetuning
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python -u pipeline/tier2_partial_unfreeze.py --task both --epochs 20 --batch_size 8 --lr 2e-5 --dedup
echo "Job finished: $(date)"
