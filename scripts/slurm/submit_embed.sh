#!/bin/bash
#SBATCH --job-name=viral_embed
#SBATCH --partition=gpu-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/embed_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/embed_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi
mkdir -p /users/sgkpazma/viral-phenotype-finetuning/logs
cd /users/sgkpazma/viral-phenotype-finetuning
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python -u embed_new_sequences.py
echo "Job finished: $(date)"
