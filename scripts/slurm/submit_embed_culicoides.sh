#!/bin/bash
#SBATCH --job-name=embed_culicoides
#SBATCH --partition=nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/embed_culicoides_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/embed_culicoides_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
cd /users/sgkpazma/viral-phenotype-finetuning
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python -u scripts/embed_culicoides_sequences.py
echo "Job finished: $(date)"
