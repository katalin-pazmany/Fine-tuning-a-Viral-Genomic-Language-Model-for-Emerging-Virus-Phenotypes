#!/bin/bash
#SBATCH --job-name=download_seqs
#SBATCH --partition=nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/download_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/download_%j.err

echo "Job started: $(date)"
cd /users/sgkpazma/viral-phenotype-finetuning
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python -u scripts/download_original_sequences.py
echo "Job finished: $(date)"
