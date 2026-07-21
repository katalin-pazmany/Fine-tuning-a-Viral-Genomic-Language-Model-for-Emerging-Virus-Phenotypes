#!/bin/bash
#SBATCH --job-name=search_unmatched
#SBATCH --partition=nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/search_unmatched_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/search_unmatched_%j.err

echo "Job started: $(date)"
cd /users/sgkpazma/viral-phenotype-finetuning
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python -u scripts/search_unmatched_sequences.py
echo "Job finished: $(date)"
