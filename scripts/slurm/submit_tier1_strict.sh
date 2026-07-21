#!/bin/bash
#SBATCH --job-name=viral_tier1_strict
#SBATCH --partition=nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/tier1_strict_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/tier1_strict_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
mkdir -p /users/sgkpazma/viral-phenotype-finetuning/logs
cd /users/sgkpazma/viral-phenotype-finetuning
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python pipeline/tier1_pipeline.py --strict
echo "Job finished: $(date)"
