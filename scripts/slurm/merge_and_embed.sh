#!/bin/bash
#SBATCH --job-name=viral_embed_all
#SBATCH --partition=gpu-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/embed_all_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/embed_all_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi
mkdir -p /users/sgkpazma/viral-phenotype-finetuning/logs
cd /users/sgkpazma/viral-phenotype-finetuning

echo "Merging all FASTA files..."
cat data/raw_matched/*.fasta > data/all_sequences.fasta
echo "Total sequences:"
grep -c ">" data/all_sequences.fasta

echo "Running embedding..."
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python -u scripts/batch_embed_refseq.py

echo "Job finished: $(date)"
