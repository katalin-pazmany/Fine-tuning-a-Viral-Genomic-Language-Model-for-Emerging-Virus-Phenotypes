#!/bin/bash
#SBATCH --job-name=viral_tier3_strict
#SBATCH --partition=gpu-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/tier3_strict_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/tier3_strict_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi
mkdir -p /users/sgkpazma/viral-phenotype-finetuning/logs
cd /users/sgkpazma/viral-phenotype-finetuning
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python -u pipeline/tier3_lora.py \
    --task both --epochs 30 --batch_size 1 --lr 1e-4 --lora_r 8 --lora_alpha 16 --max_len 1024 --strict
echo "Job finished: $(date)"
