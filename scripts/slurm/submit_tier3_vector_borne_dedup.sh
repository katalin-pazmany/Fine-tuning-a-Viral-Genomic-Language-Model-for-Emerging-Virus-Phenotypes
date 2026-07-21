#!/bin/bash
#SBATCH --job-name=viral_tier3_vb_dd
#SBATCH --partition=gpu-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/tier3_vb_dedup_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/tier3_vb_dedup_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi
mkdir -p /users/sgkpazma/viral-phenotype-finetuning/logs
cd /users/sgkpazma/viral-phenotype-finetuning
# Vir2vec is a public model already in the local HF cache from earlier runs. A stale
# HuggingFace token in the environment triggers a 401 (OAuth signature verification
# failed) on model load, so clear any token and force offline mode to use the cache.
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python -u pipeline/tier3_lora_vector_borne.py \
    --epochs 30 --batch_size 1 --lr 1e-4 --lora_r 8 --lora_alpha 16 --max_len 1024 --dedup
echo "Job finished: $(date)"
