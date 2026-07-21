#!/bin/bash
#SBATCH --job-name=viral_tier2_vb_dd
#SBATCH --partition=gpu-l40s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/tier2_vb_dedup_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/tier2_vb_dedup_%j.err

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
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python -u pipeline/tier2_partial_unfreeze_vector_borne.py --epochs 20 --batch_size 2 --lr 2e-5 --dedup
echo "Job finished: $(date)"
