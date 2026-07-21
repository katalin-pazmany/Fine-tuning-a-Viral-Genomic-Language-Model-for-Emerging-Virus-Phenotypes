#!/bin/bash
#SBATCH --job-name=viral_cpu_analysis
#SBATCH --partition=nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/users/sgkpazma/viral-phenotype-finetuning/logs/cpu_analysis_%j.out
#SBATCH --error=/users/sgkpazma/viral-phenotype-finetuning/logs/cpu_analysis_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
mkdir -p /users/sgkpazma/viral-phenotype-finetuning/logs
cd /users/sgkpazma/viral-phenotype-finetuning

# Put the conda env's bin/ on PATH so subprocess calls to external tools
# (makeblastdb, blastn, cd-hit-est, ...) resolve correctly — calling the
# python binary by full path alone does NOT do this, since PATH lookups
# for subprocess.run() happen against the job's shell environment, not
# the interpreter's own site-packages.
export PATH="/users/sgkpazma/.conda/envs/viral-phenotype/bin:$PATH"

# Pass the actual command to run as arguments to this script, e.g.:
#   sbatch submit_cpu_analysis.sh scripts/dedup_sequences.py --dataset vector_borne
#   sbatch submit_cpu_analysis.sh scripts/baseline_comparisons.py
#   sbatch submit_cpu_analysis.sh scripts/blast_baseline.py
/users/sgkpazma/.conda/envs/viral-phenotype/bin/python -u "$@"

echo "Job finished: $(date)"
