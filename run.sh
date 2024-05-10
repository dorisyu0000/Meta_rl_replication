#!/bin/bash
#SBATCH --job-name=driftingbandit
#SBATCH --cpus-per-task=1
#SBATCH --time=35:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH -e /home/sc10264/code_bandit2/stderr/slurm-%A_%a.err
#SBATCH -o /home/sc10264/code_bandit2/stdout/slurm-%A_%a.out
#SBATCH --array=0-99

python -u training.py --jobid=$SLURM_ARRAY_TASK_ID