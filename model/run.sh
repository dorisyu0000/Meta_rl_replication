#!/bin/bash
#SBATCH --job-name=driftingbandit
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH -e /home/my2689/.ssh/stderr/slurm-%A_%a.err
#SBATCH -o /home/my2689/.ssh/stdout/slurm-%A_%a.out
#SBATCH --array=0-99

python -u training.py --jobid=$SLURM_ARRAY_TASK_ID