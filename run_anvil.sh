#!/bin/bash
#SBATCH -n 1 
#SBATCH -p gpu 
#SBATCH -A cis220117-gpu 
#SBATCH -t 00:01:00 
#SBATCH --gpus-per-node 1
#SBATCH --mail-type=all    # Send email at begin and end of job

hostname
nvidia-smi
