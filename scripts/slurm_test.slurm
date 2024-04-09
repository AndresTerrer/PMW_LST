#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=slurm_test
#SBATCH --output=slurm_test.out
#SBATCH --exclude=nodo17
 
module load Anaconda3
conda activate PWM_LST
 
python -u slurm_test.py