#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=wstmodel1
#SBATCH --output=wstmodel1.out
#SBATCH --exclude=nodo17
 
module load Anaconda3
source activate PMW_LST
 
python -u scripts/wstmodel1.py --soruce_folder /data/databases/Windsat/Daily_TBToA --output_folder /home/antego/PMW_LST/models