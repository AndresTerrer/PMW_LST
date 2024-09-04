#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=wstmodel1_Descending
#SBATCH --output=wstmodel1_Descending.out
#SBATCH --exclude=nodo17
 
module load Anaconda3
source activate PMW_LST
 
python -u /home/antego/PMW_LST/scripts/wstmodel1.py --source_folder /data/databases/Windsat/Daily_TBToA --output_folder /home/antego/PMW_LST/models/ --swath_sector 1