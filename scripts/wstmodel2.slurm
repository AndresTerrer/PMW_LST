#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=wstmodel2
#SBATCH --output=wstmodel2.out
#SBATCH --exclude=nodo17
 
module load Anaconda3
source activate PMW_LST
 
python -u /home/antego/PMW_LST/scripts/wstmodel1.py --source_folder /data/databases/Windsat/Daily_TBToA --output_folder /home/antego/PMW_LST/models/ --telsem_folder /data/databases/Windsat/NetcdfTELSEM/