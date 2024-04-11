#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=windsat_data_exploration
#SBATCH --output=windsat_data_exploration.out
#SBATCH --exclude=nodo17

module load Anaconda3
source activate PWM_LST
 
python -u windsat_data_exploration.py --source-folder /data/databases/Windsat/Daily_TBToA --save-folder /data/databases/Windsat/processed