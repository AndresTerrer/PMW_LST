#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=windsat_download
#SBATCH --output=windsat_download.out
#SBATCH --exclude=nodo17
 
module load Anaconda3
source activate PWM_LST
 
python -u windsat_download.py --start-date 2017-01-01 --end-date 2017-12-31 --save-into /data/databases/Windsat/Daily_TBToA