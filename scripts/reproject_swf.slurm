#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=SWF_reproject
#SBATCH --output=SWF_reproject.out
#SBATCH --exclude=nodo17
 
module load Anaconda3
source activate swf_reproject
 
python -u /home/antego/PMW_LST/scripts/reproject_swf.py --source_folder /data/databases/LPDR_v2_2017 --output_folder /data/databases/LPDR_v2_2017_EASEv2 