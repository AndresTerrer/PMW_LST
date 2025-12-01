#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=extract_timeseries
#SBATCH --output=extract_timeseries.out
#SBATCH --exclude=nodo17
 
module load Anaconda3
source activate PMW_LST

# Test file access
echo "Current directory: $(pwd)"
echo "Directory contents: "
ls -l /data/databases/Windsat/Daily_TBToA

jupyter nbconvert --to notebook --execute /home/antego/LST_ATBD/LandSurfaceTemperature_ATBD/notebooks/Extract_data_series_server.ipynb --output /home/antego/LST_ATBD/LandSurfaceTemperature_ATBD/notebooks/Extract_data_series_server_output.ipynb