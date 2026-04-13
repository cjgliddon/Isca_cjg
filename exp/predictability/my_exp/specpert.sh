#!/bin/bash

#SBATCH -p mit_normal
#SBATCH -n 16
#SBATCH -o output/specpert_%A_%a.out
#SBATCH -e output/specpert_%A_%a.err
#SBATCH --array=1-10

module load miniforge
source activate isca_env

for TSTART in {365..375..10}
do
    python specpert.py ${TSTART}.5 ${SLURM_ARRAY_TASK_ID} 1 0
done