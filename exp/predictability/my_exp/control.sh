#!/bin/bash

#SBATCH -p mit_normal
#SBATCH -n 16
#SBATCH -o output/control_run.out
#SBATCH -e output/control_run.err

# Submit this using the following syntax:
# jid=$(sbatch --parsable control.sh)
# sbatch --dependency=afterok:$jid control.sh

module load miniforge
source activate isca_env

python control.py 1 1