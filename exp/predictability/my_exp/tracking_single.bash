#!/bin/bash

#SBATCH -p mit_normal
#SBATCH -n 4
#SBATCH -o output/specpert_tracking_%j.out
#SBATCH -e output/specpert_tracking_%j.err

module load miniforge
source activate isca_env

CLIMO=$1
INDIR=$2
LABEL=$3
NUM_CHUNKS=$4
POSTPROC_DIR=/home/cgliddon/Isca/postprocessing       # contains useful postprocessing scripts

# INDIR=/home/cgliddon/orcd/scratch/isca/data/frierson_default/ensembles/${TSTART}.${DEC}/spec_mag_0.02/b${IMEM}
# cd ${INDIR}
# rm -r TRACK MSLP vor850 pert_anoms.nc pert_ps_vor.nc
cd ${POSTPROC_DIR}
bash create_merged_files_pert.sh ${INDIR} ${CLIMO}
bash agcm_tracking_tools/make_tracks_pert.sh ${INDIR} ${LABEL} ${NUM_CHUNKS}
echo "*************"
echo "Tracking complete for directory: "
echo ${INDIR}
echo "*************"