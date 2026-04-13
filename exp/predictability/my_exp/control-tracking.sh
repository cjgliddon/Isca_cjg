#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mem=64G
#SBATCH -o output/control_tracking.out
#SBATCH -e output/control_tracking.err

module load miniforge
source activate isca_env

EXPNAME=$1
EXPROOT=/home/cgliddon/orcd/scratch/isca/data/${EXPNAME}        # change this to your data directory

# set indices of first and last chunk in the control run
CH_I=13
CH_F=49

# merge tracks
POSTPROC_DIR=/home/cgliddon/Isca/postprocessing
bash ${POSTPROC_DIR}/create_merged_files_base.sh $EXPROOT $CH_I $CH_F > output/merge_basefiles.out

# perform tracking
# first, calculate the number of chunks needed for the integration
CHUNK_LEN=80
N_STEPS=$(( ($CH_F - $CH_I + 1) * 30 * 4 ))
N_CHUNKS=$(( ($N_STEPS + 79) / $CHUNK_LEN ))
echo "Number of tracking chunks: ${N_CHUNKS}"

bash ${POSTPROC_DIR}/agcm_tracking_tools/make_tracks_base.sh $EXPROOT $CHUNK_LEN $N_CHUNKS > output/base_tracking.out