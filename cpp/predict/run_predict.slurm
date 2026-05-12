#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=gdalnn_cpp
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err
cd $SLURM_SUBMIT_DIR
export GDALNN_THREADS=${SLURM_CPUS_PER_TASK:-16}
/usr/bin/time -v /projects/bgtj/isaaccorley/envs/torchgeo/bin/python \
    predict_cpp.py samples/1717_image.tif samples/1717_cpp_probs.tif
