#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=vrt_unet
#SBATCH --output=vrt_%j.out
#SBATCH --error=vrt_%j.err
cd $SLURM_SUBMIT_DIR
export PYTHONUNBUFFERED=1
# Run gdal-VRT pipeline; redirect everything to a combined log we can read.
# /usr/bin/time -v writes its report to stderr after the wrapped command exits.
( /usr/bin/time -v /projects/bgtj/isaaccorley/envs/torchgeo/bin/python -u build_vrt.py \
        samples/1717_image.tif 1717_gdal_probs.tif ) > vrt_${SLURM_JOB_ID}.log 2>&1
echo "exit: $?"
echo "--- tail of log ---"
tail -40 vrt_${SLURM_JOB_ID}.log
