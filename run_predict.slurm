#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=gdal_unet
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err
cd $SLURM_SUBMIT_DIR
time /projects/bgtj/isaaccorley/envs/torchgeo/bin/python predict_gdal.py samples/1717_image.tif samples/1717_gdal_probs.tif
