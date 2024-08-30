#!/bin/bash
#SBATCH -o run.%j.out
#SBATCH --partition=GPUA800
#SBATCH -J cryo
#SBATCH --nodes=1            
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4           

cd ~/szs/szs1/voxmol

source activate vox_mol

python voxmol/cryo_training.py