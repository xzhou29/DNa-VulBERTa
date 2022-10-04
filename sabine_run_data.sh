#!/bin/bash
#SBATCH -J data_generation
#SBATCH -o log_data_generation.o%j
#SBATCH -t 24:00:00
#SBATCH -N 1 --ntasks-per-node=28
#SBATCH --mem=64GB
#module load torchvision/0.12.0-foss-2021b-CUDA-11.4.1
#module load PyTorch/1.11.0-foss-2021b-CUDA-11.4.1
conda activate bert
python src/data/de_naturalize.py
./data_generation.sh