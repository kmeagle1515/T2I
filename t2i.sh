#!/bin/bash
#SBATCH --job-name=t2i
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --output=./t2i.out
#SBATCH --error=./t2i.err
#SBATCH --gres=gpu:2
eval "$(conda shell.bash hook)"
conda activate torch-env
python runtime.py
