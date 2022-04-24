#!/bin/bash
#SBATCH --job-name=t2i_less
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --output=./t2i_less.out
#SBATCH --error=./t2i_less.err
#SBATCH --gres=gpu:2
eval "$(conda shell.bash hook)"
conda activate torch-env
python runtime.py
