#!/bin/bash
#SBATCH --job-name=t2i_infer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --output=./t2i_infer.out
#SBATCH --error=./t2i_infer.err
#SBATCH --gres=gpu:1
python runtime-infer.py
