#!/bin/bash
#SBATCH --job-name=t2i_infer_score
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --output=./t2i_infer_score.out
#SBATCH --error=./t2i_infer_score.err
#SBATCH --gres=gpu:1
python runtime-infer-score.py
