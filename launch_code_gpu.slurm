#!/bin/bash
#SBATCH --job-name=benchmark_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00


source ~/.bashrc
conda activate transformers_env
nvidia-smi
python xxxxxx.py