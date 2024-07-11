#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./envs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib
# export CUDA_VISIBLE_DEVICES=0

python canon_trainer.py
