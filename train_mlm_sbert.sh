#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ./envs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib
export CUDA_VISIBLE_DEVICES=0
# accelerate launch --multi_gpu --mixed_precision=fp16  --num_processes=2  mlm_sbert.py
accelerate launch  --mixed_precision=fp16  --num_processes=1  mlm_sbert.py
# time deepspeed --no_local_rank  mlm_sbert.py
