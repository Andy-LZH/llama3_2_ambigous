#!/bin/bash

CHECKPOINT_DIR=/scratch/alpine/zhli3162/.cache/llama_stack/checkpoints/Llama3.2-11B-Vision

NGPUS=2
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun \
  --nproc_per_node=$NGPUS \
  llama_models/scripts/example_chat_completion.py $CHECKPOINT_DIR \
  --model_parallel_size $NGPUS