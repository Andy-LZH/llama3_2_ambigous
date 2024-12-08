#!/bin/bash

# run full scope grounding LMM
CHECKPOINT_DIR_FULLScope=/scratch/alpine/zhli3162/GLaMM-FullScope
python model/groundingLMM/app.py --version $CHECKPOINT_DIR_FULLScope

