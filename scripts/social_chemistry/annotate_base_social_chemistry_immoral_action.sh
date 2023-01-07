#!/bin/bash

${CONDA} run -n op --no-capture-output \
    python t5_experiments/scripts/train_predict.py \
    --validation-file ${1} \
    --model-dir /scratch/mete/op/op_critic_social_chemistry \
    --val-batch-size 64 \
    --input-label immoral_action \
    --output-label norm \
    --max-input-length 256 \
    --max-output-length 60 \
    --save-results data/moral_stories/norm-actions+context+consequences/norm_distance \
    --prediction-label immoral_social_norm

