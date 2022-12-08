#!/bin/bash

${CONDA} run -n op --no-capture-output \
    python t5_experiments/scripts/train_predict.py \
    --validation-file ${1} \
    --model-dir /scratch/mete/op_baseline_actor_moral_stories \
    --val-batch-size 32 \
    --input-label actor_input \
    --output-label actor_output \
    --save-results outputs