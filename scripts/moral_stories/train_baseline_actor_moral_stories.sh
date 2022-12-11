#!/bin/bash

SITUATION_TOKEN="<|SIT|>"
INTENTION_TOKEN="<|INT|>"
MORAL_ACTION_TOKEN="<|M_ACT|>"
IMMORAL_ACTION_TOKEN="<|I_ACT|>"
NORM_TOKEN="<|NRM|>"

${CONDA} run -n op --no-capture-output \
    python t5_experiments/scripts/train_predict.py \
    --training-file data/moral_stories/norm-actions+context+consequences/norm_distance/actor_train.json \
    --validation-file data/moral_stories/norm-actions+context+consequences/norm_distance/actor_dev.json \
    --language-model allenai/unifiedqa-t5-base \
    --model-dir /scratch/mete/op/op_baseline_actor_moral_stories \
    --epochs 5 \
    --batch-size 16 \
    --val-batch-size 16 \
    --input-label actor_input \
    --output-label actor_output \
    --max-input-length 256 \
    --max-output-length 60 \
    --wandb-run-name baseline_actor_moral_stories \
    --special-tokens "${SITUATION_TOKEN}" "${INTENTION_TOKEN}" "${MORAL_ACTION_TOKEN}" "${IMMORAL_ACTION_TOKEN}" "${NORM_TOKEN}"