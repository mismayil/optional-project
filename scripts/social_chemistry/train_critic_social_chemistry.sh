#!/bin/bash

${CONDA} run -n op --no-capture-output \
    python t5_experiments/scripts/train_predict.py \
    --training-file data/social_chemistry/sc_critic_train_final.json \
    --validation-file data/social_chemistry/sc_critic_dev_final.json \
    --language-model allenai/unifiedqa-t5-base \
    --model-dir /scratch/mete/op/op_critic_social_chemistry \
    --epochs 5 \
    --batch-size 32 \
    --val-batch-size 32 \
    --input-label critic_input \
    --output-label critic_output \
    --max-input-length 256 \
    --max-output-length 60 \
    --wandb-run-name critic_social_chemistry \
    --save-results /scratch/mete/op/critic_social_chemistry_outputs