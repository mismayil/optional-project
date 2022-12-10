#!/bin/bash

${CONDA} run -n op --no-capture-output \
    python t5_experiments/scripts/train_predict.py \
    --training-file data/critic_glucose_cross_dim_train_data_with_neg.json \
    --validation-file data/critic_glucose_cross_dim_val_data_with_neg.json \
    --language-model allenai/unifiedqa-t5-base \
    --model-dir /scratch/mete/op_critic_glucose_cross_dim \
    --epochs 5 \
    --batch-size 16 \
    --val-batch-size 16 \
    --input-label critic_input \
    --output-label critic_output \
    --max-input-length 256 \
    --max-output-length 60 \
    --wandb-run-name critic_glucose_cross_dim