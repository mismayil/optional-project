#!/bin/bash

${CONDA} run -n op --no-capture-output \
    python t5_experiments/scripts/train_predict.py \
    --validation-file data/glucose_cross_dim_val_data.json \
    --tokenizer allenai/unifiedqa-t5-base \
    --model-dir /scratch/mete/op_baseline2_actor_glucose_cross_dim \
    --epochs 10 \
    --batch-size 32 \
    --val-batch-size 32 \
    --input-label model_input \
    --output-label model_output \
    --wandb-run-name baseline2_actor_glucose_cross_dim