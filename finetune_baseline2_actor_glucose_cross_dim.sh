#!/bin/bash

${CONDA} run -n op --no-capture-output \
    python t5_experiments/scripts/train_predict.py \
    --training-file data/glucose_cross_dim_train_data.json \
    --validation-file data/glucose_cross_dim_val_data.json \
    --language-model /scratch/mete/op_baseline2_actor_pretrained_glucose_cross_dim \
    --tokenizer allenai/unifiedqa-t5-base \
    --model-dir /scratch/mete/op_baseline2_actor_glucose_cross_dim \
    --epochs 5 \
    --batch-size 16 \
    --val-batch-size 16 \
    --input-label model_input \
    --output-label model_output \
    --max-input-length 256 \
    --max-output-length 60 \
    --wandb-run-name baseline2_actor_glucose_cross_dim