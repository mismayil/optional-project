#!/bin/bash

${CONDA} run -n op --no-capture-output \
    python t5_experiments/scripts/train_predict.py \
    --validation-file data/glucose_cross_dim_val_data.json \
    --model-dir /scratch/mete/op_baseline2_actor_glucose_cross_dim \
    --val-batch-size 32 \
    --input-label model_input \
    --output-label model_output \