#!/bin/bash

${CONDA} run -n op --no-capture-output \
    python t5_experiments/scripts/train_predict.py \
    --training-file data/social_chemistry/sc_train_final.json \
    --validation-file data/social_chemistry/sc_dev_final.json \
    --language-model allenai/unifiedqa-t5-base \
    --model-dir /scratch/mete/op/op_base_social_chemistry \
    --epochs 5 \
    --batch-size 32 \
    --val-batch-size 32 \
    --input-label model_input \
    --output-label model_output \
    --max-input-length 256 \
    --max-output-length 60 \
    --wandb-run-name base_social_chemistry \
    --save-results /scratch/mete/op/base_social_chemistry_outputs