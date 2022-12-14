#!/bin/bash

${CONDA} run -n op --no-capture-output \
    python t5_experiments/scripts/train_predict.py \
    --training-file data/moral_stories/critic_train_prep_augmented_final.json \
    --validation-file data/moral_stories/critic_dev_prep_final.json \
    --language-model allenai/unifiedqa-t5-base \
    --model-dir /scratch/mete/op/op_critic_moral_stories_augmented \
    --epochs 5 \
    --batch-size 32 \
    --val-batch-size 32 \
    --input-label critic_input \
    --output-label critic_output \
    --max-input-length 256 \
    --max-output-length 60 \
    --wandb-run-name critic_moral_stories_augmented \
    --save-results /scratch/mete/op/critic_moral_stories_augmented_outputs