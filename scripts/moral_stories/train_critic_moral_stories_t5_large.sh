#!/bin/bash

${CONDA} run -n op --no-capture-output \
    python t5_experiments/scripts/train_predict.py \
    --training-file data/moral_stories/norm-actions+context+consequences/norm_distance/critic_train_prep_vp_extended_with_norm_np_final.json \
    --validation-file data/moral_stories/norm-actions+context+consequences/norm_distance/critic_dev_prep_vp_extended_with_norm_np_final.json \
    --language-model allenai/unifiedqa-t5-large \
    --model-dir /scratch/mete/op/op_critic_moral_stories_norm_only_large \
    --epochs 10 \
    --batch-size 16 \
    --val-batch-size 16 \
    --input-label critic_input \
    --output-label critic_output \
    --max-input-length 256 \
    --max-output-length 60 \
    --wandb-run-name critic_moral_stories_norm_only_large \
    --save-results /scratch/mete/op/critic_moral_stories_norm_only_large_outputs