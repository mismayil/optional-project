#!/bin/bash

PYTHONPATH=$(pwd) python t5_experiments/scripts/train_predict.py \
--validation-file ${1} \
--model-dir /scratch/mete/op/op_critic_moral_stories_balanced \
--val-batch-size 32 \
--input-label critic_input \
--output-label human_feedback \
--max-input-length 256 \
--max-output-length 60 \
--save-results outputs \
--prediction-label critic_feedback

