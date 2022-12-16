#!/bin/bash

dataset_type=${1:-dev}
suffix="_aligned"

PYTHONPATH=$(pwd) \
    python scripts/moral_stories/preprocess_moral_stories_for_critic.py \
    --datapath data/moral_stories/norm-actions+context+consequences/norm_distance/${dataset_type}.jsonl \
    --anti-ms-datapath data/contrastive_moral_stories/anti-ms/${dataset_type}.jsonl \
    --anti-ms-splits-datapath data/contrastive_moral_stories/anti-ms-with-prefix/${dataset_type}.jsonl \
    --actor-datapath outputs/actor_${dataset_type}_results.json \
    --suffix ${suffix}

PYTHONPATH=$(pwd) \
    python scripts/moral_stories/prepare_moral_stories_for_critic.py \
    --datapath data/moral_stories/norm-actions+context+consequences/norm_distance/critic_${dataset_type}_prep${suffix}.json