#!/bin/bash

cd ../../

# custom config
DATA=./data/
TRAINER=CoCoOp
DATASET=$1 # {p_air, ...}
CFG=$2  # {rn50_ctxv1, ...}
ENV=$3  # {2k, full}

for SEED in 1 2 3
do
  python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --seed ${SEED} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/prompts/${TRAINER}/${CFG}.yaml \
    --output-dir output/${DATASET}/${TRAINER}/${CFG}/${ENV}/seed${SEED} \
    DATASET.ENV ${ENV}
done
