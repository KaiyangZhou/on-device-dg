#!/bin/bash

DATA_ROOT=./data/

TRAINER=$1
AUG=$2
DATASET=$3  # {p_air, ...}
HPARAM=$4
ENV=$5  # {2k, full}

TIDX_DEFAULT=1
TIDX="${6:-$TIDX_DEFAULT}"

ENV=2k

TEACHER=resnet50

if [[ $DATASET == "pacs" ]]; then
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [[ $DATASET == "oh" ]]; then
    D1=art
    D2=clipart
    D3=product
    D4=real_world
elif [[ $DATASET == "digits_dg" ]]; then
    D1=mnist
    D2=mnist_m
    D3=svhn
    D4=syn
else
    D1=none
    D2=none
    D3=none
    D4=none
fi

if [[ $TIDX == 1 ]]; then
    S1=${D2}
    S2=${D3}
    S3=${D4}
    T=${D1}
elif [[ $TIDX == 2 ]]; then
    S1=${D1}
    S2=${D3}
    S3=${D4}
    T=${D2}
elif [[ $TIDX == 3 ]]; then
    S1=${D1}
    S2=${D2}
    S3=${D4}
    T=${D3}
elif [[ $TIDX == 4 ]]; then
    S1=${D1}
    S2=${D2}
    S3=${D3}
    T=${D4}
else
    echo "TIDX=$TIDX exceeds the range of 1-4"
    exit 0
fi

for SEED in 1 2 3
do
    if [[ $DATASET == "pacs" || $DATASET == "oh" || $DATASET == "digits_dg" ]]; then
        OUTPUT_DIR=output/${TRAINER}/aug_${AUG}/${DATASET}/env_${ENV}/${HPARAM}/${T}/seed${SEED}
        TEACHER_WEIGHTS=pretrained/Vanilla/${DATASET}/env_${ENV}/${TEACHER}/${T}/seed${SEED}/model/model-best.pth.tar
    else
        OUTPUT_DIR=output/${TRAINER}/aug_${AUG}/${DATASET}/env_${ENV}/${HPARAM}/seed${SEED}
        TEACHER_WEIGHTS=pretrained/Vanilla/${DATASET}/env_${ENV}/${TEACHER}/seed${SEED}/model/model-best.pth.tar
    fi
    
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Oops! The results already exist at ${OUTPUT_DIR} (so skip this job)"
    else
        if [[ $DATASET == "pacs" || $DATASET == "oh" || $DATASET == "digits_dg" ]]; then
            python train.py \
            --root ${DATA_ROOT} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/hparam/${HPARAM}.yaml \
            --source-domains ${S1} ${S2} ${S3} \
            --target-domains ${T} \
            --output-dir ${OUTPUT_DIR} \
            DATASET.ENV ${ENV} \
            TEACHER.BACKBONE.NAME ${TEACHER} \
            TEACHER.INIT_WEIGHTS ${TEACHER_WEIGHTS} \
            OKD_AUG_TYPE ${AUG}
        else
            python train.py \
            --root ${DATA_ROOT} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/hparam/${HPARAM}.yaml \
            --output-dir ${OUTPUT_DIR} \
            DATASET.ENV ${ENV} \
            TEACHER.BACKBONE.NAME ${TEACHER} \
            TEACHER.INIT_WEIGHTS ${TEACHER_WEIGHTS} \
            OKD_AUG_TYPE ${AUG}
        fi
    fi
done
