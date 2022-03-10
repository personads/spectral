#!/bin/bash

DATA_PATH=~/Documents/datasets/ptb-sancl
EXP_PATH=~/Documents/experiments/debug/spectral
NUM_FREQS=512
NUM_BANDS=5
SEED=2771

python tasks/ptb/convert.py $DATA_PATH/raw $DATA_PATH/csv

for band_idx in $(seq 0 $((NUM_BANDS - 1))); do
  python classify.py \
    $DATA_PATH/csv/train.csv \
    $DATA_PATH/csv/dev.csv \
    "bert-base-cased" --embedding_caching \
    "eqalloc($NUM_FREQS, $NUM_BANDS, $band_idx)" \
    "linear" \
    $EXP_PATH \
    --random_seed $SEED
done