#!/bin/bash

DATA_PATH=~/data/20news
EXP_PATH=~/exp/spectral/20news
NUM_FREQS=512
NUM_BANDS=5
SEED=2771

if [ ! -f $DATA_PATH/csv/20news-train.csv ]; then
  echo "Converting 20news to CSV format..."
  python tasks/20news/convert.py $DATA_PATH/raw $DATA_PATH/csv/20news -vp 0.2 -rs $SEED
else
  echo "Using existing 20news CSV data at '$DATA_PATH/csv/'."
fi

for band_idx in $(seq 0 $((NUM_BANDS - 1))); do
    python classify.py \
      $DATA_PATH/csv/20news-train.csv \
      $DATA_PATH/csv/20news-dev.csv \
      --repeat_labels \
      "bert-base-cased" --embedding_caching \
      "eqalloc($NUM_FREQS, $NUM_BANDS, $band_idx)" \
      "linear" \
      $EXP_PATH/eqalloc$band_idx-max-rs$SEED \
      --random_seed $SEED
done

