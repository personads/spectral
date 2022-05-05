#!/bin/bash

DATA_PATH=~/data/20news
#EXP_PATH=~/exp/spectral/20news
#LM="bert-base-cased"
EXP_PATH=~/exp/spectral/20news/mbert
LM="bert-base-multilingual-cased"
NUM_FREQS=512
NUM_BANDS=5
SEED=2771

train() {
  exp_dir=$1
  filter=$2
  if [ -f "$exp_dir/best.pt" ]; then
    echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
  else
    echo "Training '${LM}' with ${filter} and random seed ${SEED}."
    python classify.py \
      $DATA_PATH/csv/20news-train.csv \
      $DATA_PATH/csv/20news-dev.csv \
      --repeat_labels \
      "$LM" --embedding_caching \
      "$filter" \
      "linear" \
      $exp_dir \
      --random_seed $SEED
  fi
}

evaluate() {
  exp_dir=$1
  filter=$2
  if [ -f "$exp_dir/20news-test-pred.csv" ]; then
    echo "[Warning] Predictions in '$exp_dir' already exist. Not re-predicting."
  else
    echo "Evaluating '${LM}' with ${filter} on $DATA_PATH/csv/20news-test.csv."
    python classify.py \
      $DATA_PATH/csv/20news-train.csv \
      $DATA_PATH/csv/20news-test.csv \
      --repeat_labels \
      "$LM" --embedding_caching \
      "$filter" \
      "linear" \
      $exp_dir \
      --random_seed $SEED \
      --prediction
  fi
  python tasks/20news/evaluate.py $DATA_PATH/csv/20news-test.csv $exp_dir/20news-test-pred.csv
}

if [ ! -f $DATA_PATH/csv/20news-train.csv ]; then
  echo "Converting 20news to CSV format..."
  python tasks/20news/convert.py $DATA_PATH/raw $DATA_PATH/csv/20news -vp 0.2 -rs $SEED
else
  echo "Using existing 20news CSV data at '$DATA_PATH/csv/'."
fi

# experiments without filter
train "$EXP_PATH/nofilter-rs$SEED" "nofilter()"
evaluate "$EXP_PATH/nofilter-rs$SEED" "nofilter()"

## train with eqalloc filters
#for band_idx in $(seq 0 $((NUM_BANDS - 1))); do
#    exp_dir=$EXP_PATH/eqalloc$band_idx-rs$SEED
#    filter="eqalloc($NUM_FREQS, $NUM_BANDS, $band_idx)"
#    train "$exp_dir" "$filter"
#    evaluate "$exp_dir" "$filter"
#done

# train fixed band filters (low)
exp_dir=$EXP_PATH/band-0-1-rs$SEED
filter="band($NUM_FREQS, 0, 1)"
train "$exp_dir" "$filter"
evaluate "$exp_dir" "$filter"

# train fixed band filters (medium low)
exp_dir=$EXP_PATH/band-2-8-rs$SEED
filter="band($NUM_FREQS, 2, 8)"
train "$exp_dir" "$filter"
evaluate "$exp_dir" "$filter"

# train fixed band filters (medium)
exp_dir=$EXP_PATH/band-9-33-rs$SEED
filter="band($NUM_FREQS, 9, 33)"
train "$exp_dir" "$filter"
evaluate "$exp_dir" "$filter"

# train fixed band filters (medium high)
exp_dir=$EXP_PATH/band-34-129-rs$SEED
filter="band($NUM_FREQS, 34, 129)"
train "$exp_dir" "$filter"
evaluate "$exp_dir" "$filter"

# train fixed band filters (high)
exp_dir=$EXP_PATH/band-130-511-rs$SEED
filter="band($NUM_FREQS, 130, 511)"
train "$exp_dir" "$filter"
evaluate "$exp_dir" "$filter"

# train with learned filter
filter="auto($NUM_FREQS)"
train "$EXP_PATH/auto-rs$SEED" "$filter"
evaluate "$EXP_PATH/auto-rs$SEED" "$filter"
