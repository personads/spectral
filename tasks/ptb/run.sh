#!/bin/bash

DATA_PATH=~/data/ptb-sancl
#EXP_PATH=~/exp/spectral/ptb
EXP_PATH=~/exp/spectral/ptb/mbert
LM="bert-base-multilingual-cased"
#LM="bert-base-cased"
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
      $DATA_PATH/csv/train.csv \
      $DATA_PATH/csv/dev.csv \
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
  if [ -f "$exp_dir/test-pred.csv" ]; then
    echo "[Warning] Predictions in '$exp_dir' already exist. Not re-predicting."
  else
    echo "Evaluating '${LM}' with ${filter} on $DATA_PATH/csv/test.csv."
    python classify.py \
      $DATA_PATH/csv/train.csv \
      $DATA_PATH/csv/test.csv \
      --repeat_labels \
      "$LM" --embedding_caching \
      "$filter" \
      "linear" \
      $exp_dir \
      --random_seed $SEED \
      --prediction
  fi
  python tasks/ptb/evaluate.py $DATA_PATH/csv/test.csv $exp_dir/test-pred.csv -t "$LM"
}

if [ ! -f $DATA_PATH/csv/train.csv ]; then
  echo "Converting PTB to CSV format..."
  python tasks/ptb/convert.py $DATA_PATH/raw $DATA_PATH/csv
else
  echo "Using existing PTB CSV data at '$DATA_PATH/csv/'."
fi

# train without filter
train "$EXP_PATH/nofilter-rs$SEED" "nofilter()"
evaluate "$EXP_PATH/nofilter-rs$SEED" "nofilter()"

## train equal allocation filters
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