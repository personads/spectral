#!/bin/bash

DATA_PATH=~/data/20news
#EXP_PATH=~/exp/spectral/20news/mbert
EXP_PATH=~/exp/spectral/20news/bert
#LM="bert-base-multilingual-cased"
LM="bert-base-cased"
NUM_FREQS=512
SEEDS=( 1932 2771 7308 8119 9095 )
NUM_EXP=0
NUM_ERR=0

train() {
  exp_dir=$1
  filter=$2
  seed=$3
  if [ -f "$exp_dir/best.pt" ]; then
    echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
  else
    echo "Training '${LM}' with ${filter} and random seed ${seed}."
    python classify.py \
      $DATA_PATH/csv/20news-train.csv \
      $DATA_PATH/csv/20news-dev.csv \
      --repeat_labels \
      "$LM" --embedding_caching \
      "$filter" \
      "linear" \
      $exp_dir \
      --random_seed $seed
    # check for error
    if [ $? -ne 0 ]; then
      echo "[Error] Could not complete training of previous model."
      (( NUM_ERR++ ))
    fi
    (( NUM_EXP++ ))
  fi
}

evaluate() {
  exp_dir=$1
  filter=$2
  eval_path="${exp_dir}/20news-test-pred-results.txt"
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
      --random_seed $seed \
      --prediction
    # check for error
    if [ $? -ne 0 ]; then
      echo "[Error] Could not complete evaluation of previous model."
      (( NUM_ERR++ ))
    fi
    (( NUM_EXP++ ))
  fi
  python tasks/eval/sentences.py $DATA_PATH/csv/20news-test.csv $exp_dir/20news-test-pred.csv | tee "$eval_path"
}

if [ ! -f $DATA_PATH/csv/20news-train.csv ]; then
  echo "Converting 20news to CSV format..."
  python tasks/20news/convert.py $DATA_PATH/raw $DATA_PATH/csv/20news -vp 0.2 -rs "${SEEDS[0]}"
else
  echo "Using existing 20news CSV data at '$DATA_PATH/csv/'."
fi

# iterate over seeds
for seed in "${SEEDS[@]}"; do
  # train without filter
  train "$EXP_PATH/nofilter-rs$seed" "nofilter()" $seed
  evaluate "$EXP_PATH/nofilter-rs$seed" "nofilter()" $seed

  # train fixed band filters (low)
  exp_dir=$EXP_PATH/band-0-1-rs$seed
  filter="band($NUM_FREQS, 0, 1)"
  train "$exp_dir" "$filter" $seed
  evaluate "$exp_dir" "$filter" $seed

  # train fixed band filters (medium low)
  exp_dir=$EXP_PATH/band-2-8-rs$seed
  filter="band($NUM_FREQS, 2, 8)"
  train "$exp_dir" "$filter" $seed
  evaluate "$exp_dir" "$filter" $seed

  # train fixed band filters (medium)
  exp_dir=$EXP_PATH/band-9-33-rs$seed
  filter="band($NUM_FREQS, 9, 33)"
  train "$exp_dir" "$filter" $seed
  evaluate "$exp_dir" "$filter" $seed

  # train fixed band filters (medium high)
  exp_dir=$EXP_PATH/band-34-129-rs$seed
  filter="band($NUM_FREQS, 34, 129)"
  train "$exp_dir" "$filter" $seed
  evaluate "$exp_dir" "$filter" $seed

  # train fixed band filters (high)
  exp_dir=$EXP_PATH/band-130-511-rs$seed
  filter="band($NUM_FREQS, 130, 511)"
  train "$exp_dir" "$filter" $seed
  evaluate "$exp_dir" "$filter" $seed

  # train with learned filter
  filter="auto($NUM_FREQS)"
  train "$EXP_PATH/auto-rs$seed" "$filter" $seed
  evaluate "$EXP_PATH/auto-rs$seed" "$filter" $seed
done

echo "Completed ${NUM_EXP} experiments with ${NUM_ERR} error(s)."