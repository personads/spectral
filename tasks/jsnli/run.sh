#!/bin/bash

DATA_PATH=~/data/jsnli
EXP_PATH=~/exp/spectral/jsnli
LM="bert-base-multilingual-cased"
LM_SHORT="mbert"
NUM_FREQS=512
SEEDS=( 1932 2771 7308 8119 9095 )
NUM_EXP=0
NUM_ERR=0

train() {
  exp_dir=$1
  filter=$2
  train_path=$3
  dev_path=$4
  seed=$5
  if [ -f "$exp_dir/best.pt" ]; then
    echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
  else
    echo "Training '${LM}' with ${filter} and random seed ${seed} ('${exp_dir}')."
    python classify.py \
      $train_path \
      $dev_path \
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
  train_path=$3
  test_path=$4
  seed=$5
  pred_path="$exp_dir/$(basename $test_path)"
  pred_path="${pred_path%.*}-pred.csv"
  eval_path="${pred_path%.*}-results.txt"
  if [ -f $pred_path ]; then
    echo "[Warning] Predictions in '$exp_dir' already exist. Not re-predicting."
  else
    echo "Evaluating '${LM}' with ${filter} on ${test_path}."
    python classify.py \
      $train_path \
      $test_path \
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
  python tasks/eval/sentences.py $test_path $pred_path | tee "$eval_path"
}

#
# Main
#

# iterate over seeds
for seed in "${SEEDS[@]}"; do
  # convert data
  if [ ! -f $DATA_PATH/csv/train.csv ]; then
    echo "Converting JSNLI to CSV format..."
    python tasks/jsnli/convert.py $DATA_PATH/raw $DATA_PATH/csv
  else
    echo "Using existing CSV data at '$DATA_PATH/csv/(train|dev).csv'."
  fi

  # setup experiment root for TB
  exp_root="$EXP_PATH/$LM_SHORT"
  if [ ! -f $exp_root ]; then
    mkdir -p "$exp_root"
  fi

  # train with learned filter
  filter="auto($NUM_FREQS)"
  exp_dir="$exp_root/auto-rs$seed"
  train "$exp_dir" "$filter" "$DATA_PATH/csv/train.csv" "$DATA_PATH/csv/dev.csv" $seed
  evaluate "$exp_dir" "$filter" "$DATA_PATH/csv/train.csv" "$DATA_PATH/csv/dev.csv" $seed

  # train without filter
  exp_dir="$exp_root/nofilter-rs$seed"
  train "$exp_dir" "nofilter()" "$DATA_PATH/csv/train.csv" "$DATA_PATH/csv/dev.csv" $seed
  evaluate "$exp_dir" "nofilter()" "$DATA_PATH/csv/train.csv" "$DATA_PATH/csv/dev.csv" $seed
done

echo "Completed ${NUM_EXP} experiments with ${NUM_ERR} error(s)."