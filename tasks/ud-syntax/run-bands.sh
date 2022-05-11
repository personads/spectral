#!/bin/bash

UD_PATH=~/data/ud29/treebanks
TB_NAME="UD_English-EWT"
TB_FILE="en-ewt"
DATA_PATH=~/data/ud29/csv
EXP_PATH=~/exp/spectral/ud
TASKS=( "pos" "relations" )
LM="bert-base-multilingual-cased"
NUM_FREQS=512
NUM_BANDS=5
SEED=2771
NUM_EXP=0
NUM_ERR=0

train() {
  exp_dir=$1
  filter=$2
  train_path=$3
  dev_path=$4
  if [ -f "$exp_dir/best.pt" ]; then
    echo "[Warning] Experiment '$exp_dir' already exists. Not retraining."
  else
    echo "Training '${LM}' with ${filter} and random seed ${SEED} ('${exp_dir}')."
    python classify.py \
      $train_path \
      $dev_path \
      --repeat_labels \
      "$LM" --embedding_caching \
      "$filter" \
      "linear" \
      $exp_dir \
      --random_seed $SEED
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
  pred_path="$exp_dir/$(basename $test_path)"
  pred_path="${pred_path%.*}-pred.csv"
  if [ -f $pred_path ]; then
    echo "[Warning] Predictions in '$exp_dir' already exist. Not re-predicting."
  else
    echo "Evaluating '${LM}' with ${filter} on $DATA_PATH/csv/test.csv."
    python classify.py \
      $train_path \
      $test_path \
      --repeat_labels \
      "$LM" --embedding_caching \
      "$filter" \
      "linear" \
      $exp_dir \
      --random_seed $SEED \
      --prediction
    # check for error
    if [ $? -ne 0 ]; then
      echo "[Error] Could not complete evaluation of previous model."
      (( NUM_ERR++ ))
    fi
    (( NUM_EXP++ ))
  fi
  python tasks/eval/tokens.py $test_path $pred_path -t "$LM"
}

#
# Main
#

for task in "${TASKS[@]}"; do
  # convert data
  if [ ! -f $DATA_PATH/$TB_FILE-train-$task.csv ]; then
    echo "Converting UD ($task) to CSV format..."
    python tasks/ud-syntax/convert.py $UD_PATH/$TB_NAME $task $DATA_PATH
  else
    echo "Using existing CSV data at '$DATA_PATH/$TB_FILE-(train|dev)-$task.csv'."
  fi

  # train without filter
  train "$EXP_PATH/$task/nofilter-rs$SEED" "nofilter()" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"
  evaluate "$EXP_PATH/$task/nofilter-rs$SEED" "nofilter()" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"

  # train with learned filter
  filter="auto($NUM_FREQS)"
  train "$EXP_PATH/$task/auto-rs$SEED" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"
  evaluate "$EXP_PATH/$task/auto-rs$SEED" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"

#  # train with equally allocated bands
#  for band_idx in $(seq 0 $((NUM_BANDS - 1))); do
#      exp_dir="$EXP_PATH/$task/eqalloc$band_idx-rs$SEED"
#      filter="eqalloc($NUM_FREQS, $NUM_BANDS, $band_idx)"
#      train "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"
#      evaluate "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"
#  done

  # train with band filter (low)
  exp_dir=$EXP_PATH/$task/band-0-1-rs$SEED
  filter="band($NUM_FREQS, 0, 1)"
  train "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"
  evaluate "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"

  # train with band filter (medium low)
  exp_dir=$EXP_PATH/$task/band-2-8-rs$SEED
  filter="band($NUM_FREQS, 2, 8)"
  train "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"
  evaluate "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"

  # train with band filter (medium)
  exp_dir=$EXP_PATH/$task/band-9-33-rs$SEED
  filter="band($NUM_FREQS, 9, 33)"
  train "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"
  evaluate "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"

  # train fixed band filters (medium high)
  exp_dir=$EXP_PATH/$task/band-34-129-rs$SEED
  filter="band($NUM_FREQS, 34, 129)"
  train "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"
  evaluate "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"

  # train fixed band filters (high)
  exp_dir=$EXP_PATH/$task/band-130-511-rs$SEED
  filter="band($NUM_FREQS, 130, 511)"
  train "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"
  evaluate "$exp_dir" "$filter" "$DATA_PATH/$TB_FILE-train-$task.csv" "$DATA_PATH/$TB_FILE-dev-$task.csv"

done

echo "Completed ${NUM_EXP} experiments with ${NUM_ERR} error(s)."