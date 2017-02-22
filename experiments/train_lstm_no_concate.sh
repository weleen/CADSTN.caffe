#!/usr/bin/env bash

set -x
set -e

ROOT="/home/wuyiming/git/Hand"
TOOL=$ROOT/caffe/build/tools
LOG=$ROOT/log
CAFFEMODEL=$ROOT/weights
MODELS=$ROOT/models

export PYTHONPATH=$ROOT/lib/data/:$ROOT/lib/data_layer:$ROOT/lib/util:$PYTHONPATH

LOG_FILE="$LOG/train_log_lstm_no_concate_`date`.txt"
exec &> >(tee -a "$LOG_FILE")
echo Logging to "$LOG_FILE"
rm "$LOG/train.log"
ln -s  "$LOG_FILE" "$LOG/train.log"

$TOOL/caffe train -solver $MODELS/hand_lstm/solver_hand_lstm_no_concate.prototxt \
		  -weights $CAFFEMODEL/hand_baseline/baseline_iter_200000.caffemodel \
		  -gpu 0
