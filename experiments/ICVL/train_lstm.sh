#!/usr/bin/env bash

set -x
set -e

ROOT="/home/wuyiming/git/Hand"
TOOL=$ROOT/caffe/build/tools
LOG=$ROOT/log/ICVL
CAFFEMODEL=$ROOT/weights/ICVL
MODELS=$ROOT/models/ICVL/hand_lstm

export PYTHONPATH=$ROOT/caffe/python:$ROOT/lib/data/:$ROOT/lib/data_layer:$ROOT/lib/util:$PYTHONPATH

LOG_FILE="$LOG/train_log_lstm_`date`.txt"
exec &> >(tee -a "$LOG_FILE")
echo Logging to "$LOG_FILE"

ln -sf  "$LOG_FILE" "$LOG/train.log"

$TOOL/caffe train -solver $MODELS/solver_hand_lstm.prototxt \
		  -weights $CAFFEMODEL/hand_baseline/baseline_iter_200000.caffemodel \
		  -gpu 0
