#!/usr/bin/env bash

set -x
set -e

ROOT="/home/wuyiming/git/Hand"
TOOL=$ROOT/caffe/build/tools
LOG=$ROOT/log/NYU
CAFFEMODEL=$ROOT/weights/NYU
MODELS=$ROOT/models/NYU/hand_bidirectional_lstm_small_frame_size_no_concate

export PYTHONPATH=$ROOT/caffe/python:$ROOT/lib/data/:$ROOT/lib/data_layer:$ROOT/lib/util:$PYTHONPATH

LOG_FILE="$LOG/bidirectional_lstm_small_frame_size_no_concate`date`.txt"
exec &> >(tee -a "$LOG_FILE")
echo Logging to "$LOG_FILE"

$TOOL/caffe train -solver $MODELS/solver_hand_bidirectional_lstm_small_frame_size_no_concate.prototxt \
		  -weights $CAFFEMODEL/hand_baseline/hand_baseline_iter_150000.caffemodel \
		  -gpu 1
