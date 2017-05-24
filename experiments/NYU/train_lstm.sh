#!/usr/bin/env bash

set -x
set -e

ROOT="/home/wuyiming/git/Hand"
TOOL=$ROOT/caffe/build/tools
LOG=$ROOT/log/NYU
CAFFEMODEL=$ROOT/weights/NYU
MODELS=$ROOT/models/NYU/hand_lstm

export PYTHONPATH=$ROOT/caffe/python:$ROOT/lib/data/:$ROOT/lib/layers:$ROOT/lib/util:$PYTHONPATH

LOG_FILE="$LOG/lstm_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG_FILE")
echo Logging to "$LOG_FILE"

$TOOL/caffe train -solver $MODELS/solver_hand_lstm.prototxt \
                  -weights $CAFFEMODEL/hand_baseline/hand_baseline_iter_100000.caffemodel \
		  -gpu 1
