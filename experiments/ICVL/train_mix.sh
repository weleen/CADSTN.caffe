#!/usr/bin/env bash

set -x
set -e

ROOT="/home/wuyiming/Hand-dev"
TOOL=$ROOT/caffe/build/tools
LOG=$ROOT/log/ICVL
CAFFEMODEL=$ROOT/weights/ICVL
MODELS=$ROOT/models/ICVL/hand_mix

export PYTHONPATH=$ROOT/caffe/python:$ROOT/lib/data/:$ROOT/lib/layers:$ROOT/lib/util:$PYTHONPATH

LOG_FILE="$LOG/mix_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG_FILE")
echo Logging to "$LOG_FILE"

$TOOL/caffe train -solver $MODELS/solver_hand_mix.prototxt \
                  -weights $CAFFEMODEL/hand_3D_and_depth/hand_3D_and_depth_iter_100000.caffemodel,$CAFFEMODEL/hand_lstm/hand_lstm_iter_100000.caffemodel \
                  -gpu 1
