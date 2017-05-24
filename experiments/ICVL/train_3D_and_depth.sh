#!/usr/bin/env bash

set -x
set -e

ROOT="/home/wuyiming/git/Hand"
TOOL=$ROOT/caffe/build/tools
LOG=$ROOT/log/ICVL
CAFFEMODEL=$ROOT/weights/ICVL
MODELS=$ROOT/models/ICVL/hand_3D_and_depth

export PYTHONPATH=$ROOT/caffe/python:$ROOT/lib/data/:$ROOT/lib/layers:$ROOT/lib/util:$PYTHONPATH

LOG_FILE="$LOG/3D_and_depth_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG_FILE")
echo Logging to "$LOG_FILE"

$TOOL/caffe train -solver $MODELS/solver_hand_3D_and_depth.prototxt \
                  -weights $CAFFEMODEL/hand_baseline/hand_baseline_iter_200000.caffemodel,$CAFFEMODEL/hand_baseline_3D_input/hand_baseline_3D_input_iter_200000.caffemodel \
                  -gpu 1
