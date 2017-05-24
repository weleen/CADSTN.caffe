#!/usr/bin/env bash

set -x
set -e

ROOT="/home/wuyiming/git/Hand"
TOOL=$ROOT/caffe/build/tools
LOG=$ROOT/log/NYU
CAFFEMODEL=$ROOT/weights/NYU
MODELS=$ROOT/models/NYU/hand_3D

export PYTHONPATH=$ROOT/caffe/python:$ROOT/lib/data/:$ROOT/lib/layers:$ROOT/lib/util:$PYTHONPATH

LOG_FILE="$LOG/3D_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG_FILE")
echo Logging to "$LOG_FILE"

$TOOL/caffe train -solver $MODELS/solver_hand_3D.prototxt \
                  -gpu 1
