#!/usr/bin/env bash

set -x
set -e

ROOT="/home/wuyiming/git/Hand"
TOOL=$ROOT/caffe/build/tools
LOG=$ROOT/log/NYU
CAFFEMODEL=$ROOT/weights/NYU
MODELS=$ROOT/models/NYU/hand_baseline_layerwise_concate_features

export PYTHONPATH=$ROOT/caffe/python:$ROOT/lib/data/:$ROOT/lib/data_layer:$ROOT/lib/util:$PYTHONPATH

LOG_FILE="$LOG/baseline_layerwise_concate_features_`date`.txt"
exec &> >(tee -a "$LOG_FILE")
echo Logging to "$LOG_FILE"

$TOOL/caffe train -solver $MODELS/solver_hand_baseline_layerwise_concate_features.prototxt \
                  -snapshot $CAFFEMODEL/hand_baseline_layerwise_concate_features/hand_baseline_layerwise_concate_features_iter_20000.solverstate \
		  -gpu 1