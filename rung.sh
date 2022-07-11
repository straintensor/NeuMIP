#!/bin/bash

SCRIPT_DIR=`dirname "$0"`

export CUDA_VISIBLE_DEVICES=$1

shift
$SCRIPT_DIR/run.sh "$@"
exit  $?
