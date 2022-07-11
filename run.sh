#!/bin/bash

SCRIPT_DIR=`dirname "$0"`
hostname=`hostname`

MITSUBA_BIN_DIR="$SCRIPT_DIR/../mitsuba-build/binaries/"

echo MITSUBA_BIN_DIR=$MITSUBA_BIN_DIR


export PATH=$MITSUBA_BIN_DIR:$MITSUBA_BIN_DIR/plugins:$PATH
export PYTHONPATH=$SCRIPT_DIR:$MITSUBA_BIN_DIR/python:$MITSUBA_BIN_DIR:$MITSUBA_BIN_DIR/../deplibs
export LD_LIBRARY_PATH=$PYTHONPATH
python3.7 "$@"
exit  $?
