#!/bin/bash

PROJECT_ROOT="/home/dylenthomas/Calliope/LARS/LiveASRonRPi-4"
OUT_NAME="test"

gcc -g -Wall onnxrt_vad_gpu_test.c \
	-I $PROJECT_ROOT/include \
    -I $PROJECT_ROOT/include/onnxruntime \
    -L $PROJECT_ROOT/libs \
	-L $PROJECT_ROOT/libs/onnxruntime \
	-lasound -lonnxruntime -lmic_access \
	-o $OUT_NAME 

echo "env vars:"
echo "export LD_LIBRARY_PATH="$PROJECT_ROOT/libs""
echo ""
echo "run command: ./$OUT_NAME ../../.model/silero_vad_16k_op15.onnx plughw:CARD=Generic_1,DEV=0"
