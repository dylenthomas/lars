#!/bin/bash

PROJECT_ROOT="/home/dylenthomas/LiveASRonRPi-4"
OUT_NAME="test"

gcc -Wall mic_test.c \
	-I $PROJECT_ROOT/include \
	-L $PROJECT_ROOT/libs \
	-lasound -lmic_access \
	-o $OUT_NAME 

echo "env vars:"
echo "export LD_LIBRARY_PATH="$PROJECT_ROOT/libs""
echo ""
echo "run command: ./$OUT_NAME plughw:CARD=Snowball,DEV=0"
