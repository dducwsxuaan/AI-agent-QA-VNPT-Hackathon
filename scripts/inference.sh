#!/bin/bash
set -e


CMD="python predict.py"

if [ -n "$INPUT_PATH" ]; then
	CMD="$CMD --input $INPUT_PATH"
fi

if [ -n "$OUTPUT_PATH" ]; then
	CMD="$CMD --output $OUTPUT_PATH"
fi

echo "Running: $CMD"
exec $CMD