#!/bin/bash

# Input arguments
BEGIN_INDEX=$1
END_INDEX=$2
IMG_INDEX=$3
GPU=$4

# Define base log directory
LOG_DIR="/home/ltnghia02/vischronos/logs"

# Base log file name
BASE_LOG_FILE="step234_${BEGIN_INDEX}_${END_INDEX}_${IMG_INDEX}.txt"
LOG_FILE="${LOG_DIR}/${BASE_LOG_FILE}"

# Check if the log file already exists and create a unique one if necessary
COUNTER=1
while [ -e "$LOG_FILE" ]; do
  LOG_FILE="${LOG_DIR}/step234_${BEGIN_INDEX}_${END_INDEX}_${IMG_INDEX} (${COUNTER}).txt"
  ((COUNTER++))
done

# Run the process
nohup python3 /home/ltnghia02/vischronos/scripts/second.py \
  /home/ltnghia02/dataset/ /home/ltnghia02/vischronos/prompts/4 \
  --begin_index "$BEGIN_INDEX" --end_index "$END_INDEX" --img_index "$IMG_INDEX" --gpu $GPU\
  > "$LOG_FILE" 2>&1 &

echo "Process started with log file: $LOG_FILE"


