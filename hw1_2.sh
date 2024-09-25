#!/bin/bash

IMAGE_DIR="$1"
SAVE_DIR="$2"
MODEL_PATH="./best_model_hw1_2.pth"


python3 hw1_2_inference.py --image_dir "$IMAGE_DIR" --save_dir "$SAVE_DIR" --model_path "$MODEL_PATH"
