#!/bin/bash

# Assign arguments to variables
mode="test"
way="C"
val_csv="$1"
val_dir="$2"
save_path="./best_finetuned_model.pth"
output_path="$3"

# Run the Python script to test the fine-tuned model on the validation data
python3 test_setting_c.py --mode "$mode" --setting "$way" --val_csv "$val_csv" --val_dir "$val_dir" --save_path "$save_path" --output_csv "$output_path"
