#!/bin/bash

# Directory containing config files
CONFIG_DIR="configs/"

# Loop through all .yaml or .yml files in the config directory
for config_file in $(find $CONFIG_DIR -name "*.yaml" -o -name "*.yml"); do
    echo "Running with config file: $config_file"
    CUDA_VISIBLE_DEVICES=2 python run_vidtome.py --config "$config_file"
    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "Successfully executed with $config_file"
    else
        echo "Failed to execute with $config_file"
    fi
done