#!/bin/bash

# Script to run the Python training process with specific arguments

echo "Running training process..."

# Navigate to the directory where main.py is located
cd Extensions/Linear_Systems_Regression || exit

# Default values for arguments that are not being directly set
patience=30
min_delta=0.0
epochs=1000
val_num_batches=10
batch_size=100
lr=0.001

# Arguments passed from the command line
matrix_type=$1
A_size=$2
fill_percentage=$3

# Run the Python script with the specified arguments
python3 main.py --patience "$patience" --min_delta "$min_delta" --epochs "$epochs" --val_num_batches "$val_num_batches" --batch_size "$batch_size" --lr "$lr" --A_size "$A_size" --matrix_type "$matrix_type" --fill_percentage "$fill_percentage"
