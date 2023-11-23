#!/bin/bash

set -x  # Enable debugging

# Define the GitHub repository URL
REPO_URL="https://github.com/Holindauer/Linear-Systems-Regression.git"

# Define the absolute or corrected relative path
LOCAL_PATH="$HOME/Projects/Go-LinAlg/Extensions/Linear-Systems-Regression"

echo "Checking if the repository is already downloaded..."

# Check if the repository exists in the local directory
if [ ! -d "$LOCAL_PATH" ]; then
    echo "Repository not found. Cloning..."
    # Clone the repository and check for errors
    if git clone "$REPO_URL" "$LOCAL_PATH"; then
        echo "Repository successfully cloned."
    else
        echo "Failed to clone repository."
        exit 1
    fi
else
    echo "Repository already exists in the local directory."
fi
