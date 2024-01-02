#!/bin/bash

set -x  # Enable debugging

# Define the GitHub repository URL
REPO_URL="https://github.com/Holindauer/Linear_Systems_Regression.git"

# Define the absolute or corrected relative path
LOCAL_PATH="$HOME/Projects/Go-LinAlg/Extensions/Linear_Systems_Regression"

# Redirecting standard output to /dev/null
echo "Checking if the repository is already downloaded..." > /dev/null

# Check if the repository exists in the local directory
if [ ! -d "$LOCAL_PATH" ]; then
    echo "Repository not found. Cloning..." > /dev/null
    # Clone the repository and check for errors
    if git clone "$REPO_URL" "$LOCAL_PATH" > /dev/null; then
        echo "Repository successfully cloned." > /dev/null
    else
        echo "Failed to clone repository." >&2  # Print to stderr
        exit 1
    fi
else
    echo "Repository already exists in the local directory." > /dev/null
fi
