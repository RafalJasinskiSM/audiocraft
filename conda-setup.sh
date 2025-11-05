#!/bin/bash

set -e  # Exit on any error

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check if .venv directory already exists
if [ -d ".venv" ]; then
    echo "Warning: .venv directory already exists. Overwriting..."
fi

# Create a conda environment with Python 3.9 in the .venv directory
echo "Creating conda environment..."
conda create --prefix=.venv python=3.9 -y || {
    echo "Error: Failed to create conda environment"
    exit 1
}

# Activate the conda environment
echo "Activating conda environment..."
conda init || {
    echo "Error: Failed to initialize conda"
    exit 1
}

conda activate .venv/ || {
    echo "Error: Failed to activate conda environment"
    exit 1
}

# Install ffmpeg from conda-forge
echo "Installing ffmpeg..."
conda install "ffmpeg<5" -c conda-forge -y || {
    echo "Error: Failed to install ffmpeg"
    exit 1
}

# Install torch and other dependencies using pip
echo "Installing torch..."
python -m pip install 'torch==2.1.0' || {
    echo "Error: Failed to install torch"
    exit 1
}

echo "Installing setuptools and wheel..."
python -m pip install setuptools wheel || {
    echo "Error: Failed to install setuptools and wheel"
    exit 1
}

echo "Installing package in development mode..."
python -m pip install -e '.[wm]' || {
    echo "Error: Failed to install package in development mode"
    exit 1
}

# Set environment variable for the conda environment
echo "Setting environment variable..."
conda env config vars set AUDIOCRAFT_TEAM=softwaremind || {
    echo "Error: Failed to set environment variable"
    exit 1
}

# Deactivate and reactivate the environment to apply changes
echo "Reactivating environment to apply changes..."
if ! conda deactivate; then
    echo "Warning: Failed to deactivate environment"
fi
conda activate .venv/ || {
    echo "Error: Failed to reactivate conda environment"
    exit 1
}

echo "Setup complete."

