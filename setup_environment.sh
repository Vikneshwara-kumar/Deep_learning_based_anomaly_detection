#!/bin/bash

# Create a virtual environment with Python 3.12
python3.12 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the required libraries from requirements.txt
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate

echo "Virtual environment setup complete. Use 'source venv/bin/activate' to activate."
