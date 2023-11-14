#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Notify the user that setup is complete
echo "Setup complete. Virtual environment activated."
