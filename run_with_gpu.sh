#!/bin/bash

echo "Running MoBA with GPU..."
echo

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python not found!"
    echo "Please install Python or ensure it's in your PATH."
    exit 1
fi

# Check if PyTorch with CUDA is installed
if ! python -c "import torch; print('CUDA Available:', torch.cuda.is_available())" &> /dev/null; then
    echo "Error checking PyTorch! Make sure PyTorch is installed."
    echo "You can install it with: pip install torch"
    exit 1
fi

# Run the quick demo script
echo
python quickrun.py "$@"

echo
echo "Completed execution." 