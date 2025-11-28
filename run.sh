#!/bin/bash
# CleanDictate Runner Script
cd /home/sharadhnaidu/Desktop/CleanDictate
source venv/bin/activate
export LD_LIBRARY_PATH="$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"

# Suppress ALSA warnings (non-critical audio library messages)
export ALSA_CARD=0
export SDL_AUDIODRIVER=pulse

echo "Starting CleanDictate V2.0..."
echo "Please wait while models load (this may take 30-60 seconds)..."
echo ""

python cleandictate.py 2>&1

# Keep terminal open on error
if [ $? -ne 0 ]; then
    echo ""
    echo "Application exited with error. Press Enter to close..."
    read
fi
