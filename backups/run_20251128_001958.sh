#!/bin/bash
# CleanDictate Runner Script
cd /home/sharadhnaidu/Desktop/CleanDictate
source venv/bin/activate
export LD_LIBRARY_PATH="$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$PWD/venv/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
python cleandictate.py
