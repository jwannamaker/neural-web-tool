#!/bin/bash

cd /home/ubuntu/neural-web-tool

source .venv/bin/activate

pkill -f run.py || true

nohup python3 run.py > output.log 2>&1 &
