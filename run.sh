#!/bin/bash

cd /home/ubuntu/neural-web-tool

source .venv/bin/activate

pkill -f app.py || true

nohup python3 app.py > output.log 2>&1 &
