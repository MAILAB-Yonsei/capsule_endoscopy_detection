#!/usr/bin/env bash
cd ..
cd UniverseNet

# inference for each model (one model)
python predict/main.py --model cbnet_faster_rcnn_swin-l_ms