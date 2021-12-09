#!/usr/bin/env bash
cd ..
cd UniverseNet

tools/dist_train.sh configs/cbnet_faster_rcnn_swin-l_ms/final.py 1 --work-dir ckpts/cbnet_faster_rcnn_swin-l_ms