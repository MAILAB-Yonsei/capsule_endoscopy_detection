#!/usr/bin/env bash
cd ..
cd mmdetection

tools/dist_train.sh configs/faster_rcnn_swin-l_ms/final.py 1 --work-dir ckpts/faster_rcnn_swin-l_ms