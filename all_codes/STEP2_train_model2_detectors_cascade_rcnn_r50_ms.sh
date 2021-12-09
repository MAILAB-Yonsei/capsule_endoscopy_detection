#!/usr/bin/env bash
cd ..
cd mmdetection

tools/dist_train.sh configs/detectors_cascade_rcnn_r50_ms/final.py 1 --work-dir ckpts/detectors_cascade_rcnn_r50_ms