#!/usr/bin/env bash
cd ..
cd mmdetection

tools/dist_train.sh configs/retinanet_swin-l/final.py 1 --work-dir ckpts/retinanet_swin-l