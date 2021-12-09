#!/usr/bin/env bash
cd ..
cd mmdetection

tools/dist_train.sh configs/retinanet_swin-t_ms/final.py 1 --work-dir ckpts/retinanet_swin-t_ms