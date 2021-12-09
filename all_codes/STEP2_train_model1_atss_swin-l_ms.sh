#!/usr/bin/env bash
cd ..
cd mmdetection

tools/dist_train.sh configs/atss_swin-l_ms/final.py 1 --work-dir ckpts/atss_swin-l_ms