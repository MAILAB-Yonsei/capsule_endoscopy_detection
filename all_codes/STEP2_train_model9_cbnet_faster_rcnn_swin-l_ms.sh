#!/usr/bin/env bash
cd ..
cd UniverseNet

# UniverseNet 라이브러리를 사용하여, cbnet faster_rcnn method에 swin-l backbone을 사용하는 모델을 학습시킵니다. (multiscale)
tools/dist_train.sh configs/cbnet_faster_rcnn_swin-l_ms/final.py 1 --work-dir ckpts/cbnet_faster_rcnn_swin-l_ms