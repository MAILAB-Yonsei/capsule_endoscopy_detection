#!/usr/bin/env bash
cd ..
cd mmdetection

# mmdetection 라이브러리를 사용하여, detectors cascade-rcnn method에 resnet50 backbone을 사용하는 모델을 학습시킵니다. (multiscale)
# 가운데 숫자 1은 gpu 개수를 의미합니다.
tools/dist_train.sh configs/detectors_cascade_rcnn_r50_ms/final.py 1 --work-dir ckpts/detectors_cascade_rcnn_r50_ms