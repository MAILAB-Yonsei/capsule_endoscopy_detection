#!/usr/bin/env bash
cd ..
cd mmdetection

# mmdetection 라이브러리를 사용하여, atss method에 swin-l backbone을 사용하는 모델을 학습시킵니다. (multiscale)
# 가운데 숫자 1은 gpu 개수를 의미합니다.
tools/dist_train.sh configs/atss_swin-l_ms/final.py 1 --work-dir ckpts/atss_swin-l_ms