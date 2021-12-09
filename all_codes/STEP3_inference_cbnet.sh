#!/usr/bin/env bash
cd ..
cd UniverseNet

# UniverseNet 라이브러리로 학습한 cbnet 모델을 inference 합니다.
# UniverseNet/ckpts/cbnet_faster_rcnn_swin-l_ms 폴더에 prediction_results 폴더가 생성되고, 그 안에 .csv 결과 파일이 저장됩니다.
python predict/main.py --model cbnet_faster_rcnn_swin-l_ms