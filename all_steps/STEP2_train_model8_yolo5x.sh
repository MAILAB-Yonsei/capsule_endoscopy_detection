#!/usr/bin/env bash
cd ..
cd YOLO
cd yolov5

# 앙상블에 사용된 세 종류의 yolov5 모델을 학습합니다.
# 각각 epoch 200, 250, 300 모델입니다.
python train.py --img 576 --batch 16 --epochs 200 --data ../endoscopy.yaml --weights ckpts --project yolov5-endoscopy --save-period 1 --name endoscopy_1130 --device 0 --multi-scale
python train.py --img 576 --batch 16 --epochs 250 --data ../endoscopy.yaml --weights ckpts --project yolov5-endoscopy --save-period 1 --name endoscopy_1130 --device 0 --multi-scale
python train.py --img 576 --batch 16 --epochs 300 --data ../endoscopy.yaml --weights ckpts --project yolov5-endoscopy --save-period 1 --name endoscopy_1130 --device 0 --multi-scale