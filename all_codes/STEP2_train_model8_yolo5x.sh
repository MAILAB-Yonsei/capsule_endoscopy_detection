#!/usr/bin/env bash
cd ..
cd YOLO
cd yolov5

python train.py --img 576 --batch 16 --epochs 200 --data ../endoscopy.yaml --weights ckpts --project yolov5-endoscopy --save-period 1 --name endoscopy_1130 --device 0 --multi-scale
python train.py --img 576 --batch 16 --epochs 250 --data ../endoscopy.yaml --weights ckpts --project yolov5-endoscopy --save-period 1 --name endoscopy_1130 --device 0 --multi-scale
python train.py --img 576 --batch 16 --epochs 300 --data ../endoscopy.yaml --weights ckpts --project yolov5-endoscopy --save-period 1 --name endoscopy_1130 --device 0 --multi-scale