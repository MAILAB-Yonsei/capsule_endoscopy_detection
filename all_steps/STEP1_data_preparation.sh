#!/usr/bin/env bash

# Train 폴더의 데이터를 Train data (90%)와 Validation data (10%) 로 나눕니다.
python python_codes/split_data.py

# YOLO를 위한 데이터 전처리를 수행하여 data_yolo 폴더에 저장합니다.
python python_codes/yolo_preprocessing.py

# mmdetection을 위한 데이터 전처리를 수행하여 data_coco 폴더에 저장합니다.
python python_codes/convert_to_coco.py