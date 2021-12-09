#!/usr/bin/env bash
python python_codes/split_data.py
python python_codes/yolo_preprocessing.py
python python_codes/convert_to_coco.py