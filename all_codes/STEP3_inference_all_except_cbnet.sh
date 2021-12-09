#!/usr/bin/env bash
cd ..
cd mmdetection

# inference for each model (all 6 models)
python predict/main.py --model atss_swin-l_ms
python predict/main.py --model detectors_cascade_rcnn_r50_ms
python predict/main.py --model faster_rcnn_swin-l_ms
python predict/main.py --model retinanet_swin-l
python predict/main.py --model retinanet_swin-l_ms
python predict/main.py --model retinanet_swin-t_ms

cd ..
cd YOLO/yolor
python detect.py --save-txt --source ../../data_yolo/images/test --weights ckpts/yolor_epoch_400_tta.pt --cfg ./cfg/yolor_w6.cfg --device 0 --img-size 576 --output ../inference/yolor_epoch_400_tta --augment
cd ..
python test_scores.py --data ./inference/yolor_epoch_400_tta --save yolor_epoch_400_tta.csv

cd yolov5
python detect.py --source ../../data_yolo/images/test --save-txt --save-conf --weight ckpts/yolov5x_epoch_200.pt --imgsz 576 --device 0
python detect.py --source ../../data_yolo/images/test --save-txt --save-conf --weight ckpts/yolov5x_epoch_250_tta.pt --imgsz 576 --device 0 --augment 
python detect.py --source ../../data_yolo/images/test --save-txt --save-conf --weight ckpts/yolov5x_epoch_300_tta.pt --imgsz 576 --device 0 --augment
cd ..
python test_scores.py --data ./runs/detect/exp --save yolov5x_epoch_200.csv
python test_scores.py --data ./runs/detect/exp2 --save yolov5x_epoch_250_tta.csv
python test_scores.py --data ./runs/detect/exp3 --save yolov5x_epoch_300_tta.csv