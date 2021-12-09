#!/usr/bin/env bash
cd ..
cd mmdetection

# mmdetection 라이브러리로 학습한 6개의 모델을 inference 합니다.
# mmdetection/ckpts 내에 있는 각 method 별로 prediction_results 폴더가 생성되고, 그 안에 .csv 결과 파일이 저장됩니다.
python predict/main.py --model atss_swin-l_ms
python predict/main.py --model detectors_cascade_rcnn_r50_ms
python predict/main.py --model faster_rcnn_swin-l_ms
python predict/main.py --model retinanet_swin-l
python predict/main.py --model retinanet_swin-l_ms
python predict/main.py --model retinanet_swin-t_ms

# yolor에서 학습한 모델을 inference 합니다.
# detect.py 를 실행하여, 각 이미지 별로 .txt 형태의 output을 --output argument에 있는 경로에 저장합니다.
# --source는 yolo형 데이터의 test셋 경로입니다.
cd ..
cd YOLO/yolor
python detect.py --save-txt --source ../../data_yolo/images/test --weights ckpts/yolor_epoch_400_tta.pt --cfg ./cfg/yolor_w6.cfg --device 0 --img-size 576 --output ../inference/yolor_epoch_400_tta --augment
cd ..
# yolor의 output을 .csv 결과 파일로 변환하여, --save argument에 있는 경로에 저장합니다.
python test_scores.py --data ./inference/yolor_epoch_400_tta --save yolor_epoch_400_tta.csv


# yolov5에서 학습한 3개의 모델을 inference 합니다.
# detect.py 를 실행하여, 각 이미지 별로 .txt 형태의 output을 저장합니다. (저장 경로는 ../runs/detect/exp{실험번호} 입니다.)
# --source는 yolo형 데이터의 test셋 경로입니다. (저장 위치는 YOLO 폴더입니다.)
cd yolov5
python detect.py --source ../../data_yolo/images/test --save-txt --save-conf --weight ckpts/yolov5x_epoch_200.pt --imgsz 576 --device 0
python detect.py --source ../../data_yolo/images/test --save-txt --save-conf --weight ckpts/yolov5x_epoch_250_tta.pt --imgsz 576 --device 0 --augment 
python detect.py --source ../../data_yolo/images/test --save-txt --save-conf --weight ckpts/yolov5x_epoch_300_tta.pt --imgsz 576 --device 0 --augment
cd ..
# yolor의 output을 .csv 결과 파일로 변환하여, --save argument에 있는 경로에 저장합니다. (저장 위치는 YOLO 폴더입니다.)
python test_scores.py --data ./runs/detect/exp --save yolov5x_epoch_200.csv
python test_scores.py --data ./runs/detect/exp2 --save yolov5x_epoch_250_tta.csv
python test_scores.py --data ./runs/detect/exp3 --save yolov5x_epoch_300_tta.csv