# capsule_endoscopy_detection
capsule_endoscopy_detection DACON challenge

### Overview

* Yolov5, Yolor, mmdetection의 모델을 사용

  * 모든 모델은 학습 시 Pretrained Weight을 yolov5, yolor, mmdetection 및 swin transformer github로부터 받아서 사용
  * 각 방식에 필요한 형태로 데이터의 format 변경

* Train set과 Validation set을 나누어 진행

* 총 11개의 결과를 앙상블 
  * detectors_casacde_rcnn_resnet50_multiscale, retinanet_swin-l, retinanet_swin-l_multiscale, retinanet_swin-t, atss_swin-l_multiscale, faster_rcnn-swin-l_multiscale, yolor_tta_multiscale, yolov5x, yolov5x_tta, yolov5x_tta_multiscale
  * Weighted boxes fusion (WBF) 방식으로 앙상블 진행 (Iou threshold = 0.4)

## 환경(env) 세팅
* Ubuntu 18.04, Cuda 11.2
* Ananconda - Python 3.8

#### cbnet을 제외한 나머지에 대한 env
```
conda create -n all_except_cbnet python=3.8
pytorch 설치 (ex. conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch)
pip -r requirements_all_except_cbnet.txt
```
#### cbnet에 대한 env
```
conda create -n cbnet python=3.8
pytorch 설치 (ex. conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
cd UniverseNet
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
pip install instaboostfast
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install albumentations>=0.3.2 --no-binary imgaug,albumentations
pip install shapely
```

## main code 실행
```
cd all_codes
```
STEP1. data_preparation (약 20~30분 소요)
```
conda activate all_except_cbnet
bash STEP1_data_preparation.sh
```
STEP2. 각 모델을 학습시킨다.

#### cbnet을 제외한 나머지에 대한 Training
```
conda activate all_except_cbnet
bash STEP2_train_model1_atss_swin-l_ms.sh
bash STEP2_train_model2_detectors_cascade_rcnn_r50_ms.sh
bash STEP2_train_model3_faster_rcnn_swin-l_ms.sh
bash STEP2_train_model4_retinanet_swin-l.sh
bash STEP2_train_model5_retinanet_swin-l_ms.sh
bash STEP2_train_model6_retinanet_swin-t_ms.sh
bash STEP2_train_model7_cbnet_faster_rcnn_swin-l_ms.sh
bash STEP2_train_model8_yolor.sh
```
#### cbnet에 대한 env
```
conda activate cbnet
bash STEP2_train_model9_cbnet_faster_rcnn_swin-l_ms.sh
```
STEP3. 모든 모델에 대해 Inference를 진행한다. (shell 하나당 20~30분 소요)

#### cbnet을 제외한 나머지에 대한 Inference
```
conda activate all_except_cbnet
bash STEP3_inference_all_except_cbnet.sh
```
#### cbnet에 대한 Inference
```
conda activate cbnet
bash STEP3_inference_cbnet.sh
```
* 만약 학습을 건너뛰고 pretrained 모델에 대해 test를 하고 싶다면, 구글 드라이브 링크로 받은 mmdetection/ckpts 폴더를 mmdetection 폴더 아래에 위치시킨다.
* 만약 학습을 건너뛰고 pretrained 모델에 대해 test를 하고 싶다면, 구글 드라이브 링크로 받은 UniverseNet/ckpts 폴더를 UniverseNet 폴더 아래에 위치시킨다.
* 만약 학습을 건너뛰고 pretrained 모델에 대해 test를 하고 싶다면, 구글 드라이브 링크로 받은 YOLO/ckpts 폴더를 YOLO 폴더 아래에 위치시킨다.
weight 파일 링크: https://drive.google.com/drive/folders/151KJC3FTUsK5mfx4TtNbhiFuuvLIeGz-?usp=sharing

SETP4. 모든 모델에 대해 앙상블을 진행한다.
```
conda activate all_except_cbnet.sh
bash STEP4_ensemble.sh
```

## 주의사항
#### 모두 순서에 맞게 코드를 구성해놓았기 때문에 하나의 코드를 2번 실행하는 등의 경우 진행에 어려움이 있을 수 있습니다. 참고해주세요.
#### 현재 코드는 validation은 진행하지 않게 주석처리했습니다. 원하시면 주석처리를 풀고 val_answer.csv 파일의 경로를 설정해주시면 됩니다.
