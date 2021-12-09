# capsule_endoscopy_detection
capsule_endoscopy_detection DACON challenge

### Overview

* Yolov5, yolor, 

** 모든 모델은 학습 시 Pretrained Weight을 yolov5 & mmdet github로부터 받아서 사용하였습니다.


## 환경(env) 세팅
* Ubuntu 18.04, Cuda 11.2
* Ananconda - Python 3.8

<cbnet을 제외한 나머지에 대한 env>
```
conda create -n all_except_cbnet python=3.8
pytorch 설치 (ex. conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch)
pip -r requirements_all_except_cbnet.txt
```
<cbnet에 대한 env>
```
conda create -n cbnet python=3.8
```
pytorch 설치 (ex. conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch)
```
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
```
bash STEP2로 시작하는 shell 실행 (model9. cbnet 제외하고)

(model9. cbnet 에 대한 shell 실행)
conda activate cbnet
bash STEP2_train_model9_cbnet_faster_rcnn_swin-l_ms
```
STEP3. 모든 모델에 대해 test를 진행한다. (shell 하나당 20~30분 소요)
```
conda activate all_except_cbnet
bash STEP3_inference_all_except_cbnet.sh
conda activate cbnet
bash STEP3_inference_cbnet.sh
```
* 만약 학습을 건너뛰고 pretrained 모델에 대해 test를 하고 싶다면, 구글 드라이브 링크로 받은 mmdetection/ckpts 폴더를 mmdetection 폴더 아래에 위치시킨다.
* 만약 학습을 건너뛰고 pretrained 모델에 대해 test를 하고 싶다면, 구글 드라이브 링크로 받은 UniverseNet/ckpts 폴더를 UniverseNet 폴더 아래에 위치시킨다.
* 만약 학습을 건너뛰고 pretrained 모델에 대해 test를 하고 싶다면, 구글 드라이브 링크로 받은 YOLO/ckpts 폴더를 YOLO 폴더 아래에 위치시킨다.

SETP4. 모든 모델에 대해 앙상블을 진행한다.
```
bash STEP4_ensemble.sh
```
