## INSTALLATION

1. CUDA, CUDNN에 맞는 torch 설치 : https://pytorch.org/get-started/locally/
2. mmdetection 설치 :
https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md

```
pip install ensemble-boxes
pip install pandas
pip install tqdm
pip install opencv-python
```
그외 필요한 라이브러리 설치


## Config file 설정
mmdetection/configs 에서 사용할 config 폴더의 dataset.py에서 data_root를 데이터가 있는 path로 변경

ex) mmdetection/configs/detectors_cascade_rcnn_r50_ms/dataset.py

```
data_root = '../data_coco/' # dataset path
```
## Train 
```
bash dist_train.sh [config파일] [사용 gpu 갯수] --work-dir [저장폴더명]

ex) bash mmdetection/dist_train.sh configs/detectors_cascade_rcnn_r50_mc/final.py 8 --work-dir result1
```
batch size 및 사용하는 gpu 갯수에 따라 learning rate를 수정해야함 (batch size = gpu 갯수 x sample per gpu > config에서 수정가능)

## Predict 
```
python predict.py --work-dir [5번에서 저장한 work-dir폴더]

ex) python predict.py --work-dir result1
```
predict를 진행하면 valid csv 파일과 mAP 결과, 그리고 test csv 파일이 만들어짐

