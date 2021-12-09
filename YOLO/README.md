## INSTALLATION

1. conda create -n yolo python=3.8
2. conda activate yolo
3. conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
4. git clone https://github.com/MAILAB-Yonsei/detection.git
5. cd detection
6. pip install -r requirements.txt


If you want to use mish activation

<pre>
<code>
1. git clone https://github.com/JunnYu/mish-cuda

2. cd mish-cuda

3. python setup.py build install

4. cd ..
</code>
</pre>


If you want to use dwt down-sampling module

<pre>
<code>
1. git clone https://github.com/fbcotter/pytorch_wavelets

2. cd pytorch_wavelets

3. pip install .

4. cd ..
</code>
</pre>

## Preprocessing



yolo_preprocessing.py 파일을 실행시켜 yolo format에 맞게 데이터 전처리 진행


## YoloR 
#### TRAIN

<pre>
<code>
cd yolor

- multi scale 적용 x

python train.py --batch-size 16 --img-size 576 576 --data ../endoscopy.yaml --cfg cfg/yolor_w6.cfg --device 0 --sync-bn --name yolor_p6 --hyp hyp.scratch.1280.yaml --epochs 600 --weights weights path

- multi scale 적용 o

python train.py --batch-size 16 --img-size 576 576 --data ../endoscopy.yaml --cfg cfg/yolor_w6.cfg --device 0 --sync-bn --name yolor_p6 --hyp hyp.scratch.1280.yaml --epochs 600 --weights weights path --multi-scale
</code>
</pre>

#### DETECT
<pre>
<code>
- tta 적용 x

python detect.py --save-txt --source ../Data/DACON/yolo/images/test --weights weights path --cfg ./cfg/yolor_w6.cfg --device 0 --img-size 576 --output output path

- tta 적용 o 

python detect.py --save-txt --source ../Data/DACON/yolo/images/test --weights weights path --cfg ./cfg/yolor_w6.cfg --device 0 --img-size 576 --output output path --augment
</code>
</pre>

  

## Yolov5 
#### TRAIN

<pre>
<code>
cd yolov5

- multi scale 적용 x

python train.py --img 576 --batch 16 --epochs 350 --data ../endoscopy.yaml --weights weights path --project yolov5-endoscopy --save-period 1 --name endoscopy_1130 --device 0

- multi scale 적용 o

python train.py --img 576 --batch 16 --epochs 350 --data ../endoscopy.yaml --weights weights path --project yolov5-endoscopy --save-period 1 --name endoscopy_1130 --device 0 --multi-scale
</code>
</pre>

#### DETECT
<pre>
<code>
tta 적용 x

python detect.py --source ../Data/DACON/yolo/images/test --save-txt --save-conf --weight weights path --imgsz 576 --device 0 

tta 적용 o 

python detect.py --source ../Data/DACON/yolo/images/test --save-txt --save-conf --weight weights path --imgsz 576 --device 0 --augment
</code>
</pre>


## Test map csv 파일 생성
<pre>
<code>
cd ../
python test_scores.py --data data path --save save file path

예시) python test_scores.py --data ./inference/output32 --save final_submission_yolor_full.csv
</code>
</pre>
