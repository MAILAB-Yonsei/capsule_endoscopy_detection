import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path
import base64
import cv2
import json 
from tqdm import tqdm
from joblib import Parallel , delayed
import multiprocessing as mp
import yaml

datapath = '../data_split'

IMG_SIZE = 576

#다운받은 데이터 경로(train,valid 9:1로 분할된 데이터)
base_path = Path(datapath)
train_path = list((base_path /'train').glob('train*'))
valid_path = list((base_path /'valid').glob('train*')) 
test_path = list((base_path / 'test').glob('test*'))

label_info = pd.read_csv((base_path /'class_id_info.csv'))
categories = {i[0]:i[1]-1 for i in label_info.to_numpy()}

def xyxy2coco(xyxy):
    x1,y1,x2,y2 =xyxy
    w,h =  x2-x1, y2-y1
    return [x1,y1,w,h] 

def xyxy2yolo(xyxy):
    
    x1,y1,x2,y2 =xyxy
    w,h =  x2-x1, y2-y1
    xc = x1 + int(np.round(w/2)) # xmin + width/2
    yc = y1 + int(np.round(h/2)) # ymin + height/2
    return [xc/IMG_SIZE,yc/IMG_SIZE,w/IMG_SIZE,h/IMG_SIZE] 

def scale_bbox(img, xyxy):
    # Get scaling factor
    scale_x = IMG_SIZE/img.shape[1]
    scale_y = IMG_SIZE/img.shape[0]
    
    x1,y1,x2,y2 =xyxy
    x1 = int(np.round(x1*scale_x, 4))
    y1 = int(np.round(y1*scale_y, 4))
    x2 = int(np.round(x2*scale_x, 4))
    y2= int(np.round(y2*scale_y, 4))

    return [x1, y1, x2, y2] # xmin, ymin, xmax, ymax

def save_image_label(json_file,mode,new_image_path,new_label_path): 
    with open(json_file,'r') as f: 
        json_file =json.load(f)

    image_id = json_file['file_name'].replace('.json','')
    
    # decode image data
    image = np.frombuffer(base64.b64decode(json_file['imageData']), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite(str(new_image_path / (image_id + '.png')) ,image)
    
    # extract bbox
    origin_bbox = []
    if mode == 'train':
        with open(new_label_path / (image_id + '.txt'), 'w') as f:
            for i in json_file['shapes']: 
                bbox = i['points'][0] + i['points'][2]
                origin_bbox.append(bbox)
                bbox = scale_bbox(image,bbox)
                bbox = xyxy2yolo(bbox)
                
                labels = [categories[i['label']]]+bbox
                f.writelines([f'{i} ' for i in labels] + ['\n'])
        
    return origin_bbox

os.makedirs('../data_yolo',exist_ok=True)
os.makedirs('../data_yolo/images/train',exist_ok=True)
os.makedirs('../data_yolo/labels/train',exist_ok=True)
os.makedirs('../data_yolo/images/valid',exist_ok=True)
os.makedirs('../data_yolo/labels/valid',exist_ok=True)
os.makedirs('../data_yolo/images/test',exist_ok=True)

#for train set
# 저장할 파일 경로
save_path = Path('../data_yolo')
new_image_path = save_path / 'images/train' # image폴더 
new_label_path = save_path / 'labels/train' # label폴더

new_image_path.mkdir(parents=True,exist_ok=True)
new_label_path.mkdir(parents=True,exist_ok=True)

# data를 생성하기 위해 mlutiprocessing 적용
tmp = Parallel(n_jobs=mp.cpu_count(),prefer="threads")(delayed(save_image_label)(str(train_json),'train',new_image_path,new_label_path) for train_json in tqdm(train_path[:]))


#for valid set
new_image_path = save_path / 'images/valid' # image폴더 
new_label_path = save_path / 'labels/valid' # label폴더

new_image_path.mkdir(parents=True,exist_ok=True)
new_label_path.mkdir(parents=True,exist_ok=True)

# data를 생성하기 위해 mlutiprocessing 적용
tmp = Parallel(n_jobs=mp.cpu_count(),prefer="threads")(delayed(save_image_label)(str(valid_json),'train',new_image_path,new_label_path) for valid_json in tqdm(valid_path[:]))


#for test set
new_image_path = save_path / 'images/test' # image폴더 
new_label_path = save_path / 'labels' # label폴더

new_image_path.mkdir(parents=True,exist_ok=True)
# new_label_path.mkdir(parents=True,exist_ok=True)

# data를 생성하기 위해 mlutiprocessing 적용
tmp = Parallel(n_jobs=mp.cpu_count(),prefer="threads")(delayed(save_image_label)(str(test_json),'test',new_image_path,new_label_path) for test_json in tqdm(test_path))

#yaml 파일 생성
data_yaml = dict(
    train = ['../data_yolo/images/train'],
    val = ['../data_yolo/yolo/images/valid'],
    nc = 4,
    names = ['01_ulcer','02_mass','04_lymph','05_bleeding']
)

with open('endoscopy.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)
    