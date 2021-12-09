import cv2
import pandas as pd
import os
from tqdm import tqdm
from ensemble_boxes import *
from pathlib import Path
import shutil

### test set 이미지 위치 경로 
img_path = '../data_coco/test/'

### 앙상블하고자 하는 csv 파일들을 하나의 폴더에 넣어두고, 그 폴더의 경로를 적으면 됨
folder_path = '../csv_results/'
csv_files = os.listdir(folder_path)
csv_files = [os.path.join(folder_path, i) for i in csv_files]

### 최종 저장되는 앙상블 csv 파일명 
final_saved_name = '../final.csv'

#--------------------- 최적 값으로 setting 되어있음 -------------------
### 101~108 번째 줄 보면 ensemble 방식을 선택할 수 있음. 방식에 따른 변수들 설정 
iou_thr = 0.4  
iou_thr = 0.4 
skip_box_thr = 0.0001
sigma = 0.1

weights = []
for w in range(len(csv_files)):
    weights.append(1)  

tmp = 'tmp_t'
if not os.path.exists(tmp):
    os.makedirs(tmp)

########### split original csv files ##########
for i in range(len(csv_files)):
    print(f'### processing >>>>>>> {csv_files[i]} ###')
    save_path = f'{tmp}/{i}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df = pd.read_csv(csv_files[i])
    for index, row in tqdm(df.iterrows()):
        file_name = row['file_name']
        confidence = row['confidence']
        classes = row['class_id']
        x1 = row['point1_x']
        y1 = row['point1_y']
        x2 = row['point3_x']
        y2 = row['point3_y']
        
        img_file = file_name.split('.')[0]+'.jpg'
        img = cv2.imread(os.path.join(img_path,img_file))
        h = img.shape[0]
        
        x1 /= h
        y1 /= h
        x2 /= h
        y2 /= h
        saved = file_name.split('.')[0]
        with open(f'./{tmp}/{i}/{saved}.csv','a') as f:
            f.write('{},{},{},{},{},{}\n'.format(classes,confidence,x1,y1,x2,y2))
            f.close()   
        
################ ensemble ##################
print('### ensemble ###')
file_list = []
for dirpath, dirnames, filenames in os.walk(img_path):
    for filename in filenames:
        ext = filename.split('.')[-1]
        if ext == 'jpg':
            file_list.append(filename.split('.')[0] +'.csv')
file_list.sort()

save_dir = f'{tmp}/ensemble'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for filename in file_list:
    box_list = []
    score_list = []
    label_list = []
    
    data = {'base':[]}
    for i in range(len(csv_files)):
        data['base'].append(f'{tmp}/{i}')
    
    check = []
    for k in range(len(data['base'])):
        gogo = os.path.isfile(os.path.join(data['base'][k], filename))
        check.append(gogo)
        if gogo:
            label_list1 = []
            score_list1 = []
            box_list1 = []
            file1 = os.path.join(data['base'][k], filename)
            file1_csv = pd.read_csv(file1, header=None)
            for a in range(len(file1_csv[0])):
                label_list1.append(file1_csv[0][a])
                score_list1.append(file1_csv[1][a])
                box_list1.append([file1_csv[2][a], file1_csv[3][a], file1_csv[4][a], file1_csv[5][a]])
            label_list.append(label_list1)
            score_list.append(score_list1)
            box_list.append(box_list1)
            
    try:
        check.index(True)
    except:
        continue
    
    # boxes, scores, labels = nms(
    #     box_list, score_list, label_list, weights=weights, iou_thr=iou_thr)
    # boxes, scores, labels = soft_nms(
    #     box_list, score_list, label_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    # boxes, scores, labels = non_maximum_weighted(
    #     box_list, score_list, label_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes, scores, labels = weighted_boxes_fusion(
        box_list, score_list, label_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

    for k in range(len(boxes)):
        with open(f'./{save_dir}/{filename}','a') as f:
            f.write('{},{},{},{},{},{}\n'.format(labels[k],scores[k],boxes[k][0],boxes[k][1],boxes[k][2],boxes[k][3]))
            f.close()
            
############### final conversion csv files ##################
print('### final processing ###')

en_path = list((Path(save_dir)).glob('*.csv'))

with open(final_saved_name,'w') as f:
    f.write('file_name,class_id,confidence,point1_x,point1_y,point2_x,point2_y,point3_x,point3_y,point4_x,point4_y\n')
    for files in tqdm(en_path):
        df = pd.read_csv(files, names=['class_id','confidence','point1_x','point1_y','point3_x','point3_y'])
        
        file_name = str(files).split('/')[-1].split('.')[0] + '.json'
        for index, row in df.iterrows():
            confidence = row['confidence']
            classes = row['class_id']
            x1 = row['point1_x']
            y1 = row['point1_y']
            x3 = row['point3_x']
            y3 = row['point3_y']
            
            img_file = file_name.split('.')[0]+'.jpg'
            img = cv2.imread(os.path.join(img_path,img_file))
            h = img.shape[0]
            
            x1 *= h
            y1 *= h
            x3 *= h
            y3 *= h
            
            x1 = int(x1)
            y1 = int(y1)
            x3 = int(x3)
            y3 = int(y3)
            
            x2 = x3
            y2 = y1
            x4 = x1
            y4 = y3
            
            f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(file_name,classes,confidence,x1,y1,x2,y2,x3,y3,x4,y4))
    f.close()            

df = pd.read_csv(final_saved_name)
df.sort_values(by='confidence', ascending=False, inplace=True)
df = df.reset_index(drop=True)

for i in range(29999, len(df)):
    df = df.drop(index=i)

df.to_csv(final_saved_name, index=False)
shutil.rmtree(f'{tmp}')
print('end')
