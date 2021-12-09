import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-data', '--data', default='./inference/output32',dest='data')
parser.add_argument('-save', '--save', default='./final_submission_yolor_full.csv',dest='savepath')
options = parser.parse_args()

def cut_samples(results, num_cut, name):
    
    elements = ['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y',
                'point2_x', 'point2_y', 'point3_x', 'point3_y', 'point4_x', 'point4_y']
    
    results_cut = {
    'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
    'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[] }
    
    num_samples = len(results['confidence'])
    sorted_ind = np.argsort(np.array(results['confidence']))[::-1][:num_cut]
    
    
    for i in range(num_samples):
        if i in sorted_ind:
            for element in elements:
                results_cut[element].append(results[element][i])
    
    submission = pd.DataFrame(results_cut)
    submission.to_csv(os.path.join('./%s_%d.csv' % (name, num_cut)), index=False)
    
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.copy()
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y

total_list = []
results = {
    'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
    'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[]
}

result_path = Path(options.data)
result_img = list(result_path.glob('*.png'))
# print(result_img)
result_label = list(result_path.glob('labels/*.txt'))
# print(result_label)
for i in tqdm(result_label):
    # print(i)
    with open(str(i),'r') as f:

        file_name = i.name.replace('.txt','.json')
        img_name = file_name.replace('.json','.png')
        ow,oh,_ = cv2.imread(str(result_path / img_name))[:,:,::-1].shape
        
        for line in f.readlines():
            # print(line)
            corrdi = line[:-1].split(' ')
            label,xc,yc,w,h,score = corrdi
            xc,yc,w,h,score = list(map(float,[xc,yc,w,h,score]))
            if score > 0.22:
                xc,w = np.array([xc,w]) * ow
                yc,h = np.array([yc,h]) * oh

                refine_cordi = xywh2xyxy([xc,yc,w,h])
                refine_cordi = np.array(refine_cordi).astype(int)
                x_min,y_min,x_max,y_max = refine_cordi

                results['file_name'].append(file_name)
                results['class_id'].append(label)
                results['confidence'].append(score)
                results['point1_x'].append(x_min)
                results['point1_y'].append(y_min)
                results['point2_x'].append(x_max)
                results['point2_y'].append(y_min)
                results['point3_x'].append(x_max)
                results['point3_y'].append(y_max)
                results['point4_x'].append(x_min)
                results['point4_y'].append(y_max)
            
df = pd.DataFrame(results)
df['class_id'] = df['class_id'].apply(lambda x:int(x)+1)    
pd.DataFrame(df).to_csv(options.savepath, index = False)

# cut_samples(results, 10000, 'yolor_epochs425')
  