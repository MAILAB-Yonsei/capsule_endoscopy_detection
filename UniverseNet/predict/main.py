import os
import json
from glob import glob
from mmcv import Config
from mmdet.apis import init_detector
from functions import predictor, cut_samples, mAP_calc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='atss_swin-l_ms')
parser.add_argument('--gpu',type=int, default=0)
args = parser.parse_args()

gpu_ind = args.gpu
folder = 'ckpts/' + args.model

valid_path = '../data_coco/valid' # <<< valid set 경로
test_path  = '../data_coco/test' #  <<< test set  경로

weights = {}
files = os.listdir(folder)
for nm in files:
    if nm.split('_')[0] == 'epoch':
        ep = int(nm.split('_')[-1].split('.')[0])
        weights[ep] = nm
        
save_path = 'prediction_results' # save path
save_dir = os.path.join(folder, save_path)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# Choose to use a config and initialize the detector
config = os.path.join(folder, 'final.py')

mAP_result ={}
# weights = [/mnt/data1/GH/endoscopy/g_1/epoch_12.pth]

ll = list(weights.keys())
ll.sort(reverse=True)
for k in ll:
    i = weights[k]
    checkpoint = os.path.join(folder, i) # chekcpoint path
        
    num_cut_valid = [10000] # 
    num_cut_test  = [30000] # 

    #%% prediction (valid and test)
    cfg = Config.fromfile(config)
    model = init_detector(cfg, checkpoint, device='cuda:%d' % gpu_ind)
    
    # test
    print('test_set')
    test_file = glob(test_path+"/*.jpg")
    results = predictor(model, test_file)
    for num_cut in num_cut_test:
        cut_samples(results, num_cut, save_dir, args.model, i.split('.')[0], 'test')

    # # validation
    # print('val_set')
    # valid_file = glob(valid_path+"/*.jpg")
    # results = predictor(model, valid_file)
    
    # for num_cut in num_cut_valid:
    #     data = cut_samples(results, num_cut, save_dir, args.model, i.split('.')[0], 'valid')
    
    # val_mAP = mAP_calc('predict/val_answer.csv', data)
    # mAP_result[i.split('.')[0]] = val_mAP
    
    # with open(os.path.join(save_dir, 'result.json'), 'w') as outfile:
    #     json.dump(mAP_result, outfile, indent='\t')