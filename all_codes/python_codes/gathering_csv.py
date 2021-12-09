import shutil
import os
from glob import glob

folder_list = [
    '../mmdetection/ckpts/atss_swin-l_ms/prediction_results',
    '../mmdetection/ckpts/detectors_cascade_rcnn_r50_ms/prediction_results',
    '../mmdetection/ckpts/faster_rcnn_swin-l_ms/prediction_results',
    '../mmdetection/ckpts/retinanet_swin-l/prediction_results',
    '../mmdetection/ckpts/retinanet_swin-l_ms/prediction_results',
    '../mmdetection/ckpts/retinanet_swin-t_ms/prediction_results',
    '../UniverseNet/ckpts/cbnet_faster_rcnn_swin-l_ms/prediction_results',
    '../YOLO',
    ]

save_path = '../csv_results'
save_path_dir = os.path.join(save_path)
if not os.path.exists(save_path_dir):
    os.makedirs(save_path_dir)

for folder in folder_list:
    csv_files = glob('%s/*.csv' % folder)
    for csv_file in csv_files:
        shutil.copyfile(csv_file, "%s/%s" % (save_path_dir, os.path.basename(csv_file)))