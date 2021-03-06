from __future__ import division
import os
import numpy as np
import pandas as pd

import mmcv
from tqdm import tqdm
from collections import defaultdict
from shapely.geometry import Polygon
from mmdet.apis import inference_detector


def predictor(model, test_file):
    results = {
    'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
    'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[] }
    
    for index, img_path in tqdm(enumerate(test_file), total = len(test_file)):
        
        file_name = img_path.split("/")[-1].split(".")[0]+".json"
    
        img = mmcv.imread(img_path)
        predictions = inference_detector(model, img)
        boxes, scores, labels = (list(), list(), list())
    
        for k, cls_result in enumerate(predictions):
            # print("cls_result", cls_result)
            if cls_result.size != 0:
                if len(labels)==0:
                    boxes = np.array(cls_result[:, :4])
                    scores = np.array(cls_result[:, 4])
                    labels = np.array([k+1]*len(cls_result[:, 4]))
                else:    
                    boxes = np.concatenate((boxes, np.array(cls_result[:, :4])))
                    scores = np.concatenate((scores, np.array(cls_result[:, 4])))
                    labels = np.concatenate((labels, [k+1]*len(cls_result[:, 4])))
    
        if len(labels) != 0:
            for label, score, bbox in zip(labels, scores, boxes):
                x_min, y_min, x_max, y_max = bbox.astype(np.int64)
    
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
                
    # results = pd.DataFrame(results)
    return results
                
def cut_samples(results, num_cut, save_path, model_name, weight, type):
    
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
    save_name =os.path.join('%s/%s_%s_%d_%s.csv' % (save_path, model_name, weight, num_cut, type))
    submission.to_csv(save_name, index=False)
    if type == 'valid':
        return save_name


def do_voc_evaluation(gts_df, preds_df):
    """
    ?????? ?????? : map ????????? ???????????? ?????? ??????
    
    gts_df : ?????? ??????????????????
    preds_df : ????????? ??????????????????
    """

    """??? ????????? 2 ?????? ??????"""
    pred_boxlists = []
    gt_boxlists = []

    """gts_df ???????????? file_name ????????? ??????????????? ???????????? ??? for???"""
    unique_files = gts_df['file_name'].unique()
    
    pred_group_idx = preds_df.groupby('file_name').groups
    gts_group_idx = gts_df.groupby('file_name').groups
    
    for images_id in unique_files:
        try:
            pred_df_by_image_id = preds_df.loc[pred_group_idx[images_id]]
            pred_boxlists.append(pred_df_by_image_id)
        except:
            pred_boxlists.append(preds_df.head(0))
        gt_df_by_image_id = gts_df.loc[gts_group_idx[images_id]]
        gt_boxlists.append(gt_df_by_image_id)


    """????????? ???????????? ???????????? ??? ?????? ????????? ????????? result??? ???????????? ??????"""
    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5
    )

    return result

def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5):
    
    """pred_boxlists??? gt_boxlists??? ????????? ?????? ?????? ??????"""
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
        
    """pred_boxlists??? gt_boxlists??? ???????????? ??? ????????? ????????? 2?????? ???????????? prec,rec??? ?????? ??????"""
    prec, rec = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=iou_thresh
    )
    
    """????????? ?????? prec??? rec??? ????????? ?????? ????????? ??????????????? ????????? ???????????? ap??? ??????"""
    ap = calc_detection_voc_ap(prec, rec)

    """ap?????? NA??? ?????? ???????????? ????????? ?????? ??????"""
    return np.nanmean(ap)

def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):

    """n_pos, score, match ?????? ???????????? ???????????? ????????? ?????? ??????"""
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
        
    """gt_boxlists??? pred_boxlists??? ????????? ?????? gt_boxlist??? pred_boxlist??? ???????????? for???"""
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
                
        """pred_boxlist??? point ??????(point1_x, point1_y ...)??? array??? ??????"""
        pred_bbox = np.array(
            [(rbox.point1_x, rbox.point1_y,
              rbox.point2_x, rbox.point2_y,
              rbox.point3_x, rbox.point3_y,
              rbox.point4_x, rbox.point4_y) for _, rbox in pred_boxlist.iterrows()])
                
        """pred_boxlist??? class_id ????????? array??? ??????"""
        pred_label = np.array([rbox.class_id for _, rbox in pred_boxlist.iterrows()])
        
        """pred_boxlist??? confidence ????????? array??? ??????"""
        pred_score = np.array([rbox.confidence for _, rbox in pred_boxlist.iterrows()])
        
        """gt_boxlist??? point ??????(point1_x, point1_y ...)??? array??? ??????"""
        gt_bbox = np.array(
            [(rbox.point1_x, rbox.point1_y,
              rbox.point2_x, rbox.point2_y,
              rbox.point3_x, rbox.point3_y,
              rbox.point4_x, rbox.point4_y) for _, rbox in gt_boxlist.iterrows()])
                
        """gt_boxlist??? class_id ????????? array??? ??????"""
        gt_label = np.array([rbox.class_id for _, rbox in gt_boxlist.iterrows()])
        
        """pred_label??? gt_label ????????? ???????????? ?????? l??? ????????? for???"""
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            
            """l??? pred_label??? ????????? ?????? pred_label??? index?????? True???, ?????? index?????? False
            
            ex) (np.array([1,2,1,3,4,2,1]) == 1)??? ????????? ?????? ?????? ??????
            array([ True, False, True, False,False,False,True]) """
            pred_mask_l = (pred_label == l)
            
            """pred_bbox ????????? pred_mask_l??? True??? ???????????? ????????? ??????
            
            ex) (np.array([1,2,3])[np.array([True,False,True])])??? ????????? ?????? ?????? ??????
            array([1,3])"""
            
            pred_bbox_l = (pred_bbox[pred_mask_l])
            
            """pred_score ????????? pred_mask_l??? True??? ???????????? ????????? ??????"""
            pred_score_l = (pred_score[pred_mask_l])
            
            """pred_score_l??? ???????????? ??? ???????????? ??????"""
            order = pred_score_l.argsort()[::-1]
            
            """????????? ???????????? ???????????? pred_bbox_l??? pred_score_l??? ?????? ?????????"""
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            """?????? ????????? ???????????? gt?????? ??????"""
            gt_mask_l = (gt_label == l)
            gt_bbox_l = (gt_bbox[gt_mask_l])

            """n_pos(dictionary ??????)??? l????????? key??? gt_bbox_l??? ????????? ?????????"""
            n_pos[l] += len(gt_bbox_l)
            
            """score(dictionary ??????)??? l????????? key??? pred_score_l?????? ???????????? extend
            
            ex) np.array([1,1]).extend(np.array([3,2,1]))??? ????????? ?????? ??????
            np.array([1,1,3,2,1])"""
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            """pred_bbox_l??? ???????????? 2?????? ???????????? ????????? ?????? ????????? 1??? ??????"""
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1

            """gt_bbox_l??? ???????????? 2?????? ???????????? ????????? ?????? ????????? 1??? ??????"""
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            
            """pred_bbox_l??? gt_bbox_l??? ???????????? rboxlist_iou?????? ????????? ????????? iou??? ??????"""
            iou = rboxlist_iou(
                pred_bbox_l, gt_bbox_l
            )
            
            """iou????????? ????????? ????????? ???????????? gt_index??? ??????"""
            gt_index = iou.argmax(axis=1)

            """iou??? ??? ????????? ???????????? iou_thresh?????? ?????? ?????? gt_index??? -1??? ??????"""
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            
            """iou??? ??????????????? ??????"""
            del iou

            """gt_bbox_l??? ????????? ?????? ?????? element??? 0?????? ????????? ???????????? ??????"""
            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)

            """gt_index??? ????????? gt_idx??? ????????? for???"""
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:

                        """match[l]??? 1??? ??????"""
                        match[l].append(1)
                    else:

                        """match[l]??? 0??? ??????"""
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    
                    """match[l]??? 0??? ??????"""
                    match[l].append(0)

    """(n_pos??? ?????? ?????????+1)??? n_fg_class??? ??????"""
    n_fg_class = max(n_pos.keys()) + 1
    
    """????????? n_fg_class?????? ?????? element??? None??? ????????? ??????
    ex) [1]*2 == [1,1]"""

    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        """cumsum??? cumulative sum??? ????????? ?????? ?????????????????? ?????? ????????? ????????? ????????????.

        ex) np.cumsum(np.array([1,2,1,5]))??? ????????? ?????? ???????????? ????????????.

            np.array([1,3,4,9])"""
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (fp + tp)

        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec

def rboxlist_iou(rboxlist1, rboxlist2):

    """?????? ????????? rboxlist1??? ????????????, ?????? ????????? rboxlist2??? ???????????? ??? element??? ?????? 0??? ????????? iou??? ????????????."""
    iou = np.zeros((len(rboxlist1), len(rboxlist2)))

    """rboxlist1??? ???????????? ???????????? ?????? idx1, rbox1??? ????????? for???"""
    for idx1, rbox1 in enumerate(rboxlist1):

        """rbox1??? ????????? ????????? ???????????? Polygon(?????????) ???????????? ??????"""
        poly1 = Polygon([[rbox1[idx], rbox1[idx + 1]] for idx in range(0, 8, 2)])
        for idx2, rbox2 in enumerate(rboxlist2):

            """rbox2??? ????????? ????????? ???????????? Polygon(?????????) ???????????? ??????"""
            poly2 = Polygon([[rbox2[idx], rbox2[idx + 1]] for idx in range(0, 8, 2)])
            
            """poly1??? poly2??? ????????? ????????? ????????? ????????? ?????? ????????? ?????? inter??? ??????

            ?????? - https://www.swtestacademy.com/intersection-convex-polygons-algorithm/"""
            inter = poly1.intersection(poly2).area
            
            """inter??? ??? ???????????? ????????? ????????? iou??? ??????????????? ?????????"""
            iou[idx1, idx2] = inter / (poly1.area + poly2.area - inter)
    return iou

def calc_detection_voc_ap(prec, rec):

    """prec??? ????????? n_fg_class??? ??????"""
    n_fg_class = len(prec)
    
    """????????? n_fg_class??? ??? ???????????? ???????????????(?????? ????????? ?????? ????????? ????????? ??????)"""
    ap = np.empty(n_fg_class)
    
    for l in range(n_fg_class):

        """prec??? l?????? ?????? ?????? rec??? l?????? ????????? ????????? ap??? l?????? ????????? NA?????? ?????????."""
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue
        
        """np.nan_to_num(prec[l])??? prec[l]??? np.nan??? ????????? ????????? 0?????? ???????????? ????????????.
        np.concatenate??? ?????? ?????? ???????????? ?????????????????????(?????????) ????????????."""
        mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
        mrec = np.concatenate(([0], rec[l], [1]))

        """np.maximum.accumulate??? ??? ????????? ???????????? element??? ???????????? ????????? ????????? ???????????? ????????????.
        
        ex) np.maximum.accumulate(np.array([2,1,0,4,2,10]))??? ????????? ???????????? ????????????.
        
            np.array([2,2,2,4,4,10])"""
        
        """mpre[::-1]??? ?????? mpre??? ????????? ????????? ????????? ????????????.
        
        ex) np.array([2,1,3])[::-1]??? ????????? ???????????? ????????????.
        
            np.array([3,1,2])"""
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]

        """mrec[1:]??? ???????????? ????????? ????????????(????????? 1??????) ???????????? ??????
        mrec[:-1]??? ???????????? ????????? ????????????(????????? 0??????) ??????????????? 2?????? ??????????????? ??????"""
        i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
        ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def mAP_calc(answer, data):
    answer = pd.read_csv(answer)
    data = pd.read_csv(data)

    #?????? ????????? ???????????????.
    result = do_voc_evaluation(answer, data)
    print(result)
    
    return result