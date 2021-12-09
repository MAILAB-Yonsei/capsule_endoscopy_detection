# This code is modified to understand mAP with rotated bounding box from https://github.com/facebookresearch/maskrcnn-benchmark
# --- Original comments --
# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)


from __future__ import division
import argparse
import json

import os
from collections import defaultdict
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

def do_voc_evaluation(gts_df, preds_df):
    """
    함수 개요 : map 산식을 계산하는 최종 함수
    
    gts_df : 답지 데이터프레임
    preds_df : 예측치 데이터프레임
    """

    """빈 리스트 2 개를 정의"""
    pred_boxlists = []
    gt_boxlists = []

    """gts_df 데이터의 file_name 컬럼의 유니크값을 인풋으로 한 for문"""
    unique_files = gts_df['file_name'].unique()
    # print(unique_files)
    pred_group_idx = preds_df.groupby('file_name').groups
    # print(pred_group_idx)
    gts_group_idx = gts_df.groupby('file_name').groups
    # print(gts_group_idx)
    
    for images_id in unique_files:
        try:
            pred_df_by_image_id = preds_df.loc[pred_group_idx[images_id]]
            # print(pred_df_by_image_id)
            pred_boxlists.append(pred_df_by_image_id)
        except:
            pred_boxlists.append(preds_df.head(0))
        gt_df_by_image_id = gts_df.loc[gts_group_idx[images_id]]
        gt_boxlists.append(gt_df_by_image_id)


    """저장된 리스트를 인풋으로 해 아래 함수를 실행해 result에 저장하고 리턴"""
    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5
    )

    return result

def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5):
    
    """pred_boxlists와 gt_boxlists의 길이가 같을 때만 실행"""
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
        
    """pred_boxlists와 gt_boxlists을 인풋으로 한 함수를 실행해 2개의 리턴값을 prec,rec에 각각 저장"""
    prec, rec = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=iou_thresh
    )
    
    """위에서 구한 prec와 rec을 이용해 아래 함수를 실행시키고 리턴된 리스트를 ap에 저장"""
    ap = calc_detection_voc_ap(prec, rec)

    """ap에서 NA인 것을 제외하고 평균을 내어 리턴"""
    return np.nanmean(ap)

def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):

    """n_pos, score, match 모두 파이썬의 딕셔너리 형태로 변수 생성"""
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
        
    """gt_boxlists와 pred_boxlists의 페어를 각각 gt_boxlist와 pred_boxlist에 저장하고 for문"""
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
                
        """pred_boxlist의 point 변수(point1_x, point1_y ...)를 array로 저장"""
        pred_bbox = np.array(
            [(rbox.point1_x, rbox.point1_y,
              rbox.point2_x, rbox.point2_y,
              rbox.point3_x, rbox.point3_y,
              rbox.point4_x, rbox.point4_y) for _, rbox in pred_boxlist.iterrows()])
                
        """pred_boxlist의 class_id 변수를 array로 저장"""
        pred_label = np.array([rbox.class_id for _, rbox in pred_boxlist.iterrows()])
        
        """pred_boxlist의 confidence 변수를 array로 저장"""
        pred_score = np.array([rbox.confidence for _, rbox in pred_boxlist.iterrows()])
        
        """gt_boxlist의 point 변수(point1_x, point1_y ...)를 array로 저장"""
        # print(gt_boxlist)
        gt_bbox = np.array(
            [(rbox.point1_x, rbox.point1_y,
              rbox.point2_x, rbox.point2_y,
              rbox.point3_x, rbox.point3_y,
              rbox.point4_x, rbox.point4_y) for _, rbox in gt_boxlist.iterrows()])
                
        """gt_boxlist의 class_id 변수를 array로 저장"""
        gt_label = np.array([rbox.class_id for _, rbox in gt_boxlist.iterrows()])
        
        """pred_label과 gt_label 중에서 유니크한 값을 l에 저장한 for문"""
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            
            """l과 pred_label의 요소가 같은 pred_label의 index에는 True를, 다른 index에는 False
            
            ex) (np.array([1,2,1,3,4,2,1]) == 1)는 아래와 같은 값이 반환
            array([ True, False, True, False,False,False,True]) """
            pred_mask_l = (pred_label == l)
            
            """pred_bbox 중에서 pred_mask_l이 True인 인덱스의 요소만 저장
            
            ex) (np.array([1,2,3])[np.array([True,False,True])])은 아래와 같은 값이 반환
            array([1,3])"""
            
            pred_bbox_l = (pred_bbox[pred_mask_l])
            
            """pred_score 중에서 pred_mask_l이 True인 인덱스의 요소만 저장"""
            pred_score_l = (pred_score[pred_mask_l])
            
            """pred_score_l을 정렬하고 그 인덱스를 반환"""
            order = pred_score_l.argsort()[::-1]
            
            """정렬된 인덱스를 기준으로 pred_bbox_l와 pred_score_l의 순서 바꾸기"""
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            """위와 비슷한 형식으로 gt에도 진행"""
            gt_mask_l = (gt_label == l)
            gt_bbox_l = (gt_bbox[gt_mask_l])

            """n_pos(dictionary 형태)의 l이라는 key에 gt_bbox_l의 길이를 더하기"""
            n_pos[l] += len(gt_bbox_l)
            
            """score(dictionary 형태)의 l이라는 key에 pred_score_l이란 리스트를 extend
            
            ex) np.array([1,1]).extend(np.array([3,2,1]))은 아래의 값을 반환
            np.array([1,1,3,2,1])"""
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            """pred_bbox_l을 저장하고 2번째 컬럼부터 끝까지 모든 요소에 1을 추가"""
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1

            """gt_bbox_l을 저장하고 2번째 컬럼부터 끝까지 모든 요소에 1을 추가"""
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            
            """pred_bbox_l과 gt_bbox_l을 인풋으로 rboxlist_iou라는 함수를 실행해 iou에 저장"""
            iou = rboxlist_iou(
                pred_bbox_l, gt_bbox_l
            )
            
            """iou중에서 요소가 최대인 인덱스를 gt_index에 저장"""
            gt_index = iou.argmax(axis=1)

            """iou의 각 행별로 최댓값이 iou_thresh보다 작은 행의 gt_index를 -1로 변환"""
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            
            """iou를 메모리에서 삭제"""
            del iou

            """gt_bbox_l의 길이와 같고 모든 element가 0으로 채워진 리스트를 생성"""
            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)

            """gt_index의 요소를 gt_idx에 저장한 for문"""
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:

                        """match[l]에 1을 추가"""
                        match[l].append(1)
                    else:

                        """match[l]에 0을 추가"""
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    
                    """match[l]에 0을 추가"""
                    match[l].append(0)

    """(n_pos의 키의 최댓값+1)을 n_fg_class에 저장"""
    n_fg_class = max(n_pos.keys()) + 1
    
    """길이가 n_fg_class이며 모든 element가 None인 리스트 생성
    ex) [1]*2 == [1,1]"""

    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        """cumsum은 cumulative sum의 약자로 해당 인덱스까지의 합을 리스트 형태로 반환한다.

        ex) np.cumsum(np.array([1,2,1,5]))는 아래와 같은 리스트를 반환한다.

            np.array([1,3,4,9])"""
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (fp + tp)

        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec

def rboxlist_iou(rboxlist1, rboxlist2):

    """행의 개수가 rboxlist1의 길이이고, 열의 개수가 rboxlist2의 길이이며 각 element가 모두 0인 행렬을 iou에 저장한다."""
    iou = np.zeros((len(rboxlist1), len(rboxlist2)))

    """rboxlist1의 인덱스와 요소들을 각각 idx1, rbox1에 저장한 for문"""
    for idx1, rbox1 in enumerate(rboxlist1):

        """rbox1의 요소를 이용해 파이썬의 Polygon(다각형) 클래스를 이용"""
        poly1 = Polygon([[rbox1[idx], rbox1[idx + 1]] for idx in range(0, 8, 2)])
        for idx2, rbox2 in enumerate(rboxlist2):

            """rbox2의 요소를 이용해 파이썬의 Polygon(다각형) 클래스를 이용"""
            poly2 = Polygon([[rbox2[idx], rbox2[idx + 1]] for idx in range(0, 8, 2)])
            
            """poly1과 poly2이 겹치는 부분의 넓이를 파이썬 내장 함수로 구해 inter에 저장

            참고 - https://www.swtestacademy.com/intersection-convex-polygons-algorithm/"""
            inter = poly1.intersection(poly2).area
            
            """inter과 각 다각형의 넓이를 이용해 iou의 엘레멘트를 바꾼다"""
            iou[idx1, idx2] = inter / (poly1.area + poly2.area - inter)
    return iou

def calc_detection_voc_ap(prec, rec):

    """prec의 길이를 n_fg_class에 저장"""
    n_fg_class = len(prec)
    
    """길이가 n_fg_class인 빈 리스트를 만들어준다(추후 과정을 위한 자리를 만들어 준다)"""
    ap = np.empty(n_fg_class)
    
    for l in range(n_fg_class):

        """prec의 l번째 요소 혹은 rec의 l번째 요소가 없다면 ap의 l번째 요소에 NA값을 넣는다."""
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue
        
        """np.nan_to_num(prec[l])은 prec[l]에 np.nan이 있으면 무조건 0으로 교체하는 함수이다.
        np.concatenate는 안에 있는 리스트를 컨캐트네잇하는(합치는) 함수이다."""
        mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
        mrec = np.concatenate(([0], rec[l], [1]))

        """np.maximum.accumulate는 각 인덱스 이전까지 element의 최댓값을 리스트 형태로 반환하는 함수이다.
        
        ex) np.maximum.accumulate(np.array([2,1,0,4,2,10]))은 아래의 리스트를 반환한다.
        
            np.array([2,2,2,4,4,10])"""
        
        """mpre[::-1]는 원래 mpre의 순서를 반대로 바꾸는 명령어다.
        
        ex) np.array([2,1,3])[::-1]은 아래의 리스트를 반환한다.
        
            np.array([3,1,2])"""
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]

        """mrec[1:]는 리스트의 두번째 요소부터(인덱스 1부터) 끝까지를 의미
        mrec[:-1]는 리스트의 첫번째 요소부터(인덱스 0부터) 마지막에서 2번째 요소까지를 의미"""
        i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
        ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-answer', '--answer', dest='answer')
    parser.add_argument('-data', '--data', dest='data')
    parser.add_argument('-print', nargs='?', const=1, type=int)
    options = parser.parse_args()

    if options.print is not None: 
        print("1. reading... %s" % options.answer)
    answer = pd.read_csv('/mnt/data2/DATA/DACON/val_answer.csv')
    # answer = pd.read_csv('/mnt/endoscopy/DACON/val_answer.csv')

    if options.print is not None: 
        print("2. reading... %s" % options.data)

    data = pd.read_csv('/mnt/data2/yj/model/detection/val_predict.csv')
    if options.print is not None: 
        print("evaluating...")

    #최종 결과가 아래입니다.
    print( do_voc_evaluation(answer, data) )

if __name__ == '__main__':
    main()