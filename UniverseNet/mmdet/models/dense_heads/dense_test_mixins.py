# Copyright (c) OpenMMLab. All rights reserved.
import sys
from inspect import signature

import numpy as np
import torch

from mmdet.core import bbox_mapping_back, merge_aug_proposals, multiclass_nms

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed


class BBoxTestMixin(object):
    """Mixin class for testing det bboxes via DenseHead.

    Code for soft voting is modified from
    https://github.com/Scalsol/RepPointsV2/blob/741922d4e5155965850e3724f96590a971d49a4a/mmdet/models/detectors/reppoints_v2_detector.py
    """

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    def aug_test_bboxes_simple(self, feats, img_metas, rescale=False):
        """Test det bboxes with simple test-time augmentation, can be applied
        in DenseHead except for ``RPNHead`` and its variants, e.g.,
        ``GARPNHead``, etc.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The length of list should always be 1.
        """
        # check with_nms argument
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        if hasattr(self, '_get_bboxes'):
            gbs_sig = signature(self._get_bboxes)
        else:
            gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        aug_factors = []  # score_factors for NMS
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.forward(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            bbox_outputs = self.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])
            # bbox_outputs of some detectors (e.g., ATSS, FCOS, YOLOv3)
            # contains additional element to adjust scores before NMS
            if len(bbox_outputs) >= 3:
                aug_factors.append(bbox_outputs[2])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        merged_factors = torch.cat(aug_factors, dim=0) if aug_factors else None
        det_bboxes, det_labels = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            score_factors=merged_factors)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        return [
            (_det_bboxes, det_labels),
        ]

    def merge_aug_vote_results(self, aug_bboxes, aug_labels, img_metas):
        """Merge augmented detection bboxes and labels.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 5)
            aug_labels (list[Tensor] or None): shape (n, )
            img_metas (list[Tensor]): shape (3, ).

        Returns:
            tuple: (bboxes, labels)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes[:, :4] = bbox_mapping_back(bboxes[:, :4], img_shape,
                                              scale_factor, flip,
                                              flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_labels is None:
            return bboxes
        else:
            labels = torch.cat(aug_labels, dim=0)
            return bboxes, labels

    def remove_boxes(self, boxes, min_scale, max_scale):
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        in_range_idxs = torch.nonzero(
            (areas >= min_scale * min_scale) &
            (areas <= max_scale * max_scale),
            as_tuple=False).squeeze(1)
        return in_range_idxs

    def vote_bboxes(self, boxes, scores, vote_thresh=0.66):
        assert self.test_cfg.fusion_cfg.type == 'soft_vote'
        eps = 1e-6
        score_thr = self.test_cfg.score_thr

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy().reshape(-1, 1)
        det = np.concatenate((boxes, scores), axis=1)
        if det.shape[0] <= 1:
            return np.zeros((0, 5)), np.zeros((0, 1))
        order = det[:, 4].ravel().argsort()[::-1]
        det = det[order, :]
        dets = []
        while det.shape[0] > 0:
            # IOU
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = area[0] + area[:] - inter
            union = np.maximum(union, eps)
            o = inter / union
            o[0] = 1

            # get needed merge det and delete these det
            merge_index = np.where(o >= vote_thresh)[0]
            det_accu = det[merge_index, :]
            det_accu_iou = o[merge_index]
            det = np.delete(det, merge_index, 0)

            if merge_index.shape[0] <= 1:
                try:
                    dets = np.row_stack((dets, det_accu))
                except ValueError:
                    dets = det_accu
                continue
            else:
                soft_det_accu = det_accu.copy()
                soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
                soft_index = np.where(soft_det_accu[:, 4] >= score_thr)[0]
                soft_det_accu = soft_det_accu[soft_index, :]

                det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(
                    det_accu[:, -1:], (1, 4))
                max_score = np.max(det_accu[:, 4])
                det_accu_sum = np.zeros((1, 5))
                det_accu_sum[:, 0:4] = np.sum(
                    det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
                det_accu_sum[:, 4] = max_score

                if soft_det_accu.shape[0] > 0:
                    det_accu_sum = np.row_stack((det_accu_sum, soft_det_accu))

                try:
                    dets = np.row_stack((dets, det_accu_sum))
                except ValueError:
                    dets = det_accu_sum

        order = dets[:, 4].ravel().argsort()[::-1]
        dets = dets[order, :]

        boxes = torch.from_numpy(dets[:, :4]).float().cuda()
        scores = torch.from_numpy(dets[:, 4]).float().cuda()

        return boxes, scores

    def aug_test_bboxes_vote(self, feats, img_metas, rescale=False):
        """Test det bboxes with soft-voting test-time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        scale_ranges = self.test_cfg.fusion_cfg.scale_ranges
        num_same_scale_tta = len(feats) // len(scale_ranges)
        aug_bboxes = []
        aug_labels = []
        for aug_idx, (x, img_meta) in enumerate(zip(feats, img_metas)):
            # only one image in the batch
            outs = self.forward(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, True)
            det_bboxes, det_labels = self.get_bboxes(*bbox_inputs)[0]
            min_scale, max_scale = scale_ranges[aug_idx // num_same_scale_tta]
            in_range_idxs = self.remove_boxes(det_bboxes, min_scale, max_scale)
            det_bboxes = det_bboxes[in_range_idxs, :]
            det_labels = det_labels[in_range_idxs]
            aug_bboxes.append(det_bboxes)
            aug_labels.append(det_labels)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_labels = self.merge_aug_vote_results(
            aug_bboxes, aug_labels, img_metas)

        det_bboxes = []
        det_labels = []
        for j in range(self.num_classes):
            inds = (merged_labels == j).nonzero(as_tuple=False).squeeze(1)

            scores_j = merged_bboxes[inds, 4]
            bboxes_j = merged_bboxes[inds, :4].view(-1, 4)
            bboxes_j, scores_j = self.vote_bboxes(bboxes_j, scores_j)

            if len(bboxes_j) > 0:
                det_bboxes.append(
                    torch.cat([bboxes_j, scores_j[:, None]], dim=1))
                det_labels.append(
                    torch.full((bboxes_j.shape[0], ),
                               j,
                               dtype=torch.int64,
                               device=scores_j.device))

        if len(det_bboxes) > 0:
            det_bboxes = torch.cat(det_bboxes, dim=0)
            det_labels = torch.cat(det_labels)
        else:
            det_bboxes = merged_bboxes.new_zeros((0, 5))
            det_labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)

        max_per_img = self.test_cfg.max_per_img
        if det_bboxes.shape[0] > max_per_img > 0:
            cls_scores = det_bboxes[:, 4]
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), det_bboxes.shape[0] - max_per_img + 1)
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            det_bboxes = det_bboxes[keep]
            det_labels = det_labels[keep]

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        return [
            (_det_bboxes, det_labels),
        ]

    def aug_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes with test-time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        fusion_cfg = self.test_cfg.get('fusion_cfg', None)
        fusion_method = fusion_cfg.type if fusion_cfg else 'simple'
        if fusion_method == 'simple':
            return self.aug_test_bboxes_simple(feats, img_metas, rescale)
        elif fusion_method == 'soft_vote':
            return self.aug_test_bboxes_vote(feats, img_metas, rescale)
        else:
            raise ValueError('Unknown TTA fusion method')

    def simple_test_rpn(self, x, img_metas):
        """Test without augmentation, only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas)
        return proposal_list

    def aug_test_rpn(self, feats, img_metas):
        """Test with augmentation for only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                        a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        samples_per_gpu = len(img_metas[0])
        aug_proposals = [[] for _ in range(samples_per_gpu)]
        for x, img_meta in zip(feats, img_metas):
            proposal_list = self.simple_test_rpn(x, img_meta)
            for i, proposals in enumerate(proposal_list):
                aug_proposals[i].append(proposals)
        # reorganize the order of 'img_metas' to match the dimensions
        # of 'aug_proposals'
        aug_img_metas = []
        for i in range(samples_per_gpu):
            aug_img_meta = []
            for j in range(len(img_metas)):
                aug_img_meta.append(img_metas[j][i])
            aug_img_metas.append(aug_img_meta)
        # after merging, proposals will be rescaled to the original image size
        merged_proposals = [
            merge_aug_proposals(proposals, aug_img_meta, self.test_cfg)
            for proposals, aug_img_meta in zip(aug_proposals, aug_img_metas)
        ]
        return merged_proposals

    if sys.version_info >= (3, 7):

        async def async_simple_test_rpn(self, x, img_metas):
            sleep_interval = self.test_cfg.pop('async_sleep_interval', 0.025)
            async with completed(
                    __name__, 'rpn_head_forward',
                    sleep_interval=sleep_interval):
                rpn_outs = self(x)

            proposal_list = self.get_bboxes(*rpn_outs, img_metas)
            return proposal_list

    def merge_aug_bboxes(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,4), where
            4 represent (tl_x, tl_y, br_x, br_y)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores
