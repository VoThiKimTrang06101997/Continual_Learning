# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList,OptConfigType, OptMultiConfig
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .base_roi_head import BaseRoIHead
from .standard_roi_head import StandardRoIHead

import os.path as osp

import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np
import scipy
import scipy.ndimage
# from torch_cluster import dbscan
import copy
@MODELS.register_module()
class StandardRoIReplayHead(StandardRoIHead):
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg)
        self.replay = False
        if previous_path != None and osp.exists(previous_path):
            self.replay = True
            print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
            self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"))
            
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples, replay=True) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples)
        device = next(self.parameters()).device
        if self.replay and replay:
            # do sampling
            mask = torch.randperm(self.bbox_featss.shape[0])[:64].to(self.bbox_featss.device)
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = self.bbox_featss[mask], self.cls_targets[mask], self.cls_weights[mask], self.bbox_targets[mask], self.bbox_weights[mask], self.roiss[mask]
            
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = bbox_featss.to(device), cls_targets.to(device), cls_weights.to(device), bbox_targets.to(device), bbox_weights.to(device), roiss.to(device)
            
            sampling_results = [cls_targets, cls_weights, bbox_targets, bbox_weights]
            replay_results = self.replay_loss(bbox_featss, sampling_results, roiss)
            # print("before update", losses)
            losses.update(replay_results['replay_loss'])
            # print("replay loss", replay_results['replay_loss'])
            # print("after update", losses)
        return losses
        
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        # teacher model
        teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        losses["replay_loss_cls"] = F.mse_loss(cls_score, teacher_cls_score)
        # losses["replay_loss_reg"] = F.mse_loss(bbox_pred, teacher_bbox_pred)

        # bbox_loss_and_target = self.bbox_head.replay_loss(
        #     cls_score=bbox_results['cls_score'],
        #     bbox_pred=bbox_results['bbox_pred'],
        #     rois=rois,
        #     sampling_results=sampling_results,
        #     rcnn_train_cfg=self.train_cfg)

        bbox_results.update(replay_loss=losses)
        return bbox_results
    
    
    def get_bbox_stuff(self, x, rpn_results_list, batch_data_samples):
        
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)
        
        rois = bbox2roi([res.priors for res in sampling_results])
        
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        
        # print(torch.sort(cls))
        
        cls_target, cls_weight, bbox_target, bbox_weight = self.bbox_head.get_roi_targets(sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)
        
        mask = cls_target != 20
        
        target_count = 5
        
        # 计算当前True的数量
        current_count = torch.sum(mask).item()

        # 确定需要增加或减少的True的数量
        delta = target_count - current_count

        if delta > 0:
            # 需要增加True的数量
            num_to_add = delta
            # 获取所有False的索引
            false_indices = torch.where(mask == False)[0]
            # 如果False的数量小于需要增加的数量，就全部设置为True
            if len(false_indices) < num_to_add:
                mask[:] = True
            else:
                # 随机选择一些False的位置并设置为True
                indices_to_add = torch.randperm(len(false_indices))[:num_to_add]
                mask[false_indices[indices_to_add]] = True
        elif delta < 0:
            # 需要减少True的数量
            num_to_remove = -delta
            # 获取所有True的索引
            true_indices = torch.where(mask == True)[0]
            # 随机选择一些True的位置并设置为False
            indices_to_remove = torch.randperm(len(true_indices))[:num_to_remove]
            mask[true_indices[indices_to_remove]] = False
        
        return bbox_feats[mask], cls_target[mask], cls_weight[mask], bbox_target[mask], bbox_weight[mask], rois[mask]
    
    
    # def get_drift_anchor(self, x, rpn_results_list: InstanceList,
            #  batch_data_samples: List[DetDataSample],  List_bboxes, lab):    
    def get_drift_anchor(self, x, rpn_results_list, batch_data_samples, sampling_results=None, mask=None): 
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        # assign gts and sample proposals
        if sampling_results == None:
            num_imgs = len(batch_data_samples)
            sampling_results = []
            for i in range(num_imgs):
                # rename rpn_results.bboxes to rpn_results.priors
                rpn_results = rpn_results_list[i]
                rpn_results.priors = rpn_results.pop('bboxes')

                assign_result = self.bbox_assigner.assign(
                    rpn_results, batch_gt_instances[i],
                    batch_gt_instances_ignore[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    rpn_results,
                    batch_gt_instances[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            
        rois = bbox2roi([res.priors for res in sampling_results])
            
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        
        # print(torch.sort(cls))
        
        cls_target = self.bbox_head.get_roi_targets(sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg)[0]
        
        if mask == None:
            mask = cls_target != 20
            
            target_count = 5
            
            # 计算当前True的数量
            current_count = torch.sum(mask).item()

            # 确定需要增加或减少的True的数量
            delta = target_count - current_count

            if delta > 0:
                # 需要增加True的数量
                num_to_add = delta
                # 获取所有False的索引
                false_indices = torch.where(mask == False)[0]
                # 如果False的数量小于需要增加的数量，就全部设置为True
                if len(false_indices) < num_to_add:
                    mask[:] = True
                else:
                    # 随机选择一些False的位置并设置为True
                    indices_to_add = torch.randperm(len(false_indices))[:num_to_add]
                    mask[false_indices[indices_to_add]] = True
            elif delta < 0:
                # 需要减少True的数量
                num_to_remove = -delta
                # 获取所有True的索引
                true_indices = torch.where(mask == True)[0]
                # 随机选择一些True的位置并设置为False
                indices_to_remove = torch.randperm(len(true_indices))[:num_to_remove]
                mask[true_indices[indices_to_remove]] = False
        
        return bbox_feats[mask], cls_target[mask], sampling_results, mask
        
    
    # def predict(self,
    #             x: Tuple[Tensor],
    #             rpn_results_list: InstanceList,
    #             batch_data_samples: SampleList,
    #             rescale: bool = False) -> InstanceList:
    #     """Perform forward propagation of the roi head and predict detection
    #     results on the features of the upstream network.

    #     Args:
    #         x (tuple[Tensor]): Features from upstream network. Each
    #             has shape (N, C, H, W).
    #         rpn_results_list (list[:obj:`InstanceData`]): list of region
    #             proposals.
    #         batch_data_samples (List[:obj:`DetDataSample`]): The Data
    #             Samples. It usually includes information such as
    #             `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
    #         rescale (bool): Whether to rescale the results to
    #             the original image. Defaults to True.

    #     Returns:
    #         list[obj:`InstanceData`]: Detection results of each image.
    #         Each item usually contains following keys.

    #             - scores (Tensor): Classification scores, has a shape
    #               (num_instance, )
    #             - labels (Tensor): Labels of bboxes, has a shape
    #               (num_instances, ).
    #             - bboxes (Tensor): Has a shape (num_instances, 4),
    #               the last dimension 4 arrange as (x1, y1, x2, y2).
    #             - masks (Tensor): Has a shape (num_instances, H, W).
    #     """
    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #     batch_img_metas = [
    #         data_samples.metainfo for data_samples in batch_data_samples
    #     ]

        
    #     assert len(rpn_results_list) == len(batch_data_samples), f"{len(rpn_results_list)} != {len(batch_data_samples)}"
    #     outputs = unpack_gt_instances(batch_data_samples)
    #     batch_gt_instances, batch_gt_instances_ignore, _ = outputs

    #     # assign gts and sample proposals
    #     num_imgs = len(batch_data_samples)
    #     sampling_results = []
    #     for i in range(num_imgs):
    #         # rename rpn_results.bboxes to rpn_results.priors
    #         rpn_results = rpn_results_list[i]
    #         rpn_results.priors = rpn_results.pop('bboxes')

    #         assign_result = self.bbox_assigner.assign(
    #             rpn_results, batch_gt_instances[i],
    #             batch_gt_instances_ignore[i])
    #         sampling_result = self.bbox_sampler.sample(
    #             assign_result,
    #             rpn_results,
    #             batch_gt_instances[i],
    #             feats=[lvl_feat[i][None] for lvl_feat in x])
    #         sampling_results.append(sampling_result)
        
        
    #     cls_target, cls_weight, bbox_target, bbox_weight = self.bbox_head.get_roi_targets(sampling_results=sampling_results,
    #         rcnn_train_cfg=self.train_cfg)
    #     # print("cls_target", cls_target)
        
    #     # TODO: nms_op in mmcv need be enhanced, the bbox result may get
    #     #  difference when not rescale in bbox_head

    #     # If it has the mask branch, the bbox branch does not need
    #     # to be scaled to the original image scale, because the mask
    #     # branch will scale both bbox and mask at the same time.
    #     bbox_rescale = rescale if not self.with_mask else False
    #     bboxes = [
    #         data_sample.gt_instances for data_sample in batch_data_samples
    #     ]
    #     bboxes = [img.bboxes for img in bboxes]
    #     bboxes = self.aug_bbox(bboxes)
    #     results_list = self.predict_bbox(
    #         x,
    #         batch_img_metas,
    #         sampling_results,
    #         rcnn_test_cfg=self.test_cfg,
    #         rescale=bbox_rescale, List_bbox=bboxes, cls_targets=cls_target)

    #     if self.with_mask:
    #         results_list = self.predict_mask(
    #             x, batch_img_metas, results_list, rescale=rescale)

    #     return results_list

    # def aug_bbox(self, bboxes):
    #     for i, obbox in enumerate(bboxes):
    #         tmp = []
    #         for shift1 in [0, 20, 50, 100]:
    
    #                         bbox = copy.deepcopy(obbox)
    #                         bbox[:, 0] = bbox[:, 0]
    #                         bbox[:, 1] = bbox[:, 1]
    #                         bbox[:, 2] = bbox[:, 2] + shift1
    #                         bbox[:, 3] = bbox[:, 3] + shift1
    #                         tmp.append(bbox)
    #         bboxes[i] = torch.cat(tmp, dim=0)
    #     return bboxes                
    
    # def predict_bbox(self,
    #                  x: Tuple[Tensor],
    #                  batch_img_metas: List[dict],
    #                  rpn_results_list: InstanceList,
    #                  rcnn_test_cfg: ConfigType,
    #                  rescale: bool = False,
    #                  List_bbox=None, cls_targets=None) -> InstanceList:
    #     """Perform forward propagation of the bbox head and predict detection
    #     results on the features of the upstream network.

    #     Args:
    #         x (tuple[Tensor]): Feature maps of all scale level.
    #         batch_img_metas (list[dict]): List of image information.
    #         rpn_results_list (list[:obj:`InstanceData`]): List of region
    #             proposals.
    #         rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
    #         rescale (bool): If True, return boxes in original image space.
    #             Defaults to False.

    #     Returns:
    #         list[:obj:`InstanceData`]: Detection results of each image
    #         after the post process.
    #         Each item usually contains following keys.

    #             - scores (Tensor): Classification scores, has a shape
    #               (num_instance, )
    #             - labels (Tensor): Labels of bboxes, has a shape
    #               (num_instances, ).
    #             - bboxes (Tensor): Has a shape (num_instances, 4),
    #               the last dimension 4 arrange as (x1, y1, x2, y2).
    #     """
    #     proposals = [res.priors for res in rpn_results_list]
    #     rois = bbox2roi(proposals)

    #     if List_bbox != None:
    #         print("=======================")
    #         rois = bbox2roi(List_bbox)
    #         proposals = List_bbox


    #     if rois.shape[0] == 0:
    #         return empty_instances(
    #             batch_img_metas,
    #             rois.device,
    #             task_type='bbox',
    #             box_type=self.bbox_head.predict_box_type,
    #             num_classes=self.bbox_head.num_classes,
    #             score_per_cls=rcnn_test_cfg is None)

    #     # print(proposals)

        

    #     bbox_results = self._bbox_forward(x, rois)
        
    #     bbox_feats = self.bbox_roi_extractor(
    #             x[:4],rois
    #         )
    #     if self.with_shared_head:
    #         bbox_feats = self.shared_head(bbox_feats)
    #     cls_scores, bbox_preds = self.bbox_head(bbox_feats)
        
    #     try:
    #         cls_scores = cls_scores.reshape(8, 512, -1)
    #         cls_targets = cls_targets.reshape(8, -1)
    #         for i in range(8):
    #             print(f"cls_scores {i}", cls_scores[i].shape, cls_scores[i].softmax(dim=-1).argmax(dim=-1))
    #             print(f"cls_target {i}", cls_targets[i].shape, cls_targets[i])
    #         cls_scores = cls_scores.reshape(4096, -1)
    #         cls_targets = cls_targets.reshape(4096, -1)
    #     except:
    #         print("error", cls_scores.shape, cls_targets.shape)
    #     # split batch bbox prediction back to each image
    #     # cls_scores = bbox_results['cls_score']
    #     # bbox_preds = bbox_results['bbox_pred']
    #     num_proposals_per_img = tuple(len(p) for p in proposals)
    #     rois = rois.split(num_proposals_per_img, 0)
    #     cls_scores = cls_scores.split(num_proposals_per_img, 0)
    #     # some detector with_reg is False, bbox_preds will be None
    #     if bbox_preds is not None:
    #         # TODO move this to a sabl_roi_head
    #         # the bbox prediction of some detectors like SABL is not Tensor
    #         if isinstance(bbox_preds, torch.Tensor):
    #             bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
    #         else:
    #             bbox_preds = self.bbox_head.bbox_pred_split(
    #                 bbox_preds, num_proposals_per_img)
    #     else:
    #         bbox_preds = (None, ) * len(proposals)

    #     result_list = self.bbox_head.predict_by_feat(
    #         rois=rois,
    #         cls_scores=cls_scores,
    #         bbox_preds=bbox_preds,
    #         batch_img_metas=batch_img_metas,
    #         rcnn_test_cfg=rcnn_test_cfg,
    #         rescale=rescale)
    #     return result_list


        
@MODELS.register_module()
class StandardPrototypeReplayHead(StandardRoIReplayHead):
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None,
                 task_id = 1,
                 task_split = [0,10,20]) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg)
        self.replay = False
        self.task_split = task_split
        self.task_id = task_id
        device = next(self.parameters()).device
        if previous_path != None and osp.exists(previous_path):
            assert task_id != 1
            self.replay = True
            print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
            self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"), map_location=device)
            previous_cls = range(task_split[0], task_split[task_id-1])
            tmp = []
            for i in previous_cls:
                cls_mask = self.cls_targets == i
                cls_bbox_feats = torch.mean(self.bbox_featss[cls_mask], dim=0, keepdim=True)
                tmp.append(cls_bbox_feats)
            self.bbox_featss = torch.cat(tmp, dim=0)
            
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples, replay=False)
        device = next(self.parameters()).device
        if self.replay:
            # do sampling
            # mask = torch.randperm(self.bbox_featss.shape[0])[:64].to(self.bbox_featss.device)
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss
            bbox_featss = self.bbox_featss
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = bbox_featss.to(device), cls_targets.to(device), cls_weights.to(device), bbox_targets.to(device), bbox_weights.to(device), roiss.to(device)
            
            sampling_results = [cls_targets, cls_weights, bbox_targets, bbox_weights]
            replay_results = self.replay_loss(bbox_featss, None, None)
            # print("before update", losses)
            losses.update(replay_results['replay_loss'])
            # print("replay loss", replay_results['replay_loss'])
            # print("after update", losses)
        return losses
    
    
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        # bbox_feats = torch.rot90(bbox_feats, torch.randint(0, 4, (1,)).item(), dims=(-2,-1))
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        # teacher model
        # teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        # print(torch.isfinite(bbox_feats).all())
        # print(torch.isfinite(cls_score).all())
        # print(bbox_feats.shape, cls_score.shape, torch.argmax(cls_score, dim=-1), torch.argmax(teacher_cls_score, dim=-1))
        
        pre_idx = self.task_split[self.task_id]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)

        # print(cls_score.shape)
        
        losses["replay_loss_cls"] = F.cross_entropy(cls_score.softmax(dim=-1), torch.Tensor([i for i in range(cls_score.shape[0])]).long().to(cls_score.device))
        # losses["replay_loss_reg"] = F.mse_loss(bbox_pred, teacher_bbox_pred)

        # bbox_loss_and_target = self.bbox_head.replay_loss(
        #     cls_score=bbox_results['cls_score'],
        #     bbox_pred=bbox_results['bbox_pred'],
        #     rois=rois,
        #     sampling_results=sampling_results,
        #     rcnn_train_cfg=self.train_cfg)

        bbox_results.update(replay_loss=losses)
        return bbox_results
    
    

        
        
@MODELS.register_module()
class StandardCoresetReplayHead(StandardRoIReplayHead):
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None,
                 task_id = 1,
                 task_split = [0,10,20]) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg)
        self.replay = False
        self.task_id = task_id
        self.task_split = task_split
        if previous_path != None and osp.exists(previous_path):
            assert task_id != 1
            self.replay = True
            print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
            self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"))
            previous_cls = range(task_split[0], task_split[task_id-1])
            tmp = []
            for i in previous_cls:
                cls_mask = self.cls_targets == i
                cls_bbox_feats = self.bbox_featss[cls_mask]
                tmp.append(cls_bbox_feats)
            self.bbox_featss = torch.cat(tmp, dim=0)
            
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples)
        device = next(self.parameters()).device
        if self.replay:
            # do sampling
            mask = torch.randperm(self.bbox_featss.shape[0])[:64].to(self.bbox_featss.device)
            bbox_featss = self.bbox_featss[mask]
            bbox_featss = bbox_featss.to(device)
            sampling_results = None
            replay_results = self.replay_loss(bbox_featss, sampling_results, None)
            # print("before update", losses)
            losses.update(replay_results['replay_loss'])
            # print("replay loss", replay_results['replay_loss'])
            # print("after update", losses)
        return losses
    
    
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        # teacher model
        teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        pre_idx = self.task_split[self.task_id-1]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)
        teacher_cls_score = torch.cat([teacher_cls_score[:, :pre_idx], teacher_cls_score[:,-1:]], dim=-1)        
        
        losses["replay_loss_cls"] = F.mse_loss(cls_score, teacher_cls_score)
        # losses["replay_loss_reg"] = F.mse_loss(bbox_pred, teacher_bbox_pred)

        # bbox_loss_and_target = self.bbox_head.replay_loss(
        #     cls_score=bbox_results['cls_score'],
        #     bbox_pred=bbox_results['bbox_pred'],
        #     rois=rois,
        #     sampling_results=sampling_results,
        #     rcnn_train_cfg=self.train_cfg)

        bbox_results.update(replay_loss=losses)
        return bbox_results


        
@MODELS.register_module()
class StandardGMReplayHead(StandardRoIReplayHead):
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None,
                 task_id = 1,
                 task_split = [0,10,20],
                 class_wise = True) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg)
        self.replay = False
        self.class_wise = class_wise
        self.previous_path = previous_path
        self.task_id = task_id
        self.task_split = task_split
    
    def get_gms(self):
        previous_path = self.previous_path
        task_id = self.task_id
        task_split = self.task_split
        device = next(self.parameters()).device
        with torch.no_grad():
            if previous_path != None and osp.exists(previous_path):
                assert task_id != 1
                self.replay = True
                print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
                self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"), map_location=device)
                previous_cls = range(task_split[0], task_split[task_id-1])
                gms = []
                bs = self.bbox_featss.shape[0]
                if self.class_wise:
                    for i in previous_cls:
                        print(f"Processing {i}th GMs")
                        cls_mask = self.cls_targets == i
                        cls_bbox_feats = self.bbox_featss[cls_mask].reshape(-1, 256*7*7)
                        epsilon = 1e-3
                        mu = torch.mean(cls_bbox_feats, dim=0)
                        sigma = torch.cov(cls_bbox_feats.T)
                        
                        U,S,V = torch.svd(sigma)
                        rank = adaptive_threshold(S) == False
                        cov_factor = torch.mm(U[:, rank], torch.diag(S[rank]).sqrt())
                        cov_diag = torch.zeros(cov_factor.size(0)).to(cov_factor.device) + 1e-6
                        
                        mvn = tdist.LowRankMultivariateNormal(mu, cov_factor=cov_factor, cov_diag = cov_diag)
                        
                        # sigma += epsilon * torch.eye(sigma.size(0)).to(sigma.device)
                        # mvn = tdist.MultivariateNormal(mu, covariance_matrix=sigma)
                        gms.append(mvn)
                else:
                    cls_bbox_feats = self.bbox_featss.reshape(-1, 256*7*7)
                    epsilon = 1e-3
                    mu = torch.mean(cls_bbox_feats, dim=0)
                    sigma = torch.cov(cls_bbox_feats.T)
                    
                    U,S,V = torch.svd(sigma)
                    rank = adaptive_threshold(S)
                    cov_factor = torch.mm(U[:, rank], torch.diag(S[rank]).sqrt())
                    cov_diag = torch.zeros(cov_factor.size(0)).to(cov_factor.device) + 1e-6
                    
                    mvn = tdist.LowRankMultivariateNormal(mu, cov_factor=cov_factor, cov_diag = cov_diag)
                    gms.append(mvn)
                self.gms = gms
                print("Done getting gms")
                torch.cuda.empty_cache()
                
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples)
        device = next(self.parameters()).device
        if self.replay:
            # do sampling
            mask = torch.randperm(self.cls_targets.shape[0])[:64].to(self.cls_targets.device)
            cls_targets = self.cls_targets[mask]
            if self.class_wise:
                bbox_featss = []
                for i in cls_targets:
                    bbox_featss.append(self.gms[i].sample_n(1))
                bbox_featss = torch.cat(bbox_featss, dim=0)
            else:
                bbox_featss = self.gms[0].sample_n(64)
            bbox_featss = bbox_featss.to(device).detach()
            bbox_featss = bbox_featss.reshape(-1, 256, 7, 7)
          
            replay_results = self.replay_loss(bbox_featss, None, None)
            # print("before update", losses)
            losses.update(replay_results['replay_loss'])
            # print("replay loss", replay_results['replay_loss'])
            # print("after update", losses)
        return losses
    
    
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        # teacher model
        teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        pre_idx = self.task_split[self.task_id-1]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)
        teacher_cls_score = torch.cat([teacher_cls_score[:, :pre_idx], teacher_cls_score[:,-1:]], dim=-1)
        
        losses["replay_loss_cls"] = F.mse_loss(cls_score, teacher_cls_score)
        # losses["replay_loss_reg"] = F.mse_loss(bbox_pred, teacher_bbox_pred)

        # bbox_loss_and_target = self.bbox_head.replay_loss(
        #     cls_score=bbox_results['cls_score'],
        #     bbox_pred=bbox_results['bbox_pred'],
        #     rois=rois,
        #     sampling_results=sampling_results,
        #     rcnn_train_cfg=self.train_cfg)

        bbox_results.update(replay_loss=losses)
        return bbox_results
    
    


@MODELS.register_module()
class StandardPCAReplayHead(StandardRoIReplayHead):
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None,
                 task_id = 1,
                 task_split = [0,10,20],
                 class_wise = True) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg, previous_path)
        self.class_wise = class_wise
        self.previous_path = previous_path
        self.task_id = task_id
        self.task_split = task_split
        if self.task_id != 1:
            self.replay = True
    
    def get_gms(self):
        cls_targets = self.cls_targets
        basis = [] # 18x256
        projected_bbox = [] # mx7x7x18
        instance_list = [] # nxm
        task_split = self.task_split
        self.gms = []
        print(self.cls_targets.shape)
        print(cls_targets.unique())
        tmp = 0
        for i in range(self.task_id-1):
            cls_mask = torch.logical_and(cls_targets < task_split[i+1], cls_targets >= task_split[i])
            tmp += cls_mask.sum()
            print(tmp, cls_mask.sum())
            # cls_mask = cls_targets == i
            f = self.bbox_featss[cls_mask].permute(0,2,3,1).reshape(-1,256)

            covariance = f.T @ f
            # covariance = torch.cov(f.T)
            U,S,V = torch.svd(covariance)
            ind = adaptive_threshold(S) == False
            
            # U_basis = V[ind, :].t()
            U_basis = U[:, ind]
            
            if len(basis)!=0:
                for idx, ii in enumerate(U_basis.t()):
                    ii = ii.unsqueeze(0)
                    projected_U = ii @ basis[0] @ basis[0].t()
                    if 1 - (ii/ii.norm(dim=-1, keepdim=True))@(projected_U / projected_U.norm(dim=-1, keepdim=True)).t() < 0.1:
                        continue
                    else:
                        basis.append(ii.t())
                        basis = torch.cat(basis, dim=1)
                        basis = gram_schmidt(basis, j=basis.shape[1]-1)
                        basis = [basis]
            else:
                basis.append(U_basis) # 256 x ind
            
            pos = f @ basis[0] # n*7*7x256 @ 256 x ind = n*7*7xind
            # print("pos 1", pos.shape)
            pos = pos.reshape(-1, 7*7*basis[0].shape[1]) # nx7*7*ind
            # print("pos 2", pos.shape)
            covariance = pos.T @ pos # 7*7*ind x 7*7*ind
            # covariance = torch.cov(f.T)
            U,S,V = torch.svd(covariance)
            ind = adaptive_threshold(S) == False
            # U_basis = V[ind, :].t()
            U_basis = U[:, ind]
            
            if len(projected_bbox)!=0:
                for idx, ii in enumerate(U_basis.t()):
                    ii = ii.unsqueeze(0)
                    ch, bs = projected_bbox[0].shape
                    before = ch // 49
                    after = ii.shape[1] // 49 - before
                    # print(projected_bbox[0].t().reshape(bs, 49, before).shape, projected_bbox[0].shape)
                    # exit()
                    projected_bbox[0] = F.pad(projected_bbox[0].t().reshape(bs, 49, before), (0, after)).reshape(bs, -1).t()
                    # print(U_basis.shape, U.shape, ind.sum(), ii.shape, projected_bbox[0].shape, len(projected_bbox))
                    projected_U = ii @ projected_bbox[0] @ projected_bbox[0].t()
                    if 1 - (ii/ii.norm(dim=-1, keepdim=True))@(projected_U / projected_U.norm(dim=-1, keepdim=True)).t() < 0.1:
                        continue
                    else:
                        projected_bbox.append(ii.t())
                        projected_bbox = torch.cat(projected_bbox, dim=1)
                        projected_bbox = gram_schmidt(projected_bbox, j=projected_bbox.shape[1]-1)
                        projected_bbox = [projected_bbox]
            else:
                projected_bbox.append(U_basis)
            
            instance_list.append(pos @ projected_bbox[0])
            
        self.instance_list = torch.cat([F.pad(p, (0, projected_bbox[0].shape[1] - p.shape[1])) for p in instance_list], dim=0)
        self.projected_bbox = projected_bbox[0].t()
        # torch.cat([F.pad(p, (0, basis[0].shape[1] - p.shape[1])) for p in projected_bbox], dim=0).reshape(-1, 7,7,basis[0].shape[1])
        self.basis = basis[0]                
                
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples)
        device = next(self.parameters()).device
        if self.replay:
            # do sampling
            mask = torch.randperm(self.instance_list.shape[0])[:64].to(self.instance_list.device)
            # print(self.instance_list.shape, self.cls_targets.shape, mask.min(), mask.max())
            bbox_featss = (self.instance_list[mask] @ self.projected_bbox).reshape(64*7*7, -1) @ self.basis.T
            bbox_featss = bbox_featss.reshape(-1, 7,7,256).permute(0,3,1,2)
            bbox_featss = bbox_featss.to(device).detach()
            bbox_featss = bbox_featss.reshape(-1, 256, 7, 7)
            replay_results = self.replay_loss(bbox_featss, None, None)
            # print("before update", losses)
            losses.update(replay_results['replay_loss'])
            # print("replay loss", replay_results['replay_loss'])
            # print("after update", losses)
        return losses
    
    
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        # teacher model
        teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        pre_idx = self.task_split[self.task_id-1]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)
        teacher_cls_score = torch.cat([teacher_cls_score[:, :pre_idx], teacher_cls_score[:,-1:]], dim=-1)
        
        losses["replay_loss_cls"] = F.mse_loss(cls_score, teacher_cls_score)
        # losses["replay_loss_reg"] = F.mse_loss(bbox_pred, teacher_bbox_pred)

        # bbox_loss_and_target = self.bbox_head.replay_loss(
        #     cls_score=bbox_results['cls_score'],
        #     bbox_pred=bbox_results['bbox_pred'],
        #     rois=rois,
        #     sampling_results=sampling_results,
        #     rcnn_train_cfg=self.train_cfg)

        bbox_results.update(replay_loss=losses)
        return bbox_results


@MODELS.register_module()
class StandardPCACWReplayHead(StandardRoIReplayHead):
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None,
                 task_id = 1,
                 task_split = [0,10,20],
                 class_wise = True) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg, previous_path)
        self.class_wise = class_wise
        self.previous_path = previous_path
        self.task_id = task_id
        self.task_split = task_split
        if self.task_id != 1:
            self.replay = True
    
    def get_gms(self):
        cls_targets = self.cls_targets
        basis = []
        projected_bbox = []
        instance_list = []
        task_split = self.task_split
        for i in range(task_split[self.task_id-1]):
            # cls_mask = torch.logical_and(cls_targets < task_split[i+1], cls_targets >= task_split[i])
            cls_mask = cls_targets == i
            f = self.bbox_featss[cls_mask].permute(0,2,3,1).reshape(-1,256)

            covariance = f.T @ f
            # covariance = torch.cov(f.T)
            U,S,V = torch.svd(covariance)
            ind = adaptive_threshold(S) == False
            
            # U_basis = V[ind, :].t()
            U_basis = U[:, ind]
            
            if len(basis)!=0:
                for idx, ii in enumerate(U_basis.t()):
                    ii = ii.unsqueeze(0)
                    projected_U = ii @ basis[0] @ basis[0].t()
                    if 1 - (ii/ii.norm(dim=-1, keepdim=True))@(projected_U / projected_U.norm(dim=-1, keepdim=True)).t() < 0.1:
                        continue
                    else:
                        basis.append(ii.t())
                        basis = torch.cat(basis, dim=1)
                        basis = gram_schmidt(basis, j=basis.shape[1]-1)
                        basis = [basis]
            else:
                basis.append(U_basis) # 256 x ind
            
            pos = f @ basis[0] # n*7*7x256 @ 256 x ind = n*7*7xind
            # print("pos 1", pos.shape)
            pos = pos.reshape(-1, 7*7*basis[0].shape[1]) # nx7*7*ind
            # print("pos 2", pos.shape)
            covariance = pos.T @ pos # 7*7*ind x 7*7*ind
            # covariance = torch.cov(f.T)
            U,S,V = torch.svd(covariance)
            ind = adaptive_threshold(S) == False
            # U_basis = V[ind, :].t()
            U_basis = U[:, ind]
            
            if len(projected_bbox)!=0:
                for idx, ii in enumerate(U_basis.t()):
                    ii = ii.unsqueeze(0)
                    ch, bs = projected_bbox[0].shape
                    before = ch // 49
                    after = ii.shape[1] // 49 - before
                    # print(projected_bbox[0].t().reshape(bs, 49, before).shape, projected_bbox[0].shape)
                    # exit()
                    projected_bbox[0] = F.pad(projected_bbox[0].t().reshape(bs, 49, before), (0, after)).reshape(bs, -1).t()
                    # print(U_basis.shape, U.shape, ind.sum(), ii.shape, projected_bbox[0].shape, len(projected_bbox))
                    projected_U = ii @ projected_bbox[0] @ projected_bbox[0].t()
                    if 1 - (ii/ii.norm(dim=-1, keepdim=True))@(projected_U / projected_U.norm(dim=-1, keepdim=True)).t() < 0.1:
                        continue
                    else:
                        projected_bbox.append(ii.t())
                        projected_bbox = torch.cat(projected_bbox, dim=1)
                        projected_bbox = gram_schmidt(projected_bbox, j=projected_bbox.shape[1]-1)
                        projected_bbox = [projected_bbox]
            else:
                projected_bbox.append(U_basis)
            
            instance_list.append(pos @ projected_bbox[0])
            
        self.instance_list = torch.cat([F.pad(p, (0, projected_bbox[0].shape[1] - p.shape[1])) for p in instance_list], dim=0)
        self.projected_bbox = projected_bbox[0].t()
        # torch.cat([F.pad(p, (0, basis[0].shape[1] - p.shape[1])) for p in projected_bbox], dim=0).reshape(-1, 7,7,basis[0].shape[1])
        self.basis = basis[0]                
                
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples)
        device = next(self.parameters()).device
        if self.replay:
            # do sampling
            mask = torch.randperm(self.cls_targets.shape[0])[:64].to(self.cls_targets.device)
            # print(self.instance_list.shape, self.projected_bbox.T.shape, self.basis.T.shape)
            bbox_featss = (self.instance_list[mask] @ self.projected_bbox).reshape(64*7*7, -1) @ self.basis.T
            bbox_featss = bbox_featss.reshape(-1, 7,7,256).permute(0,3,1,2)
            bbox_featss = bbox_featss.to(device).detach()
            bbox_featss = bbox_featss.reshape(-1, 256, 7, 7)
          
            replay_results = self.replay_loss(bbox_featss, None, None)
            # print("before update", losses)
            losses.update(replay_results['replay_loss'])
            # print("replay loss", replay_results['replay_loss'])
            # print("after update", losses)
        return losses
    
    
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        # teacher model
        teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        pre_idx = self.task_split[self.task_id-1]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)
        teacher_cls_score = torch.cat([teacher_cls_score[:, :pre_idx], teacher_cls_score[:,-1:]], dim=-1)
        
        losses["replay_loss_cls"] = F.mse_loss(cls_score, teacher_cls_score)
        # losses["replay_loss_reg"] = F.mse_loss(bbox_pred, teacher_bbox_pred)

        # bbox_loss_and_target = self.bbox_head.replay_loss(
        #     cls_score=bbox_results['cls_score'],
        #     bbox_pred=bbox_results['bbox_pred'],
        #     rois=rois,
        #     sampling_results=sampling_results,
        #     rcnn_train_cfg=self.train_cfg)

        bbox_results.update(replay_loss=losses)
        return bbox_results


@MODELS.register_module()
class StandardBridgePrototypeReplayHead(StandardRoIReplayHead):
    '''
    真正的Prototype Replay模型
    '''
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None,
                 task_id = 1,
                 task_split = [0,10,20]) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg)
        self.replay = False
        self.task_id = task_id
        self.task_split = task_split
        if previous_path != None and osp.exists(previous_path):
            assert task_id != 1
            self.replay = True
            print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
            self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"))
            previous_cls = range(task_split[0], task_split[task_id-1])
            tmp = []
            for i in previous_cls:
                cls_mask = self.cls_targets == i
                cls_bbox_feats = torch.mean(self.bbox_featss[cls_mask], dim=0, keepdim=True)
                tmp.append(cls_bbox_feats)
            self.bbox_featss = torch.cat(tmp, dim=0)
            
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples)
        device = next(self.parameters()).device
        if self.replay:
            # do sampling
            mask = torch.randperm(self.bbox_featss.shape[0])[:64].to(self.bbox_featss.device)
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = self.bbox_featss[mask], self.cls_targets[mask], self.cls_weights[mask], self.bbox_targets[mask], self.bbox_weights[mask], self.roiss[mask]
            bbox_featss = self.bbox_featss
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = bbox_featss.to(device), cls_targets.to(device), cls_weights.to(device), bbox_targets.to(device), bbox_weights.to(device), roiss.to(device)
            
            sampling_results = [cls_targets, cls_weights, bbox_targets, bbox_weights]
            replay_results = self.replay_loss(bbox_featss, sampling_results, roiss)
            # print("before update", losses)
            losses.update(replay_results['replay_loss'])
            # print("replay loss", replay_results['replay_loss'])
            # print("after update", losses)
        return losses
    
    
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        bbox_feats = torch.rot90(bbox_feats, torch.randint(0, 4, (1,)).item(), dims=(-2,-1))
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        # for n, i in self.teacher_model.bbox_head.named_parameters():
        #     if 'shared' in n:
        #         print("teacher", n, i.flatten()[0], i.requires_grad)
        #         break
        # try:
        #     for n, i in self.bbox_head.named_parameters():
        #         if 'shared' in n:
        #             print("student", n, i.flatten()[0], i.grad.flatten()[0], i.requires_grad)
        #             break
        # except:
        #     for n, i in self.bbox_head.named_parameters():
        #         if 'shared' in n:
        #             print("student", n, i.flatten()[0], None, i.requires_grad)
        #             break
        
        # teacher model
        teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        pre_idx = self.task_split[self.task_id-1]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)
        teacher_cls_score = torch.cat([teacher_cls_score[:, :pre_idx], teacher_cls_score[:,-1:]], dim=-1)
        # print(cls_score.shape)
        # losses["replay_loss_cls"] = F.mse_loss(cls_score, teacher_cls_score)
        # print(teacher_cls_score.softmax(dim=-1).argmax(dim=-1))
        # print(cls_score.softmax(dim=-1).argmax(dim=-1))
        # exit()
        # print(cls_score.shape)
        losses["replay_loss_cls"] = F.cross_entropy(cls_score.softmax(dim=-1), torch.Tensor([i for i in range(cls_score.shape[0])]).long().to(cls_score.device))


        bbox_results.update(replay_loss=losses)
        return bbox_results


def simple_attention(weight, v):
    # q = q/q.norm(dim=-1, keepdim=True)
    # k = k/k.norm(dim=-1, keepdim=True)
    # print(weight.shape, v.shape)
    # exit()
    attn_map = torch.softmax(10 * weight, dim=-1)
    bs, a,b,c = v.shape
    v = v.reshape(bs, -1)
    return (attn_map @ v).reshape(1, a,b,c)
    


@MODELS.register_module()
class StandardMultiPrototypeReplayHead(StandardRoIReplayHead):
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None,
                 task_id = 1,
                 task_split = [0,10,20]) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg)
        self.replay = False
        self.task_split = task_split
        self.task_id = task_id
        self.max_proto = 3
        device = next(self.parameters()).device
        with torch.no_grad():
            if previous_path != None and osp.exists(previous_path):
                assert task_id != 1
                self.replay = True
                print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
                self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"), map_location=device)
                previous_cls = range(task_split[0], task_split[task_id-1])
                tmp = []
                if osp.exists(osp.join(previous_path, "mask.pth")):
                    save_idx = torch.load(osp.join(previous_path, "mask.pth"), map_location='cpu')
                else:
                    save_idx = []
                for i in previous_cls:
                    cls_mask = self.cls_targets == i
                    cls_bbox_feats = torch.mean(self.bbox_featss[cls_mask], dim=0, keepdim=True)
                    tmp.append(cls_bbox_feats)
                    relu_mask = self.bbox_head.get_relu(cls_bbox_feats) == 0
        
                    c_c = self.bbox_head.get_relu(self.bbox_featss[cls_mask])
                    
                    cc_c = c_c[-1]!=0
                    
                    for iiii, c_ in enumerate(c_c):
                        c_c[iiii] = c_ / c_.norm(dim=-1, keepdim=True)
                    cca, ccb = c_c[0] @ c_c[0].t(), c_c[1] @ c_c[1].t()
                    corss_compare = (cca + ccb)/2 # bs, bs
                    c_c_v, _ = (corss_compare > 0.7).long().sum(dim=-1).sort(dim=-1, descending=True)
                    # print(c_c_v.shape)
                    c_c_threash = c_c_v[-c_c_v.shape[0]//2]
                    used_idx = (corss_compare > 0.7).long().sum(dim=-1) <= c_c_threash
                    distances_mask = corss_compare > 0.7
                    
                    relu_sum = cc_c * relu_mask # bs * 1024
                    relu_sum = relu_sum.to(torch.long)
                    _, idx = relu_sum.sum(dim=-1).sort(dim=0, descending=True) # bs
                    
                    # c_c = cc_c.to(torch.long)    
                    # corss_compare = c_c @ c_c.t() # bs, bs
              
                    
                    proto_count = 0
                    if i < len(save_idx):
                        tmp_mask = save_idx[i]
                    else:
                        tmp_mask = []
                    for id_ in idx:
                        if proto_count >= self.max_proto - 1:
                            break
                        
                        if proto_count < len(tmp_mask):
                            m = tmp_mask[proto_count].to(device)
                            print(f"{id_} ++++++++++")
                        else:
                            if used_idx[id_]:
                                continue
                            # _, compare_idx = corss_compare[id_].sort(dim=0, descending=True)
                            # one_third = compare_idx.shape[0] // self.max_proto
                            # one_third = one_third if one_third != 0 else one_third + 1
                            # m = torch.zeros_like(compare_idx, dtype=torch.bool)
                            # m.scatter_(0, compare_idx[:one_third], True)
                            m = distances_mask[id_]
                            tmp_mask.append(m)
                            print(f"{id_} ///////")
                        # m = corss_compare[i] > (relu_sum.sum(dim=-1)[i] * 0.7)
                        # print(m.sum())
                        print(used_idx.shape, m.shape)
                        used_idx = torch.logical_or(used_idx, m)
                        # prototype = torch.mean(self.bbox_featss[cls_mask][m], dim=0, keepdim=True)
                        weight = corss_compare[id_].unsqueeze(0)
                        prototype = simple_attention(weight, self.bbox_featss[cls_mask])   
                        
                        # aaa = self.bbox_head.get_relu(prototype) != 0
                        # print((aaa * relu_mask).sum(), (relu_mask!=0).sum())
                        # relu_mask = torch.logical_and(relu_mask, aaa!=False)
                        
                        tmp.append(prototype)
                        proto_count += 1
                    if i >= len(save_idx):
                        save_idx.append(tmp_mask)
                    # exit()
                self.bbox_featss = torch.cat(tmp, dim=0)
                work_dir = get_work_dir(previous_path)
                torch.save(save_idx, osp.join(work_dir, "mask.pth"))
        
        # roi_stds = torch.zeros(2, 256, 7, 7)
        # roi_stds = torch.normal(roi_stds)
        # self.roi_stds = torch.nn.ParameterList([torch.nn.Parameter(roi_stds) for i in range(task_split[-1])])
        # for i in range(len(self.roi_stds)):
        #     self.roi_stds[i].requires_grad_(False)
            
        torch.cuda.empty_cache()
        
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples, replay=False)
        device = next(self.parameters()).device
        if self.replay:
            # do sampling
            # mask = torch.randperm(self.bbox_featss.shape[0])[:64].to(self.bbox_featss.device)
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss
            bbox_featss = self.bbox_featss
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = bbox_featss.to(device), cls_targets.to(device), cls_weights.to(device), bbox_targets.to(device), bbox_weights.to(device), roiss.to(device)
            
            sampling_results = [cls_targets, cls_weights, bbox_targets, bbox_weights]
            replay_results = self.replay_loss(bbox_featss, None, None)
            # print("before update", losses)
            losses.update(replay_results['replay_loss'])
            # print("replay loss", replay_results['replay_loss'])
            # print("after update", losses)
        return losses
    
    def re_loss(self) -> dict:
        losses = {}
        device = next(self.parameters()).device
        bbox_feats = self.bbox_featss
        bbox_feats = bbox_feats.to(device)
        
        bbox_featss = []; labels = []
        for i in range(4096//bbox_feats.shape[0]):
            std = torch.cat([self.roi_stds[i][:1] for i in range(bbox_feats.shape[0])])
            drift = torch.cat([self.roi_stds[i][1:] for i in range(bbox_feats.shape[0])])
            bbox_feats = bbox_feats + drift + std * torch.randn_like(std)
            label = torch.Tensor([i for i in range(bbox_feats.shape[0])]).long().to(bbox_feats.device)
            bbox_featss.append(bbox_feats)
            labels.append(label)
        bbox_feats = torch.cat(bbox_featss, dim=0)
        labels = torch.cat(labels, dim=0)
        
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, _ = self.bbox_head(bbox_feats)

        losses = dict()
        
        pre_idx = self.task_split[self.task_id]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)
        
        losses["replay_loss_cls"] = F.cross_entropy(cls_score.softmax(dim=-1), labels)
        return losses
    
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        # bbox_feats = torch.rot90(bbox_feats, torch.randint(0, 4, (1,)).item(), dims=(-2,-1))
        # std = torch.cat([self.roi_stds[i][:1] for i in range(bbox_feats.shape[0])])
        # drift = torch.cat([self.roi_stds[i][1:] for i in range(bbox_feats.shape[0])])
        bbox_feats = bbox_feats
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        # teacher model
        # teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        # print(torch.isfinite(bbox_feats).all())
        # print(torch.isfinite(cls_score).all())
        # print(bbox_feats.shape, cls_score.shape, torch.argmax(cls_score, dim=-1), torch.argmax(teacher_cls_score, dim=-1))
        
        pre_idx = self.task_split[self.task_id]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)

        # print(cls_score.shape)
        assert cls_score.shape[0] % self.max_proto == 0
        losses["replay_loss_cls"] = F.cross_entropy(cls_score.softmax(dim=-1), torch.Tensor([[i]*self.max_proto for i in range(cls_score.shape[0]//self.max_proto)]).flatten().long().to(cls_score.device))
        # losses["replay_loss_reg"] = F.mse_loss(bbox_pred, teacher_bbox_pred)

        # bbox_loss_and_target = self.bbox_head.replay_loss(
        #     cls_score=bbox_results['cls_score'],
        #     bbox_pred=bbox_results['bbox_pred'],
        #     rois=rois,
        #     sampling_results=sampling_results,
        #     rcnn_train_cfg=self.train_cfg)

        bbox_results.update(replay_loss=losses)
        return bbox_results
    
    
    
@MODELS.register_module()
class StandardMultiPrototypeDistillHead(StandardRoIReplayHead):
    def __init__(self,
                 bbox_roi_extractor: OptMultiConfig = None,
                 bbox_head: OptMultiConfig = None,
                 mask_roi_extractor: OptMultiConfig = None,
                 mask_head: OptMultiConfig = None,
                 shared_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 previous_path = None,
                 task_id = 1,
                 task_split = [0,10,20]) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg)
        self.replay = False
        self.task_split = task_split
        self.task_id = task_id
        self.max_proto = 10
        device = next(self.parameters()).device
        with torch.no_grad():
            if previous_path != None and osp.exists(previous_path):
                assert task_id != 1
                self.replay = True
                print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
                self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"), map_location=device)
                previous_cls = range(task_split[0], task_split[task_id-1])
                tmp = []
                if osp.exists(osp.join(previous_path, "mask.pth")):
                    save_idx = torch.load(osp.join(previous_path, "mask.pth"), map_location='cpu')
                else:
                    save_idx = []
                for i in previous_cls:
                    cls_mask = self.cls_targets == i
                    cls_bbox_feats = torch.mean(self.bbox_featss[cls_mask], dim=0, keepdim=True)
                    tmp.append(cls_bbox_feats)
                    relu_mask = self.bbox_head.get_relu(cls_bbox_feats) == 0
        
                    c_c = self.bbox_head.get_relu(self.bbox_featss[cls_mask])
                    
                    # cc_c = c_c[-1]!=0
                    
                    for iiii, c_ in enumerate(c_c):
                        c_c[iiii] = c_ / c_.norm(dim=-1, keepdim=True)
                    cca, ccb = c_c[0] @ c_c[0].t(), c_c[1] @ c_c[1].t()
                    corss_compare = (cca + ccb)/2 # bs, bs
                    c_c_v, _ = (corss_compare > 0.7).long().sum(dim=-1).sort(dim=-1, descending=True)
                    print(c_c_v.shape)
                    c_c_threash = c_c_v[-c_c_v.shape[0]//3]
                    used_idx = (corss_compare > 0.7).long().sum(dim=-1) <= c_c_threash
                    distances_mask = corss_compare > 0.7 # n * n
                    tmp_feats = (distances_mask / distances_mask.sum(dim=-1, keepdim=True)) @ self.bbox_featss[cls_mask].reshape(-1, 256*7*7)
                    
                    cc_c = self.bbox_head.get_relu(tmp_feats.reshape(-1, 256, 7, 7))[-1] != 0
                    
                    
                    relu_sum = cc_c * relu_mask # bs * 1024
                    relu_sum = relu_sum.to(torch.long)
                    _, idx = relu_sum.sum(dim=-1).sort(dim=0, descending=True) # bs
                    
                    # c_c = cc_c.to(torch.long)    
                    # corss_compare = c_c @ c_c.t() # bs, bs
              
                    
                    proto_count = 0
                    if i < len(save_idx):
                        tmp_mask = save_idx[i]
                    else:
                        tmp_mask = []
                    for id_ in idx:
                        if proto_count >= self.max_proto - 1:
                            break
                        
                        if proto_count < len(tmp_mask):
                            m = tmp_mask[proto_count].to(device)
                            print(f"{id_} ++++++++++")
                        else:
                            if used_idx[id_]:
                                continue
                            # _, compare_idx = corss_compare[id_].sort(dim=0, descending=True)
                            # one_third = compare_idx.shape[0] // self.max_proto
                            # one_third = one_third if one_third != 0 else one_third + 1
                            # m = torch.zeros_like(compare_idx, dtype=torch.bool)
                            # m.scatter_(0, compare_idx[:one_third], True)
                            m = distances_mask[id_]
                            tmp_mask.append(m)
                            print(f"{id_} ///////")
                        # m = corss_compare[i] > (relu_sum.sum(dim=-1)[i] * 0.7)
                        # print(m.sum())
                        print(used_idx.shape, m.shape)
                        used_idx = torch.logical_or(used_idx, m)
                        prototype = torch.mean(self.bbox_featss[cls_mask][m], dim=0, keepdim=True)
                        
                        # aaa = self.bbox_head.get_relu(prototype) != 0
                        # print((aaa * relu_mask).sum(), (relu_mask!=0).sum())
                        # relu_mask = torch.logical_and(relu_mask, aaa!=False)
                        
                        tmp.append(prototype)
                        proto_count += 1
                    if i >= len(save_idx):
                        save_idx.append(tmp_mask)
                    # exit()
                self.bbox_featss = torch.cat(tmp, dim=0)
                work_dir = get_work_dir(previous_path)
                torch.save(save_idx, osp.join(work_dir, "mask.pth"))
        
        # roi_stds = torch.zeros(2, 256, 7, 7)
        # roi_stds = torch.normal(roi_stds)
        # self.roi_stds = torch.nn.ParameterList([torch.nn.Parameter(roi_stds) for i in range(task_split[-1])])
        # for i in range(len(self.roi_stds)):
        #     self.roi_stds[i].requires_grad_(False)
            
        # torch.cuda.empty_cache()
        
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples, replay=False)
        device = next(self.parameters()).device
        if self.replay:
            # do sampling
            # mask = torch.randperm(self.bbox_featss.shape[0])[:64].to(self.bbox_featss.device)
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss
            bbox_featss = self.bbox_featss
            bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = bbox_featss.to(device), cls_targets.to(device), cls_weights.to(device), bbox_targets.to(device), bbox_weights.to(device), roiss.to(device)
            
            sampling_results = [cls_targets, cls_weights, bbox_targets, bbox_weights]
            replay_results = self.replay_loss(bbox_featss, None, None)
            # print("before update", losses)
            losses.update(replay_results['replay_loss'])
            # print("replay loss", replay_results['replay_loss'])
            # print("after update", losses)
        return losses
    
    def re_loss(self) -> dict:
        losses = {}
        device = next(self.parameters()).device
        bbox_feats = self.bbox_featss
        bbox_feats = bbox_feats.to(device)
        
        bbox_featss = []; labels = []
        for i in range(4096//bbox_feats.shape[0]):
            std = torch.cat([self.roi_stds[i][:1] for i in range(bbox_feats.shape[0])])
            drift = torch.cat([self.roi_stds[i][1:] for i in range(bbox_feats.shape[0])])
            bbox_feats = bbox_feats + drift + std * torch.randn_like(std)
            label = torch.Tensor([i for i in range(bbox_feats.shape[0])]).long().to(bbox_feats.device)
            bbox_featss.append(bbox_feats)
            labels.append(label)
        bbox_feats = torch.cat(bbox_featss, dim=0)
        labels = torch.cat(labels, dim=0)
        
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, _ = self.bbox_head(bbox_feats)

        losses = dict()
        
        pre_idx = self.task_split[self.task_id]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)
        
        losses["replay_loss_cls"] = F.cross_entropy(cls_score.softmax(dim=-1), labels)
        return losses
    
    # def replay_loss(self, bbox_feats: Tuple[Tensor],
    #               sampling_results: List[SamplingResult],
    #               rois) -> dict:
    #     """Perform forward propagation and loss calculation of the bbox head on
    #     the features of the upstream network.

    #     Args:
    #         x (tuple[Tensor]): List of multi-level img features.
    #         sampling_results (list["obj:`SamplingResult`]): Sampling results.

    #     Returns:
    #         dict[str, Tensor]: Usually returns a dictionary with keys:

    #             - `cls_score` (Tensor): Classification scores.
    #             - `bbox_pred` (Tensor): Box energies / deltas.
    #             - `bbox_feats` (Tensor): Extract bbox RoI features.
    #             - `loss_bbox` (dict): A dictionary of bbox loss components.
    #     """
    #     bbox_feats = torch.rot90(bbox_feats, torch.randint(0, 4, (1,)).item(), dims=(-2,-1))
    #     if self.with_shared_head:
    #         bbox_feats = self.shared_head(bbox_feats)
    #     cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
    #     # teacher model
    #     teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

    #     bbox_results = dict(
    #         cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

    #     losses = dict()
        
    #     pre_idx = self.task_split[self.task_id-1]
    #     cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)
    #     teacher_cls_score = torch.cat([teacher_cls_score[:, :pre_idx], teacher_cls_score[:,-1:]], dim=-1)
    #     # print(cls_score.shape)
    #     losses["replay_loss_cls"] = F.mse_loss(cls_score, teacher_cls_score)

    #     bbox_results.update(replay_loss=losses)
    #     return bbox_results
    
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            sampling_results (list["obj:`SamplingResult`]): Sampling results.

        Returns:
            dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
                - `loss_bbox` (dict): A dictionary of bbox loss components.
        """
        # bbox_feats = torch.rot90(bbox_feats, torch.randint(0, 4, (1,)).item(), dims=(-2,-1))
        # std = torch.cat([self.roi_stds[i][:1] for i in range(bbox_feats.shape[0])])
        # drift = torch.cat([self.roi_stds[i][1:] for i in range(bbox_feats.shape[0])])
        bbox_feats = bbox_feats
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        
        # teacher model
        # teacher_cls_score, teacher_bbox_pred = self.teacher_model.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        # print(torch.isfinite(bbox_feats).all())
        # print(torch.isfinite(cls_score).all())
        # print(bbox_feats.shape, cls_score.shape, torch.argmax(cls_score, dim=-1), torch.argmax(teacher_cls_score, dim=-1))
        
        pre_idx = self.task_split[self.task_id]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)

        # print(cls_score.shape)
        assert cls_score.shape[0] % self.max_proto == 0
        losses["replay_loss_cls"] = F.cross_entropy(cls_score.softmax(dim=-1), torch.Tensor([[i]*self.max_proto for i in range(cls_score.shape[0]//self.max_proto)]).flatten().long().to(cls_score.device))
        # losses["replay_loss_reg"] = F.mse_loss(bbox_pred, teacher_bbox_pred)

        # bbox_loss_and_target = self.bbox_head.replay_loss(
        #     cls_score=bbox_results['cls_score'],
        #     bbox_pred=bbox_results['bbox_pred'],
        #     rois=rois,
        #     sampling_results=sampling_results,
        #     rcnn_train_cfg=self.train_cfg)

        bbox_results.update(replay_loss=losses)
        return bbox_results
    



def adaptive_threshold(svals: torch.Tensor, offset: float = 0):
    points: np.ndarray = svals.cpu().numpy()
    assert points.ndim == 1
    if len(points) >= 128:
        fil_points = scipy.ndimage.gaussian_filter1d(points, sigma=10)
        _delta = 1
        diff_o1 = fil_points[:-_delta] - fil_points[_delta:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        _drop_ratio = 0.03
        drop_num = int(len(points) * _drop_ratio / 2)
        assert len(points) - drop_num >= 10
        valid_o2 = diff_o2[drop_num:-drop_num]
        thres_val = points[np.argmax(valid_o2) + int((len(points) - len(valid_o2)) / 2)]
    else:
        diff_o1 = points[:-1] - points[1:]
        diff_o2 = diff_o1[:-1] - diff_o1[1:]
        thres_val = points[np.argmax(diff_o2) + int((len(points) - len(diff_o2)) / 2)]
    i_thres = np.arange(len(points))[points >= thres_val].max()
    # assert 0 <= offset < 1, offset
    # print(offset)
    if -1 <= offset <= 1:
        i_thres = min(i_thres + int(offset * (i_thres)), len(points) - 1)
        i_thres = max(0, i_thres)
    else:
        i_thres = max(min(i_thres + int(offset), len(points) - 1), 0)

    zero_idx = np.zeros(len(points), dtype=np.int64)
    zero_idx[i_thres:] = 1
    zero_idx = torch.as_tensor(torch.from_numpy(zero_idx), dtype=torch.bool, device=svals.device)
    return zero_idx

def gram_schmidt(A,j):
    # 获取矩阵A的列数
    m, n = A.size()
    # 初始化正交矩阵Q
    Q = torch.zeros(m, n)
    Q = Q.to(A.device)
    Q[:, :j] = A[:, :j]
    # 提取第j列向量
    v = A[:, j]
    
    # 减去v在之前所有正交向量上的投影
    for i in range(j):
        u = Q[:, i]
        # 计算投影
        proj = torch.sum(u * v) / torch.sum(u * u) * u
        # 减去投影
        v = v - proj
    
    # 单位化
    Q[:, j] = v / torch.norm(v)
    
    return Q

def get_norm(a):
    return (a / a.norm(dim=-1, keepdim=True)).unsqueeze(1)

def get_cos_sim(ii, projected_U):
    return (ii/ii.norm(dim=-1, keepdim=True))@(projected_U / projected_U.norm(dim=-1, keepdim=True)).t()



def get_work_dir(previous_path):
    # 用这个根据给出的之前的path去推断当前的work_dir
    splited_path = previous_path.split("_")
    task_id = int(splited_path[-1])
    splited_path[-1] = str(task_id + 1)
    return "_".join(splited_path)