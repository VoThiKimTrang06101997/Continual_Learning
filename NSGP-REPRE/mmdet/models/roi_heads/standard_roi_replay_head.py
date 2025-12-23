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
import os

import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np
import scipy
import scipy.ndimage

import random
# from torch_cluster import dbscan
import copy

from mmdet.evaluation.functional.bbox_overlaps import bbox_overlaps
from mmdet.structures.bbox import BaseBoxes

@MODELS.register_module()
class StandardMultiPrototypeReplayHead(StandardRoIHead):
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
                 max_proto=10) -> None:
        super().__init__(bbox_roi_extractor, bbox_head, mask_roi_extractor, mask_head, shared_head, train_cfg, test_cfg, init_cfg)
        self.replay = False
        self.task_split = task_split
        self.task_id = task_id
        self.max_proto = max_proto # decide the number of prototypes used for replay.
        device = next(self.parameters()).device
        with torch.no_grad():
            if previous_path != None and osp.exists(previous_path):
                assert task_id != 1
                self.replay = True
                print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
                # Note that not everything saved/loaded to/from rois_etc is useful, most of them are saved for modification convenience.
                # Also, even though all features are saved/loaded to/from rois_etc, the prototype is consistent across different stages. Features are saved for modification convenience(dont need to rerun last task for prototype). 
                self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"), map_location=device)
                previous_cls = range(task_split[0], task_split[task_id-1])
                tmp = []
                tmp_label = []
                if osp.exists(osp.join(previous_path, "mask.pth")):
                    save_idx = torch.load(osp.join(previous_path, "mask.pth"), map_location='cpu')
                else:
                    save_idx = []
                for i in previous_cls:
                    cls_mask = self.cls_targets == i
                    cls_bbox_feats = torch.mean(self.bbox_featss[cls_mask], dim=0, keepdim=True)
                    tmp.append(cls_bbox_feats) # coarse prototype.
                    tmp_label.append(i)
                    
                    tmp_ssssssss = self.bbox_featss[cls_mask].reshape(-1, 7*7*256) / self.bbox_featss[cls_mask].reshape(-1, 7*7*256).norm(dim=-1, keepdim=True)
                    corss_compare = tmp_ssssssss @ tmp_ssssssss.t()
                    
                    c_c_v, _ = (corss_compare >= 0.6).long().sum(dim=-1).sort(dim=-1, descending=True)
                    c_c_threash = c_c_v[-c_c_v.shape[0]//3]
                    used_idx = (corss_compare >= 0.6).long().sum(dim=-1) <= c_c_threash
                    distances_mask = corss_compare >= 0.6 # n * n

                    proto_count = 0
                    if i < len(save_idx):
                        tmp_mask = save_idx[i]
                    else:
                        tmp_mask = []
            
                    _, idx = distances_mask.sum(dim=-1).sort(dim=0, descending=True) # bs
                    for proto_count in range(self.max_proto-1):
                        for id_ in idx:
                            if proto_count < len(tmp_mask):
                                m = tmp_mask[proto_count].to(device)
                                print(f"{id_} is loaded from previous saves.")
                            else:
                                if used_idx[id_]:
                                    continue
                                m = distances_mask[id_]
                                tmp_mask.append(m)
                                print(f"{id_} is computed in this stage.")
                
                            print(used_idx.shape, m.shape)
                            used_idx = torch.logical_or(used_idx, m)
                            prototype = torch.mean(self.bbox_featss[cls_mask][m], dim=0, keepdim=True)
                            
                            tmp.append(prototype) # fine-grained prototype.
                            tmp_label.append(i)
                            break
                    if i >= len(save_idx):
                        save_idx.append(tmp_mask)
                self.bbox_featss = torch.cat(tmp, dim=0)
                self.tmp_label = torch.Tensor(tmp_label, device = self.bbox_featss.device).long()
                work_dir = get_work_dir(previous_path)
                torch.save(save_idx, osp.join(work_dir, "mask.pth"))
        
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        # losses = super().loss(x, rpn_results_list, batch_data_samples, replay=False)
        losses = super().loss(x, rpn_results_list, batch_data_samples)
        device = next(self.parameters()).device
        if self.replay:
            bbox_featss = self.bbox_featss
            bbox_featss = bbox_featss.to(device)
            replay_results = self.replay_loss(bbox_featss)
            losses.update(replay_results['replay_loss'])
        return losses
    
    def replay_loss(self, bbox_feats: Tuple[Tensor]) -> dict:
        """sampline_results and rois are not used.
        """
        bbox_feats = bbox_feats
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        pre_idx = self.task_split[self.task_id]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)

        losses["replay_loss_cls"] = F.cross_entropy(cls_score.softmax(dim=-1), self.tmp_label.to(cls_score.device))

        bbox_results.update(replay_loss=losses)
        return bbox_results

    
    def get_bbox_stuff(self, x, rpn_results_list, batch_data_samples, extract_gt=False):
        # Obtain bbox feature and stuff for prototype compute and replay.
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        
        copy_data_samples = copy.deepcopy(batch_data_samples)

        if not extract_gt:
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
            
            cls_target, cls_weight, bbox_target, bbox_weight = self.bbox_head.get_roi_targets(sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg)
        
        if extract_gt:
            bboxes = [
                data_sample.gt_instances.bboxes for data_sample in copy_data_samples
            ]
            
            # bboxes = [img for img in bboxes]1
            cls_target = torch.cat([
                data_sample.gt_instances.labels for data_sample in copy_data_samples
            ])
            
            cls_weight = cls_weight[:cls_target.shape[0]]
            bbox_weight = bbox_weight[:cls_target.shape[0]]
            
            bbox_target = torch.cat(bboxes)
            # print(2, cls_target.shape, bbox_target.shape)
            rois = bbox2roi(bboxes)
        
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)            
        
        mask = cls_target != 20
        
        target_count = 5
        
        # 计算当前True的数量
        current_count = torch.sum(mask).item()

        # 确定需要增加或减少的True的数量
        delta = target_count - current_count
        # print(delta)

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
            
        for c in cls_target[mask]:
            # print(c)
            self.counter[c] += 1
        
        return bbox_feats[mask], cls_target[mask], cls_weight[mask], bbox_target[mask], bbox_weight[mask], rois[mask]

def get_work_dir(previous_path):
    # 用这个根据给出的之前的path去推断当前的work_dir
    if "coco" in previous_path:
        return "./"
    splited_path = previous_path.split("_")
    task_id = int(splited_path[-1])
    splited_path[-1] = str(task_id + 1)
    return "_".join(splited_path)