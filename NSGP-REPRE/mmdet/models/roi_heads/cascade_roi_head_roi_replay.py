# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import ModuleList
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, MultiConfig, OptConfigType,
                         OptMultiConfig)
from ..utils.misc import empty_instances, unpack_gt_instances
from .cascade_roi_head import CascadeRoIHead

import copy
import os.path as osp
import os

import torch.nn.functional as F

from mmdet.evaluation.functional.bbox_overlaps import bbox_overlaps
from mmdet.structures.bbox import BaseBoxes

def get_relu_mask(t, threshold=0.3):
    if threshold == 0:
        return t==0
    values, idx = t.sort(descending=True)
    length = idx.shape[-1]
    actual_threshold = values[:, int(length * threshold)]
    # print(t.shape, int(length * threshold), (t>0).sum())
    # print(actual_threshold)
    # print((t <= actual_threshold.unsqueeze(-1)).sum(dim=-1))
    # exit()
    return t <= actual_threshold.unsqueeze(-1)


def get_work_dir(previous_path):
    # 用这个根据给出的之前的path去推断当前的work_dir
    if "coco" in previous_path:
        return "./"
    splited_path = previous_path.split("_")
    task_id = int(splited_path[-1])
    splited_path[-1] = str(task_id + 1)
    return "_".join(splited_path)

@MODELS.register_module()
class CascadeRoIHeadRoIReplay(CascadeRoIHead):
    """Cascade roi head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(self,
                 num_stages: int,
                 stage_loss_weights: Union[List[float], Tuple[float]],
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
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, \
            'Shared head is not supported in Cascade RCNN anymore'

        super().__init__(
            num_stages=num_stages,
            stage_loss_weights=stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        self.pos_neg_counter = {0.2: [0,0], 0.3: [0,0], 0.4: [0,0], 0.5: [0,0], 0.6: [0,0], 0.7: [0,0], 0.8: [0,0], 0.9: [0,0], 1.0: [0,0], 2.0: [0,0]}
        self.replay = False
        self.task_split = task_split
        self.task_id = task_id
        self.max_proto = 10
        device = next(self.parameters()).device
        # with torch.no_grad():
        #     if previous_path != None and osp.exists(previous_path):
        #         # DBSCAN baseline
        #         assert task_id != 1
        #         self.replay = True
        #         print("Load previous stuff from ", osp.join(previous_path, "rois_etc.pth"))
        #         self.bbox_featss, self.cls_targets, self.cls_weights, self.bbox_targets, self.bbox_weights, self.roiss = torch.load(osp.join(previous_path, "rois_etc.pth"), map_location=device)
        #         self.bbox_featsss = []
        #         self.tmp_labels = []
        #         for stage in range(self.num_stages):
        #             cls_targets= self.cls_targets[:, stage, ...]
        #             bbox_featss = self.bbox_featss[:, stage, ...]
        #             previous_cls = range(task_split[0], task_split[task_id-1])
        #             tmp = []
        #             tmp_label = []
        #             if osp.exists(osp.join(previous_path, "mask.pth")):
        #                 save_idx = torch.load(osp.join(previous_path, "mask.pth"), map_location='cpu')
        #             else:
        #                 save_idx = []
        #             for i in previous_cls:
        #                 cls_mask = cls_targets == i
        #                 cls_bbox_feats = torch.mean(bbox_featss[cls_mask], dim=0, keepdim=True)
        #                 tmp.append(cls_bbox_feats)
        #                 tmp_label.append(i)
        #                 for ppppp in os.listdir(previous_path):
        #                     if "best" in ppppp:
        #                         previous_best = osp.join(previous_path, ppppp)
        #                 # previous_best = "/home/Newdisk/wuqirui/detclip/mmdetection-main/work_dirs/ns3_split_id/cl_faster_rcnn_ns3_split_id_5_5_1/best_pascal_voc_mAP_epoch_9.pth"
        #                 # a = torch.load(previous_best, map_location="cpu")['state_dict']
        #                 # self.bbox_head.shared_fcs[0].weight.data = a['roi_head.bbox_head.shared_fcs.0.weight']
        #                 # self.bbox_head.shared_fcs[0].bias.data = a['roi_head.bbox_head.shared_fcs.0.bias']
                        
        #                 # self.bbox_head.shared_fcs[1].weight.data = a['roi_head.bbox_head.shared_fcs.1.weight']
        #                 # self.bbox_head.shared_fcs[1].bias.data = a['roi_head.bbox_head.shared_fcs.1.bias']
                        
        #                 tmp_ssssssss = bbox_featss[cls_mask].reshape(-1, 7*7*256) / bbox_featss[cls_mask].reshape(-1, 7*7*256).norm(dim=-1, keepdim=True)
        #                 corss_compare = tmp_ssssssss @ tmp_ssssssss.t()
                        
        #                 c_c_v, _ = (corss_compare >= 0.6).long().sum(dim=-1).sort(dim=-1, descending=True)
        #                 print(c_c_v.shape)
        #                 c_c_threash = c_c_v[-c_c_v.shape[0]//3]
        #                 used_idx = (corss_compare >= 0.6).long().sum(dim=-1) <= c_c_threash
        #                 distances_mask = corss_compare >= 0.6 # n * n
        #                 tmp_feats = (distances_mask / distances_mask.sum(dim=-1, keepdim=True)) @ bbox_featss[cls_mask].reshape(-1, 256*7*7)
                        
        #                 # cc_c = self.bbox_head.get_relu(tmp_feats.reshape(-1, 256, 7, 7))[-1] != 0
        #                 # cc_c = torch.logical_not(
        #                     # get_relu_mask(self.bbox_head.get_relu(tmp_feats.reshape(-1, 256, 7, 7))[-1], 0.2)
        #                     # )
                        
        #                 proto_count = 0
        #                 if i < len(save_idx):
        #                     tmp_mask = save_idx[i]
        #                 else:
        #                     tmp_mask = []
                            
                        
        #                 # relu_sum = cc_c * relu_mask # bs * 1024
        #                 # relu_sum = relu_sum.to(torch.long)
        #                 # _, idx = relu_sum.sum(dim=-1).sort(dim=0, descending=True) # bs
        #                 _, idx = distances_mask.sum(dim=-1).sort(dim=0, descending=True) # bs
        #                 for proto_count in range(self.max_proto-1):
        #                     # print(idx)
        #                     for id_ in idx:
        #                         if proto_count < len(tmp_mask):
        #                             m = tmp_mask[proto_count].to(device)
        #                             print(f"{id_} ++++++++++")
        #                         else:
        #                             if used_idx[id_]:
        #                                 continue
        #                             # _, compare_idx = corss_compare[id_].sort(dim=0, descending=True)
        #                             # one_third = compare_idx.shape[0] // self.max_proto
        #                             # one_third = one_third if one_third != 0 else one_third + 1
        #                             # m = torch.zeros_like(compare_idx, dtype=torch.bool)
        #                             # m.scatter_(0, compare_idx[:one_third], True)
        #                             m = distances_mask[id_]
        #                             tmp_mask.append(m)
        #                             print(f"{id_} ///////")
                    
        #                         print(used_idx.shape, m.shape)
        #                         used_idx = torch.logical_or(used_idx, m)
        #                         prototype = torch.mean(bbox_featss[cls_mask][m], dim=0, keepdim=True)
        #                         # print(relu_mask, "\n", cc_c[id_])
        #                         # relu_mask = torch.logical_and(relu_mask, cc_c[id_]==False)
        #                         # aaa = self.bbox_head.get_relu(prototype) != 0
        #                         # print((aaa * relu_mask).sum(), (relu_mask!=0).sum())
        #                         # relu_mask = torch.logical_and(relu_mask, aaa!=False)
                                
        #                         tmp.append(prototype)
        #                         tmp_label.append(i)
        #                         break
        #                 if i >= len(save_idx):
        #                     save_idx.append(tmp_mask)
        #                 # exit()
        #             self.bbox_featsss.append(torch.cat(tmp, dim=0))
        #             self.tmp_labels.append(torch.Tensor(tmp_label, device = self.bbox_featss.device).long())
        #             work_dir = get_work_dir(previous_path)
        #             torch.save(save_idx, osp.join(work_dir, f"mask_{stage}.pth"))
                    
    
    def loss(self, x: Tuple[Tensor], rpn_results_list, batch_data_samples) -> dict:
        losses = super().loss(x, rpn_results_list, batch_data_samples)
        device = next(self.parameters()).device
        if self.replay:
            for stage in range(self.num_stages):
                # do sampling
                # mask = torch.randperm(self.bbox_featss.shape[0])[:64].to(self.bbox_featss.device)
                bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = self.bbox_featsss[stage], self.cls_targets[stage], self.cls_weights[stage], self.bbox_targets[stage], self.bbox_weights[stage], self.roiss[stage]
                bbox_featss = self.bbox_featsss[stage]
                bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss = bbox_featss.to(device), cls_targets.to(device), cls_weights.to(device), bbox_targets.to(device), bbox_weights.to(device), roiss.to(device)
                
                sampling_results = [cls_targets, cls_weights, bbox_targets, bbox_weights]
                replay_results = self.replay_loss(bbox_featss, None, None, stage)
                # print("before update", losses)
                losses.update(replay_results["replay_loss"])
        return losses
    
    
    def replay_loss(self, bbox_feats: Tuple[Tensor],
                  sampling_results: List[SamplingResult],
                  rois, stage) -> dict:
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
        bbox_feats = bbox_feats
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head[stage](bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        losses = dict()
        
        pre_idx = self.task_split[self.task_id]
        cls_score = torch.cat([cls_score[:, :pre_idx], cls_score[:,-1:]], dim=-1)

        losses[f"replay_loss_cls_{stage}"] = F.cross_entropy(cls_score.softmax(dim=-1), self.tmp_labels[stage].to(cls_score.device)) * self.stage_loss_weights[stage]

        bbox_results.update(replay_loss = losses)
        # print(losses)
        return bbox_results
    
    
    def get_bbox_stuff(self, x, rpn_results_list, batch_data_samples, extract_gt=False):
        
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs
        
        copy_data_samples = copy.deepcopy(batch_data_samples)

        bbox_featss = []
        cls_targets = []
        cls_weights = []
        bbox_targets = []
        bbox_weights = []
        roiss = []
        
        num_imgs = len(batch_data_samples)
        results_list = rpn_results_list
        for stage in range(self.num_stages):
            self.current_stage = stage

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[stage]
                bbox_sampler = self.bbox_sampler[stage]

                for i in range(num_imgs):
                    results = results_list[i]
                    # rename rpn_results.bboxes to rpn_results.priors
                    try:
                        results.priors = results.pop('bboxes')
                    except:
                        pass

                    assign_result = bbox_assigner.assign(
                        results, batch_gt_instances[i],
                        batch_gt_instances_ignore[i])

                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        results,
                        batch_gt_instances[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)
        
            rois = bbox2roi([res.priors for res in sampling_results])
            bbox_roi_extractor = self.bbox_roi_extractor[stage]
            bbox_feats = bbox_roi_extractor(
                x[:bbox_roi_extractor.num_inputs], rois)
            
            # print(torch.sort(cls))
            bbox_head = self.bbox_head[stage]
            cls_target, cls_weight, bbox_target, bbox_weight = bbox_head.get_roi_targets(sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg)
            # print(1, cls_target.shape, bbox_target.shape)
        
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
                
            bbox_feats = bbox_feats[mask].unsqueeze(1)
            cls_target = cls_target[mask].unsqueeze(1)
            cls_weight = cls_weight[mask].unsqueeze(1)
            bbox_target = bbox_target[mask].unsqueeze(1)
            bbox_weight = bbox_weight[mask].unsqueeze(1)
            rois = rois[mask].unsqueeze(1)

            bbox_featss.append(bbox_feats)
            cls_targets.append(cls_target)
            cls_weights.append(cls_weight)
            bbox_targets.append(bbox_target)
            bbox_weights.append(bbox_weight)
            roiss.append(rois)
        
        bbox_featss = torch.cat(bbox_featss, dim=1)
        cls_targets = torch.cat(cls_targets, dim=1)
        cls_weights = torch.cat(cls_weights, dim=1)
        bbox_targets = torch.cat(bbox_targets, dim=1)
        bbox_weights = torch.cat(bbox_weights, dim=1)
        roiss = torch.cat(roiss, dim=1)
            
        return bbox_featss, cls_targets, cls_weights, bbox_targets, bbox_weights, roiss
    
    
    
    
    
    
    
    
    def predict_wo_postprocess(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        proposals = [res.bboxes for res in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)

        # if rois.shape[0] == 0:
        #     return empty_instances(
        #         batch_img_metas,
        #         rois.device,
        #         task_type='bbox',
        #         box_type=self.bbox_head.predict_box_type,
        #         num_classes=self.bbox_head.num_classes,
        #         score_per_cls=rcnn_test_cfg is None)

        rois, cls_scores, bbox_preds = self._refine_roi(
            x=x,
            rois=rois,
            batch_img_metas=batch_img_metas,
            num_proposals_per_img=num_proposals_per_img)
        
        ret_scores = copy.deepcopy(cls_scores) 

        results_list = self.bbox_head[-1].predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            rcnn_test_cfg=None)
        return results_list, ret_scores
    
    
    def predict_w_postprocess(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False,
                prepared_scores = None, use_gts=False, use_print=False, scores_thresh = 2.0):
        def scale_bbox(data_sample):
            scale_factor = [s for s in data_sample.scale_factor]
            boxes = data_sample.gt_instances.bboxes
            if isinstance(boxes, BaseBoxes):
                boxes.rescale_(scale_factor)
                return boxes
            else:
                # Tensor boxes will be treated as horizontal boxes
                repeat_num = int(boxes.size(-1) / 2)
                scale_factor = boxes.new_tensor(scale_factor).repeat((1, repeat_num))
                return boxes * scale_factor
        
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        proposals = [res.bboxes for res in rpn_results_list]
        proposals = []
        for idx, res in enumerate(rpn_results_list):
            bboxes = res.bboxes
            gt_bboxes = scale_bbox(batch_data_samples[idx])
            
            rois = bbox_overlaps(gt_bboxes.cpu().numpy(), bboxes.cpu().numpy(), mode='iou')
            rois = rois.T
            # print(rois.shape)
            rois = torch.Tensor(rois).to(gt_bboxes.device)            
            # scores = rois.max(dim=-1)[0]
            scores = rois

            bboxes_mask = (scores <= scores_thresh).all(dim=-1)
            self.pos_neg_counter[scores_thresh][0] += (torch.ones_like(bboxes_mask.flatten())).sum().item()
            self.pos_neg_counter[scores_thresh][1] += (bboxes_mask.flatten() == False).sum().item()
            proposals.append(bboxes[bboxes_mask])
            if use_print:
                print(f"{bboxes_mask.sum()} / {len(bboxes_mask)}")
        if use_gts:
            bboxes = [
                scale_bbox(data_sample) for data_sample in batch_data_samples
            ]
            proposals = bboxes
            
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)

        # if rois.shape[0] == 0:
        #     return empty_instances(
        #         batch_img_metas,
        #         rois.device,
        #         task_type='bbox',
        #         box_type=self.bbox_head.predict_box_type,
        #         num_classes=self.bbox_head.num_classes,
        #         score_per_cls=rcnn_test_cfg is None)

        rois, cls_scores, bbox_preds = self._refine_roi(
            x=x,
            rois=rois,
            batch_img_metas=batch_img_metas,
            num_proposals_per_img=num_proposals_per_img)
        
        if prepared_scores is not None:
            cls_scores = prepared_scores
            
        # ret_scores = copy.deepcopy(cls_scores) 
            
        results_list = self.bbox_head[-1].predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        # print(result_list)
        return results_list
    