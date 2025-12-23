_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection_40_40_task1.py',  # Base dataset cho task 1 (40 class đầu)
    '../_base_/schedules/schedule_1x_sgdnscl.py',
    '../_base_/brnsrunetime.py'
]

# ==================== ĐƯỜNG DẪN DATASET & CHECKPOINT ====================
data_root = 'E:/NghienCuu/Continual_Learning/NSGP-REPRE/dataset/coco/'  # Chứa train2017/, val2017/, annotations/
work_dir = 'E:/NghienCuu/Continual_Learning/NSGP-REPRE/work_dirs/checkpoints/coco_40_40_task1/'
# =====================================================================

# ==================== TASK SETTING (40+40 split) ======================
task_id = 1
train_task_split = [0, 40, 80]  # 0-39: base classes, 40-79: new classes (COCO index từ 0)
val_task_split = [0, train_task_split[task_id]]

previous_dir = f'E:/NghienCuu/Continual_Learning/NSGP-REPRE/work_dirs/checkpoints/coco_40_40_task{task_id-1}/'
# =====================================================================

ns_thresh = 0.0
ns_init = False
rr_thresh = [0.5, 0.7]

# model settings
model = dict(
    type='FasterRCNNRoIReplay',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=r'E:\NghienCuu\Continual_Learning\NSGP-REPRE\pretrained_checkpoint\resnet50-0676ba61.pth'  # Raw string
        )),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardMultiPrototypeReplayHead',
        previous_path=previous_dir,
        task_id=task_id,
        task_split=train_task_split,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHeadTask',
            task_id=task_id,
            task_split=train_task_split,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,  # COCO tổng cộng 80 class
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    
    # ==================== TRAINING CONFIG (fix loss_bbox = nan) ====================
    train_cfg=dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_pre=2000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.4,  # Giảm xuống
            neg_iou_thr=0.4,
            min_pos_iou=0.4,
            match_low_quality=True,  # Bật
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=2048,  # Tăng mạnh
            pos_fraction=0.5,  # Tăng lên
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
)
