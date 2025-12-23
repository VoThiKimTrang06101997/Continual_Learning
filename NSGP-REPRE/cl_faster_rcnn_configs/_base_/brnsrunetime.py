default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best = "auto"),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

runner_type = "BRNullSpaceRunner"
# randomness = dict(seed=625895394)
find_unused_parameters=True



# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically Infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/segmentation/VOCdevkit/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/segmentation/',
#         'data/': 's3://openmmlab/datasets/segmentation/'
#     }))
# backend_args = None


all_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VOCTask',
        data_root='/home/Newdisk/wuqirui/datasets/VOC/VOCdevkit/',
        ann_file='VOC2007/ImageSets/Main/trainval.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        task_split = [0,20],
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(1000, 600), keep_ratio=True),
            # avoid bboxes being resized
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor'))
        ],
        backend_args=None))
