# dataset settings for COCO Incremental Detection (đọc trực tiếp từ zip - không cần unzip)
dataset_type = 'CocoTaskDataset'
data_root = 'E:/NghienCuu/Continual_Learning/NSGP-REPRE/dataset/coco/'  # Thư mục chứa các file .zip

task_id = 1
train_task_split = [0, 40, 80]  # Ví dụ 40+40 split
val_task_split = [0, train_task_split[task_id]]

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        task_split=train_task_split,
        task_id=task_id,
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',  
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        task_split=val_task_split,
        task_id=1,
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',     # File JSON phải unzip
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    classwise=True,
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator
