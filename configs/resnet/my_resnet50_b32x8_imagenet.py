train_images_root = r'data/data_iqa/train'
train_annotations_root = r'data/data_iqa/meta/train.txt'
test_images_root = r'data/data_iqa/test'
test_annotations_root = r'data/data_iqa/meta/test.txt'
num_classes=1
size = (384,128)#H*W
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='HyperNet',
        lda_out_channels=16,#16
        hyper_in_channels=192,#192
        target_in_size=384,#384
        target_fc1_size=192,#192
        target_fc2_size=96,#96
        target_fc3_size=48,#48
        target_fc4_size=24,#24
        feature_size1=12,#12
        feature_size2=4
        # depth=50,
        # num_stages=4,
        # out_indices=(3, ),
        # style='pytorch'
    ),
    # neck=dict(type='GlobalAveragePooling'),
    neck=dict(type='Conv', stride=(3, 2), kernel_size=(3, 3)),
    head=dict(
        type='LinearRegHead',
        num_classes=num_classes,
        in_channels=2048,
        loss=dict(type='L1Loss'),#loss_weight=1.0
        topk=(1, 1)))
dataset_type = 'IQA'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Pad', size=size),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Pad', size=size),
    # dict(type='CenterCrop', crop_size=size),#224
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=40,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=train_images_root,
        ann_file=train_annotations_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=test_images_root,
        ann_file=test_annotations_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=test_images_root,
        ann_file=test_annotations_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')#, metric_options={'topk': (1, )}, save_best='accuracy_top-1'
optimizer = dict(type='Adam', lr=2e-5, weight_decay=5e-4)
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = r'checkpoints/resnet50_batch256_imagenet_20200708-cfb998bf.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/00.resnet50_b32x8_imagenet-20220728'
gpu_ids = range(0, 1)