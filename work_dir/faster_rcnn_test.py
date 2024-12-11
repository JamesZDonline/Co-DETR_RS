model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='CustomDinoViTV2',
        pretrained_weight_path=
        '/mnt/sarl_commons06/Wernke_projects/zimmejr1/geopacha/output/unsupervised/from_container/NEH_new/6gpus/eval/training_324999/teacher_checkpoint.pth',
        freeze_backbone=True),
    neck=dict(
        type='SFP',
        in_channels=[1024],
        out_channels=256,
        num_outs=5,
        use_p2=True,
        use_act_checkpoint=False),
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
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='CIoULoss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=6,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='CIoULoss', loss_weight=1.0))),
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
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=50,
    hooks=[
        dict(
            type='MMDetWandbHook',
            init_kwargs=dict(project='mmdetection_models'),
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100)
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
custom_imports = dict(
    imports=[
        'mmdet.datasets.rastervision_dataset', 'mmdet.models.backbones.dinov2'
    ],
    allow_failed_imports=False)
IMAGE_DIR = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images'
VECTOR_DIR = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/'
TRAIN_SCENE_PATH = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/ml_data_2024_12_09/scenes_train.csv'
VAL_SCENE_PATH = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/ml_data_2024_12_09/scenes_validation.csv'
train_pipeline = [
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
dataset_type = ('RasterVisionDataset', )
data = dict(
    train=dict(
        type='RasterVisionDataset',
        image_dir=
        '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images',
        vector_dir=
        '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/',
        scene_csv_path=
        '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/ml_data_2024_12_09/scenes_train.csv',
        pipeline=[
            dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        data_type='training',
        neg_ratio=5,
        max_windows=100,
        rgb=False),
    val=dict(
        type='RasterVisionDataset',
        image_dir=
        '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images',
        vector_dir=
        '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/',
        data_type='validation',
        rgb=False,
        scene_csv_path=
        '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/ml_data_2024_12_09/scenes_validation.csv',
        pipeline=[
            dict(
                type='MultiScaleFlipAug',
                img_scale=(224, 224),
                flip=False,
                transforms=[
                    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RasterVisionDataset',
        image_dir=
        '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images',
        vector_dir=
        '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/',
        data_type='testing',
        rgb=False,
        scene_csv_path=
        '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/ml_data_2024_12_09/scenes_validation.csv',
        pipeline=[
            dict(
                type='MultiScaleFlipAug',
                img_scale=(224, 224),
                flip=False,
                transforms=[
                    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(metric='mAP')
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=50)
optimizer = dict(
    type='AdamW', lr=0.0001, weight_decay=0.05, eps=1e-08, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
work_dir = 'work_dir'
auto_resume = False
gpu_ids = range(0, 2)
