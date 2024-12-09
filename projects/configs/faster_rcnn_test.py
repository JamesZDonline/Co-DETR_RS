_base_ = [
    '../configs/_base_/models/faster_rcnn_r50_fpn_focal_ciou_loss.py',
    '../configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmdet.datasets.rastervision_dataset'],
    allow_failed_imports=False)

IMAGE_DIR = "/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/Analysis_Images"
VECTOR_DIR = "/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/"
TRAIN_SCENE_PATH ='/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/ml_data_2024_12_09/scenes_train.csv'
VAL_SCENE_PATH = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/ml_data_2024_12_09/scenes_validation_sample.csv'
# VAL_SCENE_PATH = '/mnt/sarl_commons06/Wernke_projects/GeoPACHA/Imagery_Machine_Learning/ObjectDetection/FinalData/Ignore_techo/ml_data_2024_12_09/scenes_train_tiny.csv'


train_pipeline = [
    dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=.5),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),]

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

        ]
    )
]
dataset_type='RasterVisionDataset',
data = dict(
    train=dict(type='RasterVisionDataset',
               image_dir = IMAGE_DIR,
               vector_dir = VECTOR_DIR,
               scene_csv_path=TRAIN_SCENE_PATH,pipeline=train_pipeline,
               data_type='training',
               neg_ratio=5,
               max_windows=100,
            #    test_code=True,
               rgb=True),            
    val=dict(type='RasterVisionDataset',
               image_dir = IMAGE_DIR,
               vector_dir = VECTOR_DIR,
               data_type='validation',
               rgb=True,
            #    test_code=True,
               scene_csv_path=VAL_SCENE_PATH,pipeline=test_pipeline),
    test=dict(type='RasterVisionDataset',
               image_dir = IMAGE_DIR,
               vector_dir = VECTOR_DIR,
               data_type='testing',
            #    test_code=True,
               rgb=True,
               scene_csv_path=VAL_SCENE_PATH,pipeline=test_pipeline))

# train_dataloader = dict(
#     samples_per_gpu=256,
#     workers_per_gpu=20,
# )

# val_dataloader = dict(
#     samples_per_gpu=512,
#     workers_per_gpu=20,
# )

evaluation = dict(metric='mAP')

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[7])
runner = dict(type='EpochBasedRunner', max_epochs=12)

optimizer = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
    type='SGD',  # Type of optimizers, refer to https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/optimizer/default_constructor.py#L13 for more details
    lr=0.02,  # Learning rate of optimizers, see detail usages of the parameters in the documentation of PyTorch
    momentum=0.9,  # Momentum
    weight_decay=0.0001)  # Weight decay of SGD
optimizer_config = dict(  # Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details.
    grad_clip=None)  # Most of the methods do not use gradient clip
lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',  # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
    warmup='linear',  # The warmup policy, also support `exp` and `constant`.
    warmup_iters=500,  # The number of iterations for warmup
    warmup_ratio=
    0.001,  # The ratio of the starting learning rate used for warmup
    step=[8, 11])  # Steps to decay the learning rate
runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=12) # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    interval=4)  # The save interval is 1
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        # dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
        # dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
         init_kwargs={'project': 'mmdetection_models'},
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=100)
    ]
)  # The logger used to record the training process.
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'  # The level of logging.
load_from = None  # load models as a pre-trained model from a given path. This will not resume training.
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.
work_dir = 'work_dir'  # Directory to save the model checkpoints and logs for the current experiments.
