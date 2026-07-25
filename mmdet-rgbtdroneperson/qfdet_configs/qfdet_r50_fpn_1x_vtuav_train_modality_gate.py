_base_ = ['./qfdet_r50_fpn_1x_vtuav.py']

model = dict(
    use_modality_gate=True
)

# Pretrained checkpoint to fine-tune from
load_from = 'p:/project/hackothon/jnn_shivamogga/epoch_11_qfdet_vtuav.pth'

img_norm_cfg = dict(
    mean_list=([83.20, 92.24, 97.70], [134.84, 134.84, 134.84]),
    std_list=([57.77, 57.41, 57.69], [81.58, 81.58, 81.58]), to_rgb=True)

train_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('VTUAV_co/train/images', 'VTUAV_ir/train/images')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='MultiNormalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

val_test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('VTUAV_co/val/images', 'VTUAV_ir/val/images')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='MultiNormalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

# Fine-tuning schedule: 5 epochs with reduced learning rate
runner = dict(type='EpochBasedRunner', max_epochs=5)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=val_test_pipeline)
)

# Output directory for logs and fine-tuned checkpoints
work_dir = 'p:/project/hackothon/jnn_shivamogga/output/strategy_A_modality_gate/work_dir'
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='bbox')
