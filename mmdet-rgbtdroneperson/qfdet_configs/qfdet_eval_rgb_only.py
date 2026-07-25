_base_ = ['./qfdet_r50_fpn_1x_vtuav.py']

val_test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('VTUAV_co/val/images', 'VTUAV_co/val/images')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='MultiNormalize', mean_list=([83.20, 92.24, 97.70], [83.20, 92.24, 97.70]), std_list=([57.77, 57.41, 57.69], [57.77, 57.41, 57.69]), to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

test_test_pipeline = [
    dict(type='LoadImagePairFromFile', spectrals=('VTUAV_co/test/images', 'VTUAV_co/test/images')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='MultiNormalize', mean_list=([83.20, 92.24, 97.70], [83.20, 92.24, 97.70]), std_list=([57.77, 57.41, 57.69], [57.77, 57.41, 57.69]), to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    val=dict(pipeline=val_test_pipeline),
    test=dict(pipeline=test_test_pipeline)
)
