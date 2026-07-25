_base_ = ['./qfdet_r50_fpn_1x_vtuav_strategy_B.py']

model = dict(
    use_modality_gate=True,
    bbox_head=dict(
        use_small_object_loss=True,
        anchor_generator=dict(
            strides=[4, 8, 16, 32, 64]
        )
    ),
    bbox_prehead=dict(
        use_small_object_loss=True,
        anchor_generator=dict(
            strides=[4, 8, 16, 32, 64]
        )
    ),
    neck=dict(
        start_level=0,
        num_outs=5
    )
)

# Output directory for Strategy C (P2 High-Res FPN Level)
work_dir = 'p:/project/hackothon/jnn_shivamogga/output/strategy_C_highres_fpn/work_dir'
