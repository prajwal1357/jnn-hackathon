_base_ = ['./qfdet_r50_fpn_1x_vtuav_train_modality_gate.py']

model = dict(
    use_modality_gate=True,
    bbox_head=dict(use_small_object_loss=True),
    bbox_prehead=dict(use_small_object_loss=True)
)

# Output directory for Strategy B (Small-Object-Weighted Loss) checkpoints
work_dir = 'p:/project/hackothon/jnn_shivamogga/output/strategy_B_small_object_loss/work_dir'
