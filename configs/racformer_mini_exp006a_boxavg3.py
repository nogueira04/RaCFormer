_base_ = ['./racformer_r50_nuimg_704x256_f8.py']
data = dict(val=dict(max_samples=300))
model = dict(pts_bbox_head=dict(bbox_coder=dict(box_ensemble=3)))
