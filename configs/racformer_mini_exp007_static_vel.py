_base_ = ['./racformer_r50_nuimg_704x256_f8.py']
data = dict(val=dict(max_samples=300))
# barrier=5, traffic_cone=9
model = dict(pts_bbox_head=dict(bbox_coder=dict(static_classes=[5, 9])))
