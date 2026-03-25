_base_ = ['./racformer_r50_nuimg_704x256_f8.py']
data = dict(val=dict(max_samples=300))

# Enable horizontal flip in test pipeline
# Override the MultiScaleFlipAug3D to use flip=True
test_pipeline = {{_base_.test_pipeline}}
