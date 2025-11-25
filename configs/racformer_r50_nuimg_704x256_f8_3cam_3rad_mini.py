_base_ = ['./racformer_r50_nuimg_704x256_f8_3cam_3rad.py']

# Data settings
data = dict(
    val=dict(
        max_samples=50
    ),
    test=dict(
        max_samples=50
    )
)
