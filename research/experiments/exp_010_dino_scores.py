"""Experiment 010: Save per-prediction DINOv3 similarity scores.

For each prediction, saves the full 10-class cosine similarity vector.
This allows offline experimentation with different strategies:
- H9.2: Score fusion (score * (1 + alpha * class_agreement))
- H9.3: FP filtering (remove if max_sim < threshold)
- H9.5: Replace score with DINOv3 confidence

Output: research/outputs/exp_010/dino_scores.pt
  List of dicts per sample: {'sims': (N_pred, 10), 'projected_mask': (N_pred,)}
"""
import os
import sys
import json
import pickle
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = '/srv/nfs/shared/gnmp/RaCFormer'
DATA_ROOT = '/mnt/nfs/shared/nuscenes'
BATCH_SIZE = 64
HF_TOKEN = os.environ.get('HF_TOKEN', None)

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
MIN_BOX_SIZE = 15


def project_3d_to_2d(center_3d, r, t, K):
    pt_cam = r @ center_3d[:3] + t
    if pt_cam[2] <= 0:
        return None
    u = K[0, 0] * pt_cam[0] / pt_cam[2] + K[0, 2]
    v = K[1, 1] * pt_cam[1] / pt_cam[2] + K[1, 2]
    return float(u), float(v), float(pt_cam[2])


def get_crop(center, dims, cam_params, images):
    best = None
    best_depth = float('inf')
    for ci, cp in enumerate(cam_params):
        proj = project_3d_to_2d(center, cp['r'], cp['t'], cp['K'])
        if proj is None:
            continue
        u, v, depth = proj
        img = images[ci]
        if img is None:
            continue
        iw, ih = img.size
        if u < 0 or u > iw or v < 0 or v > ih or depth >= best_depth:
            continue
        l, w, h = dims[0], dims[1], dims[2]
        hw = cp['K'][0, 0] * max(l, w) / (2 * depth) * 1.2
        hh = cp['K'][1, 1] * h / (2 * depth) * 1.2
        x1, y1 = max(0, int(u - hw)), max(0, int(v - hh))
        x2, y2 = min(iw, int(u + hw)), min(ih, int(v + hh))
        if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
            continue
        best_depth = depth
        best = (ci, x1, y1, x2, y2)
    if best is None:
        return None
    ci, x1, y1, x2, y2 = best
    return images[ci].crop((x1, y1, x2, y2))


def main():
    print("Loading inputs...", flush=True)
    with open(os.path.join(PROJECT_ROOT, 'research/outputs/mini_preds/predictions_simple.pkl'), 'rb') as f:
        results = pickle.load(f)
    proto_data = torch.load(os.path.join(PROJECT_ROOT, 'research/dino_features/prototypes.pt'),
                            map_location='cpu', weights_only=True)
    prototypes = torch.nn.functional.normalize(proto_data['prototypes'], dim=1)
    with open(os.path.join(PROJECT_ROOT, 'nuscenes_infos_val_sweep.pkl'), 'rb') as f:
        val_infos = pickle.load(f)['infos'][:300]
    print(f"  {len(results)} samples, {prototypes.shape[0]} prototypes", flush=True)

    print("Loading DINOv3...", flush=True)
    model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModel.from_pretrained(model_name, token=HF_TOKEN)
    model.eval().cuda()
    print("  Ready", flush=True)

    all_scores = []
    for si in range(len(results)):
        result = results[si]
        info = val_infos[si]
        n = len(result['boxes_3d'])

        # Preload cameras
        cam_params = []
        images = []
        for cn in CAM_NAMES:
            ci = info['cams'][cn]
            r = np.linalg.inv(ci['sensor2lidar_rotation'])
            t = -ci['sensor2lidar_translation'] @ r.T
            cam_params.append({'r': r, 't': t, 'K': ci['cam_intrinsic']})
            try:
                images.append(Image.open(ci['data_path'].replace('data/nuscenes', DATA_ROOT)).convert('RGB'))
            except:
                images.append(None)

        # Collect crops
        crop_map = {}  # pred_idx -> crop
        for pi in range(n):
            crop = get_crop(result['boxes_3d'][pi, :3], result['boxes_3d'][pi, 3:6], cam_params, images)
            if crop is not None:
                crop_map[pi] = crop

        for img in images:
            if img is not None:
                img.close()

        # Extract features
        sims = torch.zeros(n, 10)
        projected = torch.zeros(n, dtype=torch.bool)

        if crop_map:
            indices = sorted(crop_map.keys())
            crops = [crop_map[i] for i in indices]
            feats = []
            for bs in range(0, len(crops), BATCH_SIZE):
                batch = crops[bs:bs + BATCH_SIZE]
                inputs = processor(images=batch, return_tensors="pt").to('cuda')
                with torch.no_grad():
                    out = model(**inputs)
                feats.append(out.pooler_output.cpu())
            feats = torch.cat(feats, dim=0)
            feats = torch.nn.functional.normalize(feats, dim=1)
            batch_sims = feats @ prototypes.T

            for bi, pi in enumerate(indices):
                sims[pi] = batch_sims[bi]
                projected[pi] = True

        all_scores.append({'sims': sims, 'projected': projected})

        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{len(results)}: {sum(s['projected'].sum().item() for s in all_scores)} projected", flush=True)

    out_dir = os.path.join(PROJECT_ROOT, 'research/outputs/exp_010')
    os.makedirs(out_dir, exist_ok=True)
    torch.save(all_scores, os.path.join(out_dir, 'dino_scores.pt'))
    print(f"Saved to {out_dir}/dino_scores.pt", flush=True)


if __name__ == '__main__':
    main()
