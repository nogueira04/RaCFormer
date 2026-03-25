"""Experiment 009: DINOv3 classification verification (H9.1) — optimized.

Preloads camera images per sample to avoid repeated NFS reads.
Runs in dino_extract env (Python 3.10+).
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


def project_3d_to_2d(center_3d, lidar2cam_r, lidar2cam_t, cam_intrinsic):
    pt_cam = lidar2cam_r @ center_3d[:3] + lidar2cam_t
    if pt_cam[2] <= 0:
        return None
    u = cam_intrinsic[0, 0] * pt_cam[0] / pt_cam[2] + cam_intrinsic[0, 2]
    v = cam_intrinsic[1, 1] * pt_cam[1] / pt_cam[2] + cam_intrinsic[1, 2]
    return float(u), float(v), float(pt_cam[2])


def get_crop(center, dims, cam_params, images):
    """Find best camera and crop for a 3D prediction. Images pre-loaded."""
    best = None
    best_depth = float('inf')
    for cam_idx, cp in enumerate(cam_params):
        proj = project_3d_to_2d(center, cp['r'], cp['t'], cp['K'])
        if proj is None:
            continue
        u, v, depth = proj
        img = images[cam_idx]
        if img is None:
            continue
        iw, ih = img.size
        if u < 0 or u > iw or v < 0 or v > ih:
            continue
        if depth >= best_depth:
            continue
        l, w, h = dims[0], dims[1], dims[2]
        half_w = cp['K'][0, 0] * max(l, w) / (2 * depth) * 1.2
        half_h = cp['K'][1, 1] * h / (2 * depth) * 1.2
        x1 = max(0, int(u - half_w))
        y1 = max(0, int(v - half_h))
        x2 = min(iw, int(u + half_w))
        y2 = min(ih, int(v + half_h))
        if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
            continue
        best_depth = depth
        best = (cam_idx, x1, y1, x2, y2)
    if best is None:
        return None
    cam_idx, x1, y1, x2, y2 = best
    return images[cam_idx].crop((x1, y1, x2, y2))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reclass_threshold', type=float, default=0.15)
    parser.add_argument('--min_dino_sim', type=float, default=0.3)
    parser.add_argument('--predictions', type=str,
                        default=os.path.join(PROJECT_ROOT, 'research/outputs/mini_preds/predictions_simple.pkl'))
    parser.add_argument('--prototypes', type=str,
                        default=os.path.join(PROJECT_ROOT, 'research/dino_features/prototypes.pt'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'research/outputs/exp_009'))
    parser.add_argument('--max_samples', type=int, default=300)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading predictions...")
    with open(args.predictions, 'rb') as f:
        results = pickle.load(f)
    print(f"  {len(results)} samples", flush=True)

    print("Loading prototypes...")
    proto_data = torch.load(args.prototypes, map_location='cpu', weights_only=True)
    prototypes = torch.nn.functional.normalize(proto_data['prototypes'], dim=1)
    print(f"  {prototypes.shape[0]} class prototypes", flush=True)

    print("Loading val info...")
    with open(os.path.join(PROJECT_ROOT, 'nuscenes_infos_val_sweep.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    val_infos = val_data['infos'][:args.max_samples]
    print(f"  {len(val_infos)} val samples", flush=True)

    print("Loading DINOv3 model...", flush=True)
    model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModel.from_pretrained(model_name, token=HF_TOKEN)
    model.eval().cuda()
    print("  Model ready", flush=True)

    total_reclass = 0
    total_preds = 0
    total_projected = 0
    reclass_log = []
    modified_results = []

    for si in range(min(len(results), len(val_infos))):
        result = results[si]
        info = val_infos[si]

        boxes_np = result['boxes_3d']
        scores_np = result['scores_3d']
        labels_np = result['labels_3d'].copy()
        n = len(boxes_np)
        total_preds += n

        # Preload all 6 camera images for this sample
        cam_params = []
        images = []
        for cam_name in CAM_NAMES:
            ci = info['cams'][cam_name]
            r = np.linalg.inv(ci['sensor2lidar_rotation'])
            t = -ci['sensor2lidar_translation'] @ r.T
            K = ci['cam_intrinsic']
            path = ci['data_path'].replace('data/nuscenes', DATA_ROOT)
            cam_params.append({'r': r, 't': t, 'K': K})
            try:
                images.append(Image.open(path).convert('RGB'))
            except Exception:
                images.append(None)

        # Collect crops for all predictions
        crops_info = []
        for pi in range(n):
            crop = get_crop(boxes_np[pi, :3], boxes_np[pi, 3:6], cam_params, images)
            if crop is not None:
                crops_info.append((pi, crop))
                total_projected += 1

        # Close images
        for img in images:
            if img is not None:
                img.close()

        # Batch DINOv3 feature extraction
        if crops_info:
            all_crops = [c[1] for c in crops_info]
            all_feats = []
            for bs in range(0, len(all_crops), BATCH_SIZE):
                batch = all_crops[bs:bs + BATCH_SIZE]
                inputs = processor(images=batch, return_tensors="pt").to('cuda')
                with torch.no_grad():
                    out = model(**inputs)
                all_feats.append(out.pooler_output.cpu())
            all_feats = torch.cat(all_feats, dim=0)
            all_feats = torch.nn.functional.normalize(all_feats, dim=1)
            sims = all_feats @ prototypes.T  # (N_crops, 10)

            for ci_idx, (pi, _) in enumerate(crops_info):
                sv = sims[ci_idx]
                dino_cls = sv.argmax().item()
                dino_sim = sv[dino_cls].item()
                rac_cls = int(labels_np[pi])
                rac_sim = sv[rac_cls].item()
                adv = dino_sim - rac_sim

                if (dino_cls != rac_cls and adv >= args.reclass_threshold and dino_sim >= args.min_dino_sim):
                    labels_np[pi] = dino_cls
                    total_reclass += 1
                    reclass_log.append({
                        's': si, 'p': pi,
                        'from': CLASS_NAMES[rac_cls], 'to': CLASS_NAMES[dino_cls],
                        'adv': round(adv, 4), 'dsim': round(dino_sim, 4),
                        'rsim': round(rac_sim, 4), 'score': round(float(scores_np[pi]), 4),
                    })

        modified_results.append({
            'boxes_3d': boxes_np,
            'scores_3d': scores_np,
            'labels_3d': labels_np,
        })

        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{len(val_infos)}: {total_projected} projected, {total_reclass} reclassified", flush=True)

    # Save as torch tensors (cross-numpy-version compatible)
    torch_results = []
    for mr in modified_results:
        torch_results.append({
            'boxes_3d': torch.tensor(mr['boxes_3d'], dtype=torch.float32),
            'scores_3d': torch.tensor(mr['scores_3d'], dtype=torch.float32),
            'labels_3d': torch.tensor(mr['labels_3d'], dtype=torch.long),
        })
    torch.save(torch_results, os.path.join(args.output_dir, 'modified_predictions.pt'))
    with open(os.path.join(args.output_dir, 'reclassification_log.json'), 'w') as f:
        json.dump({
            'total_predictions': total_preds, 'total_projected': total_projected,
            'total_reclassified': total_reclass,
            'reclass_threshold': args.reclass_threshold, 'min_dino_sim': args.min_dino_sim,
            'reclassifications': reclass_log,
        }, f, indent=2)

    print(f"\n=== Results ===", flush=True)
    print(f"Predictions: {total_preds}, Projected: {total_projected}, Reclassified: {total_reclass}", flush=True)


if __name__ == '__main__':
    main()
