"""Compute DINOv3 class prototypes from training GT 2D crops.

Runs in dino_extract env (Python 3.10+).
Uses first N_SAMPLES training samples, crops GT 2D bboxes, extracts DINOv3 CLS features,
averages per class to produce 10 prototype vectors (768-dim each).

Output: research/dino_features/prototypes.pt
  dict with keys: 'prototypes' (10, 768), 'class_names' list, 'counts' per class
"""
import os
import sys
import pickle
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from collections import defaultdict

PROJECT_ROOT = '/srv/nfs/shared/gnmp/RaCFormer'
DATA_ROOT = '/mnt/nfs/shared/nuscenes'
N_SAMPLES = 500  # training samples to use
MIN_BOX_SIZE = 20  # minimum crop size in pixels
BATCH_SIZE = 64
HF_TOKEN = os.environ.get('HF_TOKEN', None)

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def load_and_crop(img_path, bbox):
    """Load image and crop bbox region. Returns PIL image or None if too small."""
    img = Image.open(img_path).convert('RGB')
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h = x2 - x1, y2 - y1
    if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
        return None
    # Clamp to image bounds
    iw, ih = img.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    if x2 - x1 < MIN_BOX_SIZE or y2 - y1 < MIN_BOX_SIZE:
        return None
    return img.crop((x1, y1, x2, y2))


def main():
    print(f"Loading training pkl...")
    with open(os.path.join(PROJECT_ROOT, 'nuscenes_infos_train_sweep.pkl'), 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'][:N_SAMPLES]
    print(f"Using {len(infos)} training samples")

    print(f"Loading DINOv3 model...")
    model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModel.from_pretrained(model_name, token=HF_TOKEN)
    model.eval().cuda()
    print(f"Model loaded on GPU")

    # Collect all crops per class
    crops_by_class = defaultdict(list)
    total_crops = 0
    for idx, info in enumerate(infos):
        if idx % 100 == 0:
            print(f"  Collecting crops: {idx}/{len(infos)} samples, {total_crops} crops so far")
        bboxes2d = info.get('bboxes2d', [])
        labels2d = info.get('labels2d', [])
        if not bboxes2d:
            continue
        for cam_idx, cam_name in enumerate(CAM_NAMES):
            if cam_idx >= len(bboxes2d):
                break
            cam_bboxes = bboxes2d[cam_idx]
            cam_labels = labels2d[cam_idx]
            if len(cam_bboxes) == 0:
                continue
            # Get image path
            cam_info = info['cams'].get(cam_name)
            if cam_info is None:
                continue
            data_path = cam_info['data_path']
            # data_path is like "data/nuscenes/samples/CAM_FRONT/..."
            # Replace data/nuscenes with actual DATA_ROOT
            img_path = data_path.replace('data/nuscenes', DATA_ROOT)
            if not os.path.exists(img_path):
                continue
            for bbox, label in zip(cam_bboxes, cam_labels):
                label = int(label)
                if label < 0 or label >= len(CLASS_NAMES):
                    continue
                crop = load_and_crop(img_path, bbox)
                if crop is not None:
                    crops_by_class[label].append(crop)
                    total_crops += 1

    print(f"Total crops collected: {total_crops}")
    for cls_idx in range(len(CLASS_NAMES)):
        print(f"  {CLASS_NAMES[cls_idx]}: {len(crops_by_class[cls_idx])}")

    # Extract features per class
    prototypes = torch.zeros(len(CLASS_NAMES), 768)
    counts = torch.zeros(len(CLASS_NAMES), dtype=torch.long)

    for cls_idx in range(len(CLASS_NAMES)):
        crops = crops_by_class[cls_idx]
        if len(crops) == 0:
            print(f"  WARNING: No crops for {CLASS_NAMES[cls_idx]}")
            continue
        print(f"  Extracting features for {CLASS_NAMES[cls_idx]} ({len(crops)} crops)...")
        all_features = []
        for batch_start in range(0, len(crops), BATCH_SIZE):
            batch_crops = crops[batch_start:batch_start + BATCH_SIZE]
            inputs = processor(images=batch_crops, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = model(**inputs)
            cls_features = outputs.pooler_output  # (batch, 768)
            all_features.append(cls_features.cpu())
        all_features = torch.cat(all_features, dim=0)  # (N, 768)
        # L2 normalize before averaging for better prototype
        all_features = torch.nn.functional.normalize(all_features, dim=1)
        prototype = all_features.mean(dim=0)
        prototype = torch.nn.functional.normalize(prototype, dim=0)
        prototypes[cls_idx] = prototype
        counts[cls_idx] = len(crops)

    # Save
    out_dir = os.path.join(PROJECT_ROOT, 'research', 'dino_features')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'prototypes.pt')
    torch.save({
        'prototypes': prototypes,
        'class_names': CLASS_NAMES,
        'counts': counts,
        'n_samples': N_SAMPLES,
        'model': model_name,
    }, out_path)
    print(f"Saved prototypes to {out_path}")
    print("Done!")


if __name__ == '__main__':
    main()
