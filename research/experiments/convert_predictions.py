"""Convert predictions.pkl from mmdet3d format to simple numpy format.
Runs in racformerfix env. Output is loadable in any Python env."""
import os
import sys
import pickle
import numpy as np
import torch

sys.path.insert(0, '/srv/nfs/shared/gnmp/RaCFormer')
import importlib
importlib.import_module('models')
importlib.import_module('loaders')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    with open(args.input, 'rb') as f:
        results = pickle.load(f)

    simple_results = []
    for r in results:
        pred = r['pts_bbox'] if 'pts_bbox' in r else r
        boxes = pred['boxes_3d']
        if hasattr(boxes, 'tensor'):
            boxes_np = boxes.tensor.cpu().numpy()
        else:
            boxes_np = np.array(boxes)
        scores = pred['scores_3d']
        if torch.is_tensor(scores):
            scores_np = scores.cpu().numpy()
        else:
            scores_np = np.array(scores)
        labels = pred['labels_3d']
        if torch.is_tensor(labels):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.array(labels)
        simple_results.append({
            'boxes_3d': boxes_np,
            'scores_3d': scores_np,
            'labels_3d': labels_np,
        })

    print(f"Converted {len(simple_results)} samples")
    with open(args.output, 'wb') as f:
        pickle.dump(simple_results, f)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
