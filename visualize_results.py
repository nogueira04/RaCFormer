import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from PIL import Image
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from nuscenes.utils.geometry_utils import view_points

# A mapping from detection name to a color
PALETTE = {
    "car": (255, 158, 0),  # Orange
    "truck": (255, 99, 71),  # Tomato
    "bus": (255, 69, 0),  # OrangeRed
    "trailer": (255, 140, 0),  # DarkOrange
    "construction_vehicle": (233, 150, 70),  # Darksalmon
    "pedestrian": (0, 0, 230),  # Blue
    "motorcycle": (148, 0, 211),  # DarkViolet
    "bicycle": (75, 0, 130),  # Indigo
    "traffic_cone": (255, 255, 0),  # Yellow
    "barrier": (100, 149, 237),  # CornflowerBlue
}
# Convert to 0-1 range for matplotlib
for key, val in PALETTE.items():
    PALETTE[key] = tuple(v / 255.0 for v in val)


def plot_bev(boxes, ego_pose, output_dir, sample_token):
    """
    Renders and saves a Bird's-Eye View plot of detected boxes for a given sample.
    """
    out_path = os.path.join(output_dir, f"{sample_token}_BEV.jpg")
    bev_range = ((-50, 50), (-50, 50)) # X and Y range in meters

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 1. Plot Ego Vehicle
    # nuScenes ego vehicle is approx. 4.084m long and 1.73m wide
    ego_l, ego_w = 4.084, 1.73
    ego_pts = np.array([
        [-ego_l / 2, -ego_w / 2], [-ego_l / 2, ego_w / 2],
        [ego_l / 2, ego_w / 2], [ego_l / 2, -ego_w / 2]
    ])
    ax.fill(ego_pts[:, 0], ego_pts[:, 1], color='red', label='Ego Vehicle')
    
    # 2. Plot Detections
    for box in boxes:
        # Transform box from global to ego frame
        box_ego = box.copy()
        box_ego.translate(-np.array(ego_pose['translation']))
        box_ego.rotate(Quaternion(ego_pose['rotation']).inverse)

        # Get the 2D footprint (bottom corners) of the box
        corners_ego = box_ego.bottom_corners()
        
        color = PALETTE.get(box.name, (0.5, 0.5, 0.5))
        ax.fill(corners_ego[0, :], corners_ego[1, :], color=color, alpha=0.6)

    # 3. Configure Plot
    ax.set_xlim(bev_range[0])
    ax.set_ylim(bev_range[1])
    ax.set_aspect('equal')
    ax.set_title(f'Bird\'s-Eye View: {sample_token}')
    ax.set_xlabel('X-axis (meters)')
    ax.set_ylabel('Y-axis (meters)')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Create a legend for object classes
    legend_elements = [Patch(facecolor=color, label=name) for name, color in PALETTE.items()]
    legend_elements.append(Patch(facecolor='red', label='Ego Vehicle'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_results(nusc_root, nusc_version, results_path, output_dir, score_thresh, num_samples):
    """
    Loads nuScenes data and detection results, and plots 3D bounding boxes on
    the 6 camera images and a BEV plot for a few samples.
    """
    print(f"Loading nuScenes {nusc_version} dataset from {nusc_root}...")
    nusc = NuScenes(version=nusc_version, dataroot=nusc_root, verbose=True)
    
    print(f"Loading results from {results_path}...")
    with open(results_path, 'r') as f:
        results_data = json.load(f)

    sample_results = results_data.get('results', {})
    if not sample_results:
        print("Error: Could not find 'results' key in the JSON file.")
        return

    print(f"Creating output directory at {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    sample_tokens = list(sample_results.keys())
    if num_samples > 0 and len(sample_tokens) > num_samples:
        sample_tokens = sample_tokens[:num_samples]
    
    print(f"Processing {len(sample_tokens)} samples...")
    
    cam_names = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    for sample_token in tqdm(sample_tokens):
        box_dicts = sample_results[sample_token]
        sample = nusc.get('sample', sample_token)
        
        boxes_to_render = []
        for box_dict in box_dicts:
            if box_dict['detection_score'] < score_thresh:
                continue
            
            vel_2d = box_dict.get('velocity', [0.0, 0.0])
            vel_3d = np.array([vel_2d[0], vel_2d[1], 0.0])

            det_box = Box(
                center=box_dict['translation'],
                size=box_dict['size'],
                orientation=Quaternion(box_dict['rotation']),
                name=box_dict['detection_name'],
                score=box_dict['detection_score'],
                velocity=vel_3d
            )
            boxes_to_render.append(det_box)
            
        # --- Call BEV plotting once per sample ---
        # We need the ego_pose of the lidar to align with the point cloud,
        # which is the reference for BEV.
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        ego_pose_for_bev = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        plot_bev(boxes_to_render, ego_pose_for_bev, output_dir, sample_token)
        # --- End BEV plotting call ---

        for cam_name in cam_names:
            cam_token = sample['data'][cam_name]
            out_path = os.path.join(output_dir, f"{sample_token}_{cam_name}.jpg")
            
            sd_record = nusc.get('sample_data', cam_token)
            cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            
            image_path = nusc.get_sample_data_path(cam_token)
            img = Image.open(image_path)

            dpi = 100
            fig, ax = plt.subplots(1, 1, figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

            ax.imshow(img)
            ax.set_xlim(0, img.width)
            ax.set_ylim(img.height, 0)
            ax.axis('off')
            ax.set_aspect('equal')
            
            for box in boxes_to_render:
                box_cam = box.copy()
                box_cam.translate(-np.array(pose_record['translation']))
                box_cam.rotate(Quaternion(pose_record['rotation']).inverse)
                box_cam.translate(-np.array(cs_record['translation']))
                box_cam.rotate(Quaternion(cs_record['rotation']).inverse)

                if np.any(box_cam.corners()[2, :] < 0.1):
                    continue

                color = PALETTE.get(box.name, (0.5, 0.5, 0.5))
                
                box_cam.render(
                    ax,
                    view=cam_intrinsic,
                    normalize=True,
                    colors=(color, color, color),
                    linewidth=2
                )

                corners_3d = box_cam.corners()
                in_front = corners_3d[2, :] > 0.1
                if not np.any(in_front):
                    continue

                corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
                
                min_x = np.min(corners_2d[0, in_front])
                max_x = np.max(corners_2d[0, in_front])
                min_y = np.min(corners_2d[1, in_front])
                
                text_x = (min_x + max_x) / 2
                text_y = min_y - 10
                
                if 0 < text_x < img.width and 0 < text_y < img.height:
                    text = f"{box.name}: {box.score:.2f}"
                    ax.text(
                        text_x, 
                        text_y, 
                        text, 
                        fontsize=10, 
                        color='white', 
                        ha='center',
                        va='bottom',
                        bbox=dict(facecolor=color, alpha=0.6, pad=1.5, edgecolor='none')
                    )

            plt.savefig(out_path, dpi=300, pad_inches=0)
            plt.close(fig)
            
    print(f"\nDone! Saved rendered images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot nuScenes 3D detection results on 2D images and BEV.')
    parser.add_argument('--nusc_root', type=str, required=True, help='Path to the root nuScenes dataset directory.')
    parser.add_argument('--results_path', type=str, default='results_nusc.json', help='Path to the input JSON results file.')
    parser.add_argument('--output_dir', type=str, default='inference_plots', help='Directory to save the rendered images.')
    parser.add_argument('--nusc_version', type=str, default='v1.0-trainval', help='NuScenes dataset version.')
    parser.add_argument('--score_thresh', type=float, default=0.25, help='Score threshold to filter detections.')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to plot (-1 for all).')
    args = parser.parse_args()
    
    try:
        import matplotlib
        import PIL
    except ImportError:
        print("Please install matplotlib and pillow: pip install matplotlib pillow")
        return

    plot_results(
        nusc_root=args.nusc_root,
        nusc_version=args.nusc_version,
        results_path=args.results_path,
        output_dir=args.output_dir,
        score_thresh=args.score_thresh,
        num_samples=args.num_samples
    )

if __name__ == '__main__':
    main()