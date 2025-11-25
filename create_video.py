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
import cv2
import random

# New Import for 3D plotting
from mpl_toolkits.mplot3d import Axes3D

# --- Constants and Palette (Unchanged) ---
PALETTE = {
    "car": (255, 158, 0), "truck": (255, 99, 71), "bus": (255, 69, 0),
    "trailer": (255, 140, 0), "construction_vehicle": (233, 150, 70),
    "pedestrian": (0, 0, 230), "motorcycle": (148, 0, 211), "bicycle": (75, 0, 130),
    "traffic_cone": (255, 255, 0), "barrier": (100, 149, 237),
}
for key, val in PALETTE.items():
    PALETTE[key] = tuple(v / 255.0 for v in val)

# --- START: Updated Camera Rendering Function ---

def render_camera_frame(nusc, sample, boxes_to_render, cam_name='CAM_FRONT'):
    """ Renders the camera view, now with a robust fix for projection artifacts. """
    cam_token = sample['data'][cam_name]
    sd_record = nusc.get('sample_data', cam_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    image_path = nusc.get_sample_data_path(cam_token)
    img = Image.open(image_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(img.width / 100, img.height / 100), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    ax.imshow(img)
    ax.set_xlim(0, img.width); ax.set_ylim(img.height, 0); ax.axis('off'); ax.set_aspect('equal')

    for box in boxes_to_render:
        box_cam = box.copy()
        box_cam.translate(-np.array(pose_record['translation']))
        box_cam.rotate(Quaternion(pose_record['rotation']).inverse)
        box_cam.translate(-np.array(cs_record['translation']))
        box_cam.rotate(Quaternion(cs_record['rotation']).inverse)
        
        # --- FIX: Prevent diagonal line artifacts ---
        # This robustly checks if ANY corner of the box is behind the camera.
        # If so, the entire box is skipped, preventing projection errors.
        if np.any(box_cam.corners()[2, :] < 0.1):
            continue
            
        color = PALETTE.get(box.name, (0.5, 0.5, 0.5))
        box_cam.render(ax, view=cam_intrinsic, normalize=True, colors=(color, color, color), linewidth=2)
        
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    img_plot = Image.frombytes("RGB", (ncols, nrows), buf)
    plt.close(fig)
    return img_plot

# --- END: Updated Camera Rendering Function ---


# --- START: Updated 3D View Rendering ---

def draw_3d_box(ax, box, color):
    """ Plots a 3D wireframe of a nuScenes Box object. """
    corners = box.corners()
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    for start, end in edges:
        p1 = corners[:, start]
        p2 = corners[:, end]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=1.5)

def render_bev_frame(boxes_to_render, ego_pose, bev_height):
    """ Renders a clean 3D third-person view from behind the ego vehicle. """
    fig = plt.figure(figsize=(bev_height / 100, bev_height / 100), dpi=100)
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    ego_box = Box([0, 0, 0], [4.084, 1.73, 1.5], Quaternion())
    draw_3d_box(ax, ego_box, color='red')

    for box in boxes_to_render:
        box_ego = box.copy()
        box_ego.translate(-np.array(ego_pose['translation']))
        box_ego.rotate(Quaternion(ego_pose['rotation']).inverse)
        color = PALETTE.get(box.name, (0.5, 0.5, 0.5))
        draw_3d_box(ax, box_ego, color=color)

    ax.set_xlim(-40, 40)
    ax.set_ylim(-10, 70)
    ax.set_zlim(-2, 8)

    # --- FIX: Enforce an equal aspect ratio to prevent distortion ---
    # This ensures the view is not stretched and looks like a true 3rd-person perspective.
    ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))

    # Set the third-person camera angle
    ax.view_init(elev=25, azim=-90)
    ax.axis('off')

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    img_plot = Image.frombytes("RGB", (ncols, nrows), buf)
    plt.close(fig)
    return img_plot

# --- END: Updated 3D View Rendering ---


# --- Main Logic (Unchanged) ---

def create_scene_video(nusc, sample_results, scene_index, output_path, score_thresh, fps):
    scene = nusc.scene[scene_index]
    first_sample_token = scene['first_sample_token']
    sample = nusc.get('sample', first_sample_token)
    
    cam_img = render_camera_frame(nusc, sample, [])
    bev_img = render_bev_frame([], {}, cam_img.height)
    frame_width = cam_img.width + bev_img.width
    frame_height = cam_img.height
    
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    current_sample_token = scene['first_sample_token']
    
    with tqdm(total=scene['nbr_samples'], desc=f"Scene {scene_index} -> {os.path.basename(output_path)}") as pbar:
        while current_sample_token:
            sample = nusc.get('sample', current_sample_token)
            box_dicts = sample_results.get(current_sample_token, [])
            boxes_to_render = []
            for box_dict in box_dicts:
                if box_dict['detection_score'] < score_thresh: continue
                boxes_to_render.append(Box(
                    center=box_dict['translation'], size=box_dict['size'],
                    orientation=Quaternion(box_dict['rotation']),
                    name=box_dict['detection_name'], score=box_dict['detection_score']
                ))

            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            ego_pose_for_bev = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            cam_frame = render_camera_frame(nusc, sample, boxes_to_render)
            bev_frame = render_bev_frame(boxes_to_render, ego_pose_for_bev, cam_frame.height)
            combined_frame = Image.new('RGB', (frame_width, frame_height))
            combined_frame.paste(cam_frame, (0, 0))
            combined_frame.paste(bev_frame, (cam_frame.width, 0))
            video_writer.write(cv2.cvtColor(np.array(combined_frame), cv2.COLOR_RGB2BGR))
            
            current_sample_token = sample['next']
            pbar.update(1)
            
    video_writer.release()

def main():
    parser = argparse.ArgumentParser(description='Create 3D side-by-side videos for random nuScenes scenes.')
    parser.add_argument('--nusc_root', type=str, required=True, help='Path to the root nuScenes dataset directory.')
    parser.add_argument('--results_path', type=str, default='results_nusc.json', help='Path to the input JSON results file.')
    parser.add_argument('--output_dir', type=str, default='video_outputs_3d', help='Directory to save the output MP4 videos.')
    parser.add_argument('--nusc_version', type=str, default='v1.0-trainval', help='NuScenes dataset version.')
    parser.add_argument('--num_videos', type=int, default=3, help='Number of random scene videos to generate.')
    parser.add_argument('--score_thresh', type=float, default=0.25, help='Score threshold to filter detections.')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second for the output video.')
    args = parser.parse_args()
    
    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_root, verbose=False)
    
    with open(args.results_path, 'r') as f:
        results_data = json.load(f)

    sample_results = results_data.get('results', {})
    if not sample_results:
        print("\n[ERROR] Could not find 'results' key in the JSON file.")
        return

    results_tokens = set(sample_results.keys())
    valid_scene_indices = []
    for i, scene in enumerate(nusc.scene):
        if scene['first_sample_token'] in results_tokens:
            valid_scene_indices.append(i)
    
    if not valid_scene_indices:
        print("\n[ERROR] No scenes in the dataset match the sample tokens in your results file.")
        return
        
    num_to_generate = min(args.num_videos, len(valid_scene_indices))
    selected_indices = random.sample(valid_scene_indices, num_to_generate)
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(valid_scene_indices)} valid scenes. Will generate {num_to_generate} videos from indices: {selected_indices}\n")

    for scene_idx in selected_indices:
        output_path = os.path.join(args.output_dir, f"scene_{scene_idx:04d}_video_3rd_person.mp4")
        create_scene_video(
            nusc=nusc, sample_results=sample_results, scene_index=scene_idx,
            output_path=output_path, score_thresh=args.score_thresh, fps=args.fps
        )
    
    print(f"\nAll videos have been generated and saved in '{args.output_dir}'")

if __name__ == '__main__':
    main()