
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import pickle
import shutil

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}

classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def get_gt(info):
    """Generate GT boxes and labels in ego coordinates."""
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT']['ego2global_translation']

    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse

    gt_boxes = []
    gt_labels = []

    for ann_info in info['ann_infos']:
        category = map_name_from_general_to_detection[
            ann_info['category_name']
        ]

        if (category not in classes or
                ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue

        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )

        box.translate(trans)
        box.rotate(rot)

        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])

        gt_box = np.concatenate([
            box_xyz, box_dxdydz, box_yaw, box_velo
        ])

        gt_boxes.append(gt_box)
        gt_labels.append(classes.index(category))

    return gt_boxes, gt_labels


def nuscenes_data_prep(
    root_path,
    info_prefix,
    version,
    max_sweeps=0,
):
    """Generate base nuScenes info files."""
    nuscenes_converter.create_nuscenes_infos(
        root_path,
        info_prefix,
        version=version,
        max_sweeps=max_sweeps,
    )


def move_info_files(root_path, info_dir, info_prefix):
    """Move generated PKL files to the requested output directory."""
    os.makedirs(info_dir, exist_ok=True)

    for split in ['train', 'val']:
        filename = f'{info_prefix}_infos_{split}.pkl'
        src = os.path.join(root_path, filename)
        dst = os.path.join(info_dir, filename)

        if not os.path.isfile(src):
            raise FileNotFoundError(
                f'Generated info file not found: {src}'
            )

        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.move(src, dst)

        print(f'Info file: {dst}')


def add_ann_adj_info(
    root_path,
    info_dir,
    info_prefix,
    version,
    occ_root='gts',
):
    """Add GT annotations, scene token and occupancy path."""
    nuscenes = NuScenes(
        version=version,
        dataroot=root_path,
        verbose=True,
    )

    for split in ['train', 'val']:
        info_path = os.path.join(
            info_dir,
            f'{info_prefix}_infos_{split}.pkl'
        )

        with open(info_path, 'rb') as f:
            dataset = pickle.load(f)

        total = len(dataset['infos'])
        print(f'Processing {split}: {total} samples')

        for idx, info in enumerate(dataset['infos']):
            if idx % 10 == 0:
                print(f'{idx}/{total}')

            sample = nuscenes.get('sample', info['token'])

            ann_infos = []
            for ann in sample['anns']:
                ann_info = nuscenes.get(
                    'sample_annotation', ann
                ).copy()

                velocity = nuscenes.box_velocity(
                    ann_info['token']
                )

                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)

                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)

            # Keep original behavior: store (boxes, labels).
            info['ann_infos'] = ann_infos
            info['ann_infos'] = get_gt(info)

            info['scene_token'] = sample['scene_token']

            scene = nuscenes.get(
                'scene', sample['scene_token']
            )

            info['occ_path'] = os.path.join(
                occ_root,
                scene['name'],
                info['token'],
            )

        with open(info_path, 'wb') as f:
            pickle.dump(dataset, f)

        print(f'Saved: {info_path}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare nuScenes infos with annotations'
    )

    parser.add_argument(
        '--root-path',
        type=str,
        default='data/nuscenes',
        help='nuScenes dataset root directory',
    )

    parser.add_argument(
        '--info-dir',
        type=str,
        default=None,
        help='Output directory for PKL files; '
             'defaults to root-path',
    )

    parser.add_argument(
        '--info-prefix',
        type=str,
        default='bevdetv2-nuscenes',
        help='Prefix of generated info files',
    )

    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-trainval',
        choices=['v1.0-trainval', 'v1.0-mini'],
        help='nuScenes dataset version',
    )

    parser.add_argument(
        '--max-sweeps',
        type=int,
        default=0,
        help='Number of previous LiDAR sweeps',
    )

    parser.add_argument(
        '--occ-root',
        type=str,
        default='gts',
        help='Occupancy annotation path prefix stored in PKL',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    root_path = os.path.abspath(args.root_path)
    info_dir = os.path.abspath(
        args.info_dir or args.root_path
    )

    os.makedirs(info_dir, exist_ok=True)

    print('Dataset root:', root_path)
    print('Info output:', info_dir)
    print('Version:', args.version)
    print('Info prefix:', args.info_prefix)

    # Step 1: Generate base infos.
    nuscenes_data_prep(
        root_path=root_path,
        info_prefix=args.info_prefix,
        version=args.version,
        max_sweeps=args.max_sweeps,
    )

    # Step 2: Move infos to output directory.
    move_info_files(
        root_path=root_path,
        info_dir=info_dir,
        info_prefix=args.info_prefix,
    )

    # Step 3: Add annotations and occupancy paths.
    add_ann_adj_info(
        root_path=root_path,
        info_dir=info_dir,
        info_prefix=args.info_prefix,
        version=args.version,
        occ_root=args.occ_root,
    )

    print('nuScenes data preparation completed.')


if __name__ == '__main__':
    main()
