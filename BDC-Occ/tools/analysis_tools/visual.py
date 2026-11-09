
import os
import argparse
import pickle

import numpy as np
import mayavi.mlab as mlab


# ============================================================
# Semantic Occupancy color map
# ============================================================

COLORS = np.array(
    [
        [0, 0, 0, 255],         # 0: others
        [255, 120, 50, 255],   # 1: barrier
        [255, 192, 203, 255],  # 2: bicycle
        [255, 255, 0, 255],    # 3: bus
        [0, 150, 245, 255],    # 4: car
        [0, 255, 255, 255],    # 5: construction_vehicle
        [200, 180, 0, 255],    # 6: motorcycle
        [255, 0, 0, 255],      # 7: pedestrian
        [255, 240, 150, 255],  # 8: traffic_cone
        [135, 60, 0, 255],     # 9: trailer
        [160, 32, 240, 255],   # 10: truck
        [255, 0, 255, 255],    # 11: driveable_surface
        [139, 137, 137, 255],  # 12: other_flat
        [75, 0, 75, 255],      # 13: sidewalk
        [150, 240, 80, 255],   # 14: terrain
        [230, 230, 250, 255],  # 15: manmade
        [0, 175, 0, 255],      # 16: vegetation
        [0, 255, 127, 255],    # 17: empty (not rendered)
        [255, 99, 71, 255],
        [0, 191, 255, 255],
    ],
    dtype=np.uint8,
)

FREE_LABEL = 17


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize BDC-Occ semantic occupancy predictions"
    )

    parser.add_argument(
        "prediction",
        type=str,
        help="Path to per-frame prediction: scene/sample_token.pkl"
    )

    parser.add_argument(
        "--occ-root",
        type=str,
        default="/home/wys/wangyushen/data/Datasets/nuscenes/gts",
        help="Occ3D GT root directory"
    )

    parser.add_argument(
        "--camera-mask",
        action="store_true",
        help="Only visualize non-empty voxels inside mask_camera"
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.4,
        help="Voxel size in meters"
    )

    parser.add_argument(
        "--pc-range",
        type=float,
        nargs=6,
        default=[-40, -40, -1, 40, 40, 5.4],
        metavar=(
            "XMIN", "YMIN", "ZMIN",
            "XMAX", "YMAX", "ZMAX"
        ),
        help="Point cloud range"
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional PNG output path"
    )

    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Render without interactive window"
    )

    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=[1600, 900],
        metavar=("WIDTH", "HEIGHT")
    )

    parser.add_argument(
        "--opacity",
        type=float,
        default=1.0
    )

    return parser.parse_args()


def load_prediction(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        pred_occ = pickle.load(f)

    if not isinstance(pred_occ, np.ndarray):
        raise TypeError(
            f"Expected numpy.ndarray, got {type(pred_occ)}"
        )

    if pred_occ.ndim != 3:
        raise ValueError(
            f"Expected 3D occupancy grid, got {pred_occ.shape}"
        )

    if not np.issubdtype(pred_occ.dtype, np.integer):
        raise TypeError(
            f"Expected integer labels, got {pred_occ.dtype}"
        )

    if pred_occ.size == 0:
        raise ValueError("Prediction is empty")

    if pred_occ.min() < 0 or pred_occ.max() >= len(COLORS):
        raise ValueError(
            "Prediction contains labels outside the color map: "
            f"[{pred_occ.min()}, {pred_occ.max()}]"
        )

    return pred_occ


def load_camera_mask(prediction_path, occ_root, expected_shape):
    """
    Automatically resolve:

    Prediction:
        predictions/scene-xxxx/sample_token.pkl

    Occ3D GT:
        gts/scene-xxxx/sample_token/labels.npz
    """

    prediction_path = os.path.abspath(prediction_path)

    scene_name = os.path.basename(
        os.path.dirname(prediction_path)
    )

    sample_token = os.path.splitext(
        os.path.basename(prediction_path)
    )[0]

    gt_path = os.path.join(
        occ_root,
        scene_name,
        sample_token,
        "labels.npz"
    )

    if not os.path.isfile(gt_path):
        raise FileNotFoundError(
            f"Cannot find corresponding Occ3D GT: {gt_path}"
        )

    with np.load(gt_path) as gt_data:
        if "mask_camera" not in gt_data.files:
            raise KeyError(
                f"mask_camera not found in {gt_path}"
            )

        camera_mask = gt_data["mask_camera"].astype(bool)

    if camera_mask.shape != expected_shape:
        raise ValueError(
            f"Camera mask shape {camera_mask.shape} "
            f"does not match prediction shape {expected_shape}"
        )

    print(f"Scene: {scene_name}")
    print(f"Sample token: {sample_token}")
    print(f"GT path: {gt_path}")
    print(f"Camera observable voxels: {camera_mask.sum()}")

    return camera_mask


def occupancy_to_points(
    pred_occ,
    voxel_size,
    pc_range,
    camera_mask=None
):
    """
    Convert occupancy grid (X, Y, Z) to voxel-center coordinates.

    Only retain:
      1. pred_occ != 17
      2. mask_camera == True, if camera_mask is provided

    Returns:
        points: (N, 3)
        labels: (N,)
    """

    occupied_mask = pred_occ != FREE_LABEL

    if camera_mask is not None:
        if camera_mask.shape != pred_occ.shape:
            raise ValueError(
                f"Camera mask shape {camera_mask.shape} "
                f"does not match prediction shape {pred_occ.shape}"
            )

        occupied_mask = np.logical_and(
            occupied_mask,
            camera_mask
        )

    indices = np.argwhere(occupied_mask)

    if len(indices) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.uint8)
        )

    labels = pred_occ[occupied_mask].astype(np.uint8)

    points = (
        (indices.astype(np.float32) + 0.5) * voxel_size
        + np.asarray(pc_range[:3], dtype=np.float32)
    )

    return points, labels


def visualize(points, labels, args):
    if len(points) == 0:
        print("No occupied voxels to visualize.")
        return

    if args.offscreen:
        mlab.options.offscreen = True

    figure = mlab.figure(
        size=tuple(args.window_size),
        bgcolor=(1, 1, 1),
        fgcolor=(0, 0, 0)
    )

    plot = mlab.points3d(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        labels.astype(np.float32),
        mode="cube",
        scale_factor=args.voxel_size * 0.95,
        opacity=args.opacity,
        vmin=0,
        vmax=len(COLORS) - 1,
        figure=figure
    )

    # Keep every voxel the same size
    plot.glyph.glyph.scale_mode = "data_scaling_off"

    # Apply semantic class colors
    plot.module_manager.scalar_lut_manager.lut.table = COLORS

    mlab.view(
        azimuth=45,
        elevation=65,
        distance="auto",
        focalpoint=(0, 0, 0),
        figure=figure
    )

    mlab.draw(figure=figure)

    if args.save_path:
        save_path = os.path.abspath(args.save_path)

        os.makedirs(
            os.path.dirname(save_path),
            exist_ok=True
        )

        mlab.savefig(
            save_path,
            figure=figure,
            size=tuple(args.window_size)
        )

        print(f"Visualization saved to: {save_path}")

    if not args.offscreen:
        mlab.show()
    else:
        mlab.close(figure)


def main():
    args = parse_args()

    if args.offscreen and not args.save_path:
        raise ValueError(
            "--offscreen requires --save-path"
        )

    pred_occ = load_prediction(args.prediction)

    print(f"Prediction: {args.prediction}")
    print(f"Shape: {pred_occ.shape}")
    print(f"Dtype: {pred_occ.dtype}")
    print(f"Classes: {np.unique(pred_occ)}")

    expected_shape = np.rint(
        (
            np.asarray(args.pc_range[3:], dtype=np.float64)
            - np.asarray(args.pc_range[:3], dtype=np.float64)
        ) / args.voxel_size
    ).astype(int)

    if not np.array_equal(expected_shape, pred_occ.shape):
        raise ValueError(
            f"Grid shape {pred_occ.shape} does not match "
            f"voxel size and pc_range: {tuple(expected_shape)}"
        )

    camera_mask = None

    if args.camera_mask:
        camera_mask = load_camera_mask(
            prediction_path=args.prediction,
            occ_root=args.occ_root,
            expected_shape=pred_occ.shape
        )

    points, labels = occupancy_to_points(
        pred_occ,
        args.voxel_size,
        args.pc_range,
        camera_mask=camera_mask
    )

    print(f"Occupied voxels to render: {len(points)}")

    visualize(points, labels, args)


if __name__ == "__main__":
    main()
