import sys
import os
import argparse
import time
import math
import cv2
import shutil
import numpy as np
import open3d as o3d
import torch
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup
from utils.grasp_projection import filter_grasp_in_mask, look_at


GRASP_GAP = 0.005
GRASP_DEPTH = 0.0075
WORKSPACE_MASK = True
TABLE_HEIGHT = 0.065
TABLE_WIDTH = 0.8

GREEN_HSV_BOUND = np.array([[50, 50, 50], [90, 255, 255]])
VIRTUAL_CAMERA = look_at(np.array([0.3, 0.0, 0.5]), np.array([0.3, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
# VIRTUAL_CAMERA = look_at(np.array([0.3, 0.0, 0.3]), np.array([0.3, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
O3D_AXIS = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])

parser = argparse.ArgumentParser()
parser.add_argument('--candidates_path', required=True, help='Grasp candidates path')
parser.add_argument('--num_point', type=int, default=200000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--pointcloud_path', type=Path, default='data/pointcloud/pointcloud.npy', help='Path to point cloud data')
parser.add_argument('--mask_path', type=Path, default='data/mask.npy', help='Path to mask data')
parser.add_argument('--camera_info_path', type=Path, default='data/camera_info', help='Path to camera info')
parser.add_argument('--extrinsic_path', type=Path, default='data/camera_extrinsic', help='Path to extrinsic matrix')
parser.add_argument('--output_path', type=Path, default='data/grasp.npy', help='Path to output grasp pose')
parser.add_argument('--cam', type=str, default='cam2', help='Which camera to use [default: cam2]')
cfgs = parser.parse_args()


def collision_detection(gg, cloud, cfgs):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def pack_data(cloud):
    '''Args:
        cloud: point cloud data N x [x,y,z,r,g,b]
    '''
    cloud_masked = filter_cloud(cloud)
    # transform to virtual camera
    cloud_masked[:, :3] = np.dot(VIRTUAL_CAMERA[:3, :3], cloud_masked[:, :3].T).T + VIRTUAL_CAMERA[:3, 3]
    cloud[:, :3] = np.dot(VIRTUAL_CAMERA[:3, :3], cloud[:, :3].T).T + VIRTUAL_CAMERA[:3, 3]
    # convert data
    cloud_viz = o3d.geometry.PointCloud()
    cloud_viz.points = o3d.utility.Vector3dVector(cloud[:, :3].astype(np.float32))
    cloud_viz.colors = o3d.utility.Vector3dVector(cloud[:, 3:].astype(np.float32) / 255.0)
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_masked[np.newaxis, :, :3].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = cloud_masked[:, 3:] / 255.0
    cloud_sampled_viz = o3d.geometry.PointCloud()
    cloud_sampled_viz.points = o3d.utility.Vector3dVector(cloud_masked[:, :3].astype(np.float32))
    cloud_sampled_viz.colors = o3d.utility.Vector3dVector(cloud_masked[:, 3:].astype(np.float32) / 255.0)

    return end_points, cloud_viz, cloud_sampled_viz


def filter_cloud(cloud):
    # filter according to y value
    mask_y = np.logical_and(cloud[:, 1] > -TABLE_HEIGHT/2, cloud[:, 1] < TABLE_WIDTH/2)
    # filter according to z value
    mask_z = np.logical_and(cloud[:, 2] > TABLE_HEIGHT, cloud[:, 2] < 0.5)

    # mask = np.logical_and(mask_hsv==0, np.logical_and(mask_y, mask_z))
    mask = np.logical_and(mask_y, mask_z)
    filtered_cloud = cloud[mask]
    # segment supporting plane
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_cloud[:, :3].astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(filtered_cloud[:, 3:].astype(np.float32) / 255.0)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.0075,
                                            ransac_n=6,
                                            num_iterations=1000)
    inliner_cloud = pcd.select_by_index(inliers)
    inliner_cloud.paint_uniform_color([1.0, 0, 0])
    outliner_cloud = pcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([inliner_cloud, outliner_cloud])
    filtered_cloud = np.concatenate([np.array(outliner_cloud.points), np.array(outliner_cloud.colors)], axis=1)
    filtered_cloud[:, 3:] = filtered_cloud[:, 3:] * 255.0
    # filter out the green background
    hsv = cv2.cvtColor(filtered_cloud[np.newaxis, :, 3:].astype(np.uint8), cv2.COLOR_RGB2HSV)
    mask_hsv = cv2.inRange(hsv, GREEN_HSV_BOUND[0], GREEN_HSV_BOUND[1]).squeeze()
    filtered_cloud = filtered_cloud[mask_hsv==0]
    return filtered_cloud


def save_grasp(grasp_pose):
    grasp_array = np.zeros((4, 4))
    grasp_array[:3, :3] = grasp_pose.rotation_matrix
    grasp_array[:3, 3] = grasp_pose.translation.squeeze()
    grasp_array[3, 3] = 1.0
    # transform the grasp pose from virtual camera frame to world frame
    grasp_array = np.dot(np.linalg.inv(VIRTUAL_CAMERA), grasp_array)
    np.save(cfgs.output_path, grasp_array)
    np.save(cfgs.output_path.parent / 'grasp_graspnet.npy', grasp_pose.grasp_array)
    print('Grasp pose saved to {}'.format(cfgs.output_path))

def run():
    # x y z width in meter, angle in radian
    pointcloud_path = cfgs.pointcloud_path
    mask_path = cfgs.mask_path

    print(VIRTUAL_CAMERA)

    # load camera info
    if cfgs.cam == 'cam1':
        intrinsic = np.load(cfgs.camera_info_path / 'camera_info1.npy')
        extrinsic = np.linalg.inv(np.load(cfgs.extrinsic_path / 'camera_extrinsic1.npy'))
    elif cfgs.cam == 'cam2':
        intrinsic = np.load(cfgs.camera_info_path / 'camera_info2.npy')
        extrinsic = np.linalg.inv(np.load(cfgs.extrinsic_path / 'camera_extrinsic2.npy'))
    factor_depth = 1000

    while True:
        # get keyboard input
        print('Press \'c\' to continue, \'q\' to quit')
        key = input()
        if key == 'q':
            break
        elif key == 'c':
            # load point cloud
            pointcloud = np.load(pointcloud_path)
            shutil.copyfile(pointcloud_path, cfgs.output_path.parent / 'pointcloud_frozen.npy')
            endpoints, cloud_viz, cloud_sampled_viz = pack_data(pointcloud)
            # load mask
            mask = np.load(mask_path)
            # get grasps
            grasps = GraspGroup(np.load(cfgs.candidates_path))
            print('Number of grasps: {}'.format(len(grasps)))
            # visualize all grasps
            # grasps.sort_by_score()
            # gripper = grasps.to_open3d_geometry_list()
            # o3d.visualization.draw_geometries([cloud_sampled_viz, *gripper])
            # filter collided grasps
            if cfgs.collision_thresh > 0:
                grasps = collision_detection(grasps, np.array(cloud_viz.points), cfgs)
            # visualize collision-free grasps
            gripper = grasps.to_open3d_geometry_list()[:50]
            for grp in gripper:
                grp.paint_uniform_color([0.058, 0.705, 0.976])
            o3d.visualization.draw_geometries([cloud_sampled_viz, *gripper, O3D_AXIS])
            # filter grasps in mask
            grasps = filter_grasp_in_mask(grasps, mask, intrinsic, extrinsic@np.linalg.inv(VIRTUAL_CAMERA), factor_depth)
            if len(grasps) == 0:
                print('Warning: No grasp in mask!')
                continue
            # visualize
            gripper = grasps.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([cloud_viz, *gripper, O3D_AXIS])
            # choose the best grasp
            grasps.sort_by_score()
            grasp_pose = grasps[0]
            # visualize the chosen grasp
            gripper = grasp_pose.to_open3d_geometry()
            gripper.paint_uniform_color([0.058, 0.705, 0.976])
            center_pt = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            center_pt.translate(grasp_pose.translation.squeeze())
            print('Grasp pose: {}'.format(grasp_pose))
            o3d.visualization.draw_geometries([cloud_viz, gripper, center_pt, O3D_AXIS])

            # save grasp pose
            save_grasp(grasp_pose)

    print('Done!')

if __name__ == '__main__':
    run()