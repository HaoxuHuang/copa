import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


DATA_ROOT = 'dataset/scenes'
WORKSPACE_MASK = True

def get_net(cfgs):
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(index):
    # load data
    index = str(index).zfill(4)
    color = np.array(Image.open(os.path.join(DATA_ROOT, 'rgb', index+'.png')), dtype=np.float32) / 255.0
    depth = scio.loadmat(os.path.join(DATA_ROOT, 'depth', index+'.mat'))['A']
    meta = scio.loadmat(os.path.join(DATA_ROOT, 'meta', index+'.mat'))
    if WORKSPACE_MASK:
        workspace_mask = scio.loadmat(os.path.join(DATA_ROOT, 'mask', index+'.mat'))['A']
    else:
        workspace_mask = None
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    return process_data(cfgs, color, depth, intrinsic, factor_depth, workspace_mask)  

def process_data(cfgs, color, depth, intrinsic, factor_depth, workspace_mask=None):
    # generate cloud
    camera = CameraInfo(640.0, 480.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    if WORKSPACE_MASK:
        mask = ((workspace_mask > 1) & (depth > 0))
    else:
        mask = depth > 0
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud_viz = o3d.geometry.PointCloud()
    cloud_viz.points = o3d.utility.Vector3dVector(cloud[depth>0].astype(np.float32))
    cloud_viz.colors = o3d.utility.Vector3dVector(color[depth>0].astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_viz

def get_grasps(net, end_points) -> GraspGroup:
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud, cfgs):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(index, cfgs):
    net = get_net(cfgs)
    end_points, cloud = get_and_process_data(index)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points), cfgs)
    vis_grasps(gg, cloud)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
    parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
    parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
    cfgs = parser.parse_args()

    index = 1
    demo(index, cfgs)
