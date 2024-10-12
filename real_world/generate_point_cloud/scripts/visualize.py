import open3d as o3d
import numpy as np


# pointcloud = np.load('/root/data/pointcloud/point_cloud.npy')
pointcloud = np.load('/root/data/pointcloud/pointcloud.npy')
print(pointcloud.shape)
pointcloud = pointcloud.reshape(-1, 6)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
pcd.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:] / 255.0)
o3d.visualization.draw_geometries_with_editing([pcd])