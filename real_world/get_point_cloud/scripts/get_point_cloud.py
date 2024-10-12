#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from pathlib import Path
import cv2

class PointCloudGenerator:
    def __init__(self):
        rospy.init_node('point_cloud_generator', anonymous=True)

        # 订阅所需的主题
        rospy.Subscriber('/cam_2/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        rospy.Subscriber('/cam_2/color/image_raw', Image, self.color_callback)
        rospy.Subscriber('/cam_2/color/camera_info', CameraInfo, self.camera_info_callback)

        self.bridge = CvBridge()
        self.depth_image = None
        self.color_image = None
        self.camera_info = None
        # import pdb;pdb.set_trace()
        rospy.sleep(1)
        self.generate_point_cloud()
        # 将RGB保存为图片,转化为RGB
        # cv2.imwrite('/root/ws_moveit/src/franka_scripts/get_point_cloud/scripts/color.png', cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('color.png', )

    def depth_callback(self, data):
        self.depth_image = self.bridge.imgmsg_to_cv2(data)

    def color_callback(self, data):
        self.color_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')

    def camera_info_callback(self, data):
        self.camera_info = data

    def hole_filling(self, depth_image: np.ndarray):
        import matplotlib.pyplot as plt
        plt.imshow(depth_image)
        plt.show()
        depth_image_shape = depth_image.shape
        depth_image = depth_image.copy().reshape(-1)
        outliners = depth_image < 0.05
        print('outliners: ', np.shape(outliners))
        outliner_idx = np.where(outliners)
        outliner_idx = np.squeeze(outliner_idx)
        inliner_idx = np.where(~outliners)
        inliner_idx = np.squeeze(inliner_idx)
        print(np.shape(outliner_idx))
        depth_image[outliners] = np.interp(outliner_idx, inliner_idx, depth_image[~outliners])
        depth_image = depth_image.reshape(depth_image_shape)
        plt.imshow(depth_image)
        plt.show()
        return depth_image

    def generate_point_cloud(self):
        if self.depth_image is None or self.color_image is None or self.camera_info is None:
            return None

        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        depth_image = self.hole_filling(self.depth_image)
        h, w = depth_image.shape
        point_cloud_with_color = np.zeros((h, w, 6), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                color = self.color_image[i, j]
                depth = depth_image[i, j] / 1000.0
                # x = (i - cx) * depth / fx
                # y = (j - cy) * depth / fy
                x = (j - cx) * depth / fx
                y = (i - cy) * depth / fy
                z = depth
                point_cloud_with_color[i, j] = [x, y, z, color[0], color[1], color[2]]
        # visualize point_cloud_with_color
        # import pdb;pdb.set_trace()
        import open3d as o3d
        print(point_cloud_with_color[:, :, :3].shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_with_color[:, :, :3].reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(point_cloud_with_color[:, :, 3:].reshape(-1, 3) / 255.0)
        task_name = 'pour_water'
        data_dir = Path('/root/ws_moveit/src/env_pics') / task_name
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f'{task_name}.npy', point_cloud_with_color)
        cv2.imwrite(str(data_dir / f'{task_name}.png'), cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR))
        o3d.visualization.draw_geometries([pcd])
        
             

        return point_cloud_with_color

if __name__ == '__main__':
    generator = PointCloudGenerator()
    rospy.spin()