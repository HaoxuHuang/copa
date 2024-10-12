import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation as R
import queue
import open3d as o3d


class PointCloudGenerator:
    EVERY_N_FRAMES = 10
    ALPHA = 0.4
    DELTA_THRESHOLD = 10  # 1cm
    SAVE_PATH = Path('/root/data/')
    USE_FILTER = True
    # compensate for calibration error
    OFFSET1 = np.array([0.0, 0.0, 0.0])
    OFFSET2 = np.array([0.008, 0.0, 0.01])

    def __init__(self) -> None:
        rospy.init_node('point_cloud_generator', anonymous=True)

        rospy.Subscriber('/cam_1/aligned_depth_to_color/image_raw', Image, self.depth_callback1)
        rospy.Subscriber('/cam_2/aligned_depth_to_color/image_raw', Image, self.depth_callback2)
        rospy.Subscriber('/cam_1/color/image_raw', Image, self.color_callback1)
        rospy.Subscriber('/cam_2/color/image_raw', Image, self.color_callback2)
        rospy.Subscriber('/cam_1/color/camera_info', CameraInfo, self.camera_info_callback1)
        rospy.Subscriber('/cam_2/color/camera_info', CameraInfo, self.camera_info_callback2)
        rospy.Subscriber('/tf_static', TFMessage, self.tf_callback)

        self.bridge = CvBridge()
        self.depth_image1 = None
        self.depth_image2 = None
        self.color_image1 = None
        self.color_image2 = None
        self.camera_info1 = None
        self.camera_info2 = None
        self.cam1_extrinsic = None
        self.cam2_extrinsic = None
        # using count to reduce the number of point cloud generation
        # FIXME: using two seperate counter for color and depth may cause out of sync
        self.color_count1 = 0
        self.color_count2 = 0
        self.depth_count1 = 0
        self.depth_count2 = 0


        # check save path
        if not self.SAVE_PATH.exists():
            self.SAVE_PATH.mkdir(parents=True)
            rospy.loginfo('Create save path {}'.format(self.SAVE_PATH))
        for ddir in ['pointcloud', 'color', 'depth', 'camera_info', 'camera_extrinsic']:
            (self.SAVE_PATH / ddir).mkdir(parents=True, exist_ok=True)
            rospy.loginfo('Create save path {}'.format(self.SAVE_PATH / ddir))

    def depth_callback1(self, data):
        if self.depth_image1 is None:
            self.depth_image1 = self.bridge.imgmsg_to_cv2(data).copy()
            self.depth_image1[np.isnan(self.depth_image1)] = 0
        else:
            # robust exponential moving average
            last_frame = self.bridge.imgmsg_to_cv2(data)
            delta = self.depth_image1 - last_frame
            delta[np.isnan(delta)] = 0
            # print('Max delta: {}'.format(np.max(delta)))
            # print('Mean abs delta: {}'.format(np.mean(np.abs(delta))))
            # delta[delta > self.DELTA_THRESHOLD] = self.DELTA_THRESHOLD
            # delta[delta < -self.DELTA_THRESHOLD] = -self.DELTA_THRESHOLD
            delta[np.abs(delta) > self.DELTA_THRESHOLD] = 0
            self.depth_image1 = self.depth_image1 - self.ALPHA * delta
        self.depth_count1 += 1

    def depth_callback2(self, data):
        if self.depth_image2 is None:
            self.depth_image2 = self.bridge.imgmsg_to_cv2(data).copy()
            self.depth_image2[np.isnan(self.depth_image2)] = 0
        else:
            # robust exponential moving average
            last_frame = self.bridge.imgmsg_to_cv2(data)
            delta = self.depth_image2 - last_frame
            delta[np.isnan(delta)] = 0
            # delta[delta > self.DELTA_THRESHOLD] = self.DELTA_THRESHOLD
            # delta[delta < -self.DELTA_THRESHOLD] = -self.DELTA_THRESHOLD
            delta[np.abs(delta) > self.DELTA_THRESHOLD] = 0
            self.depth_image2 = self.depth_image2 - self.ALPHA * delta

    def color_callback1(self, data):
        if self.color_count1 % self.EVERY_N_FRAMES == 0:
            self.color_image1 = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
        self.color_count1 += 1

    def color_callback2(self, data):
        if self.color_count2 % self.EVERY_N_FRAMES == 0:
            self.color_image2 = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
        self.color_count2 += 1

    def camera_info_callback1(self, data):
        self.camera_info1 = data

    def camera_info_callback2(self, data):
        self.camera_info2 = data

    def tf_callback(self, data):
        # get camera extrinsic
        if data.transforms[0].child_frame_id == 'cam_1_color_optical_frame':
            self.cam1_extrinsic = self.construct_extrinsic_mat(data.transforms[0].transform)
        elif data.transforms[0].child_frame_id == 'cam_2_color_optical_frame':
            self.cam2_extrinsic = self.construct_extrinsic_mat(data.transforms[0].transform)

    def construct_extrinsic_mat(self, transform):
        extrinsic_mat = np.zeros((4, 4))
        extrinsic_mat[3, 3] = 1
        rot_mat = R.from_quat([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]).as_matrix()
        extrinsic_mat[:3, :3] = rot_mat
        extrinsic_mat[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
        print('Extrinsic matrix: ', extrinsic_mat)
        return extrinsic_mat

    def generate_point_cloud(self, drop_nan=True):
        if (self.depth_image1 is None or self.color_image1 is None or self.camera_info1 is None or
            self.depth_image2 is None or self.color_image2 is None or self.camera_info2 is None or
            self.cam1_extrinsic is None or self.cam2_extrinsic is None):
            rospy.logwarn('Data not ready for generating point cloud!')
            return None

        pc1 = self.img_to_pointcloud(self.camera_info1.K, self.depth_image1, self.color_image1, self.cam1_extrinsic, drop_nan)
        pc2 = self.img_to_pointcloud(self.camera_info2.K, self.depth_image2, self.color_image2, self.cam2_extrinsic, drop_nan)
        pc1[:, :3] += self.OFFSET1
        pc2[:, :3] += self.OFFSET2
        point_cloud_with_color = np.concatenate([pc1, pc2], axis=0)
        if self.USE_FILTER:
            point_cloud_with_color = self.filter_point_cloud(point_cloud_with_color)
        # save pointcloud
        np.save(self.SAVE_PATH / 'pointcloud' / 'pointcloud.npy', point_cloud_with_color)
        rospy.loginfo('Point cloud saved to {}'.format(self.SAVE_PATH / 'pointcloud' / 'point_cloud.npy'))
        # save color image
        cv2.imwrite(str(self.SAVE_PATH / 'color' / 'color1.png'), cv2.cvtColor(self.color_image1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.SAVE_PATH / 'color' / 'color2.png'), cv2.cvtColor(self.color_image2, cv2.COLOR_RGB2BGR))
        # save depth image
        np.save(self.SAVE_PATH / 'depth' / f'depth1.npy', self.depth_image1)
        np.save(self.SAVE_PATH / 'depth' / f'depth2.npy', self.depth_image2)
        # save camera info
        np.save(self.SAVE_PATH / 'camera_info' / 'camera_info1.npy', np.array(self.camera_info1.K))
        np.save(self.SAVE_PATH / 'camera_info' / 'camera_info2.npy', np.array(self.camera_info2.K))
        # save camera extrinsic
        np.save(self.SAVE_PATH / 'camera_extrinsic' / 'camera_extrinsic1.npy', self.cam1_extrinsic)
        np.save(self.SAVE_PATH / 'camera_extrinsic' / 'camera_extrinsic2.npy', self.cam2_extrinsic)
        # reset
        self.depth_image1 = None
        self.depth_image2 = None
        self.color_image1 = None
        self.color_image2 = None
        self.color_count1 = 0
        self.color_count2 = 0
        self.depth_count1 = 0
        self.depth_count2 = 0
        return point_cloud_with_color
        
    def img_to_pointcloud(self, K, depth_image, color_image, extrinsic_mat, drop_nan=True):
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]

        h, w = depth_image.shape

        depth = depth_image / 1000.0
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        x = ((xx - cx) * depth / fx).reshape(-1)
        y = ((yy - cy) * depth / fy).reshape(-1)
        z = depth.reshape(-1)
        cloud_color = color_image.reshape(-1, 3)

        if drop_nan:
            valid_idx = np.logical_not(np.isnan(z))
            x = x[valid_idx]
            y = y[valid_idx]
            z = z[valid_idx]
            cloud_color = cloud_color[valid_idx]

        cloud_points = np.stack([x, y, z], axis=1)

        # transform point cloud to base frame
        cloud_points = np.matmul(extrinsic_mat, np.concatenate([cloud_points.T, np.ones((1, cloud_points.shape[0]))], axis=0))
        cloud_points = cloud_points[:3].T
        point_cloud_with_color = np.concatenate([cloud_points, cloud_color], axis=1)

        return point_cloud_with_color
    
    def filter_point_cloud(self, point_cloud_with_color, num_points=15, radius=0.005):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_with_color[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_cloud_with_color[:, 3:] / 255.0)
        ror_pcd, idx = pcd.remove_radius_outlier(num_points, radius)
        return np.concatenate([np.asarray(ror_pcd.points), np.asarray(ror_pcd.colors) * 255.0], axis=1)



if __name__ == '__main__':
    point_cloud_generator = PointCloudGenerator()
    while not rospy.is_shutdown():
        point_cloud_generator.generate_point_cloud()
        rospy.sleep(5)