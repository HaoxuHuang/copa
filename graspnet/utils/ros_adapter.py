# convert Grasp from graspnet to grasp message in ROS
import moveit_msgs.msg
import geometry_msgs.msg
import rospy
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from graspnetAPI.grasp import Grasp

def graspnet_to_ros(grasp, extrinsic_mat):
    grasp_w = grasp_cam2world(grasp, extrinsic_mat)
    rot_mat = grasp_w.rotation_matrix
    rot_quat = R.from_matrix(rot_mat).as_quat()
    grasp_msg = moveit_msgs.msg.Grasp()
    # grasp pose
    grasp_msg.grasp_pose.header.frame_id = "panda_link0"
    orientation = geometry_msgs.msg.Quaternion()
    orientation.x = rot_quat[0]
    orientation.y = rot_quat[1]
    orientation.z = rot_quat[2]
    orientation.w = rot_quat[3]
    grasp_msg.grasp_pose.pose.orientation = orientation

    translation = grasp_w.translation # FIXME: need to compensate for the gripper height
    grasp_msg.grasp_pose.pose.position.x = translation[0]
    grasp_msg.grasp_pose.pose.position.y = translation[1]
    grasp_msg.grasp_pose.pose.position.z = translation[2]
    # pre-grasp approach
    grasp_msg.pre_grasp_approach.direction.header.frame_id = "panda_link0"
    # direction is the same as gripper orientation
    direction_vec = np.dot(rot_mat, np.array([1, 0, 0]))
    grasp_msg.pre_grasp_approach.direction.vector.x = direction_vec[0]
    grasp_msg.pre_grasp_approach.direction.vector.y = direction_vec[1]
    grasp_msg.pre_grasp_approach.direction.vector.z = direction_vec[2]
    grasp_msg.pre_grasp_approach.min_distance = 0.095
    grasp_msg.pre_grasp_approach.desired_distance = 0.115
    # post-grasp retreat
    grasp_msg.post_grasp_retreat.direction.header.frame_id = "panda_link0"
    # direction is positive z axis
    grasp_msg.post_grasp_retreat.direction.vector.z = 1.0
    grasp_msg.post_grasp_retreat.min_distance = 0.05
    grasp_msg.post_grasp_retreat.desired_distance = 0.10
    # open gripper
    open_gripper(grasp_msg.pre_grasp_posture)
    # close gripper
    close_gripper(grasp_msg.grasp_posture)

    return grasp_msg


def open_gripper(pre_grasp_posture):
    joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
    joint_values = [0.04, 0.04]
    pre_grasp_posture.joint_names = joint_names
    pre_grasp_posture.points = [moveit_msgs.msg.JointTrajectoryPoint(
        positions=joint_values, time_from_start=rospy.Duration(0.5))]
    
def close_gripper(grasp_posture):
    joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
    joint_values = [0.0, 0.0]
    grasp_posture.joint_names = joint_names
    grasp_posture.points = [moveit_msgs.msg.JointTrajectoryPoint(
        positions=joint_values, time_from_start=rospy.Duration(0.5),
        effort=[1.0, 1.0])]
    

def grasp_cam2world(grasp, extrinsic_mat):
    # convert grasp pose from camera frame to world frame
    # grasp: Grasp object from graspnet
    # extrinsic_mat: 4x4 extrinsic matrix from camera to world
    # return: Grasp object in world frame
    rot_mat = grasp.rotation_matrix
    translation = grasp.translation
    # convert translation
    translation = np.dot(extrinsic_mat, np.append(translation, 1.0))[:3]
    # convert rotation
    rot_mat = np.dot(extrinsic_mat[:3, :3], rot_mat)
    # swap axises
    # rot_mat = np.dot(np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]), rot_mat)
    grasp_w = copy.deepcopy(grasp)
    grasp_w.translation = translation
    grasp_w.rotation_matrix = rot_mat
    return grasp_w
        