import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import control_msgs.msg
from trajectory_msgs.msg import JointTrajectoryPoint
import tf2_ros
from math import pi, tau, dist, fabs, sin, cos
import numpy as np
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

GRIPPER_HEIGHT = 0.10
GRASP_DEPTH = 0.0
Z_OFFSET = 0.0

def all_close(goal, actual, tolerance=1e-3):
    for i in range(len(goal)):
        if fabs(goal[i] - actual[i]) > tolerance:
            return False
    return True


class GraspDemo(object):
    def __init__(self) -> None:
        super().__init__()
        # First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('robot_homing', anonymous=True)

        # Instantiate a `RobotCommander`_ object. This object is an interface to
        # kinematic model and the current state of the robot:
        robot = moveit_commander.RobotCommander()

        # Instantiate a `PlanningSceneInterface`_ object.  This object is an interface
        # to the world surrounding the robot:
        scene = moveit_commander.PlanningSceneInterface()

        # Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        # to one group of joints.  In this case the group is the joints in the left
        # arm.  This interface can be used to plan and execute motions on the left
        # arm:
        group_name = "panda_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        # Create a `DisplayTrajectory`_ ROS publisher which is used to display
        # trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)
        gripper_publisher = rospy.Publisher(
            '/franka_gripper/gripper_action/goal',
            control_msgs.msg.GripperCommandActionGoal,
            queue_size=20)
        
        # Getting Basic Information
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)
        # end effector link
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)
        # We can also print the name of the end-effector link for this group:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # create collision objects
        # table
        table_pose = geometry_msgs.msg.PoseStamped()
        table_pose.header.frame_id = "panda_link0"
        table_pose.pose.position.x = 0.6
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = -0.305
        table_pose.pose.orientation.w = 1.0
        scene.add_box("table", table_pose, size=(1.0, 1.0, 0.75))
        # walls
        # left wall
        wall1_pose = geometry_msgs.msg.PoseStamped()
        wall1_pose.header.frame_id = "panda_link0"
        wall1_pose.pose.position.x = 0.5
        wall1_pose.pose.position.y = 0.5
        wall1_pose.pose.position.z = 0.5
        wall1_pose.pose.orientation.w = 1.0
        scene.add_box("wall1", wall1_pose, size=(1.0, 0.01, 1.0))
        # right wall
        wall2_pose = geometry_msgs.msg.PoseStamped()
        wall2_pose.header.frame_id = "panda_link0"
        wall2_pose.pose.position.x = 0.5
        wall2_pose.pose.position.y = -0.5
        wall2_pose.pose.position.z = 0.5
        wall2_pose.pose.orientation.w = 1.0
        scene.add_box("wall2", wall2_pose, size=(1.0, 0.01, 1.0))
        # # virtual object to be grasped
        # object_pose = geometry_msgs.msg.PoseStamped()
        # object_pose.header.frame_id = "panda_link0"
        # object_pose.pose.position.x = 0.5
        # object_pose.pose.position.y = 0.0
        # object_pose.pose.position.z = 0.08
        # object_pose.pose.orientation.w = 1.0
        # scene.add_box("object", object_pose, size=(0.02, 0.02, 0.02))

        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.gripper_publisher = gripper_publisher
        self.eef_link = eef_link
        self.group_names = group_names
        # home joint angles
        self.home_pose = [0.000,-0.785,0.0,-2.356,0.0,1.571,0.785]

    def get_robot_states(self):
        joint_state = self.move_group.get_current_joint_values()
        eef_pose = self.move_group.get_current_pose().pose
        return joint_state, eef_pose
    
    def graspnet_to_ros(self, rot_mat, translation):
        # if rot_mat[2, 2] < 0:
        #     # inverse z, y axis of rotation matrix for symmetry
        #     rot_mat[:, 1] *= -1
        #     rot_mat[:, 2] *= -1
        # convert to ROS coordinate system
        mu = np.sin(np.pi/4)
        mat_graspnet_to_ros = np.array([[0.0, 0.0, 1.0],
                                        [-mu, -mu, 0.0],
                                        [mu, -mu, 0.0]])
        rot_mat = np.dot(rot_mat, mat_graspnet_to_ros)
        rot_quat = R.from_matrix(rot_mat).as_quat()
        print('Rotation matrix: ', rot_mat)
        print('Rotation quaternion: ', rot_quat)
        print('Translation matrix: ', translation)
        # grasp pose
        grasp_pose = geometry_msgs.msg.Pose()
        grasp_pose.orientation.x = rot_quat[0]
        grasp_pose.orientation.y = rot_quat[1]
        grasp_pose.orientation.z = rot_quat[2]
        grasp_pose.orientation.w = rot_quat[3]

        # compensate for the gripper height
        translation -= (GRIPPER_HEIGHT - GRASP_DEPTH) * rot_mat[:,2]
        grasp_pose.position.x = translation[0]
        grasp_pose.position.y = translation[1]
        grasp_pose.position.z = translation[2] + Z_OFFSET
        print('Grasp pose: ', grasp_pose)
        # pre-grasp approach
        pre_grasp_pose = copy.deepcopy(grasp_pose)
        # direction is the same as gripper orientation
        direction_vec = rot_mat[:,2]
        pre_grasp_pose.position.x -= direction_vec[0] * 0.115
        pre_grasp_pose.position.y -= direction_vec[1] * 0.115
        pre_grasp_pose.position.z -= direction_vec[2] * 0.115
        print('Pre grasp pose: ', pre_grasp_pose)
        # post-grasp retreat
        post_grasp_retreat = copy.deepcopy(grasp_pose)
        # direction is positive z axis
        post_grasp_retreat.position.z += 0.10
        print('Post grasp retreat: ', post_grasp_retreat)

        return pre_grasp_pose, grasp_pose, post_grasp_retreat

    def grasp(self, grasp_pose):
        rotation = grasp_pose[:3,:3]
        translation = grasp_pose[:3,3]

        pre_grasp, grasp, post_grasp = self.graspnet_to_ros(rotation, translation)
        # set support surface
        self.move_group.set_support_surface_name("table")
        # set velocity
        self.move_group.set_max_velocity_scaling_factor(0.1)    # It seems that this is not working
        # execute grasp
        # to pre grasp
        self.move_group.set_pose_target(pre_grasp)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        if not success:
            rospy.loginfo("Pre grasp failed!")
            return
        # open gripper
        self.open_gripper()
        # to grasp
        # self.move_group.set_pose_target(grasp)
        self.move_group.set_joint_value_target(grasp, True)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        if not success:
            rospy.loginfo("Grasp failed!")
            return
        # close gripper
        self.close_gripper()
        # to post grasp
        self.move_group.set_pose_target(post_grasp)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        if not success:
            rospy.loginfo("Post grasp failed!")
            return

    def open_gripper(self):
        self.gripper_publisher.publish(control_msgs.msg.GripperCommandActionGoal(
            goal=control_msgs.msg.GripperCommandGoal(
                command=control_msgs.msg.GripperCommand(
                    position=0.039, max_effort=5.0))))
        rospy.sleep(2.5)
        
    def close_gripper(self):
        self.gripper_publisher.publish(control_msgs.msg.GripperCommandActionGoal(
            goal=control_msgs.msg.GripperCommandGoal(
                command=control_msgs.msg.GripperCommand(
                    position=0.0, max_effort=100.0))))
        rospy.sleep(2.5)
        
    def go_home(self):
        # get current state
        joint_state, eef_pose = self.get_robot_states()
        # set home pose
        joint_state[0:7] = self.home_pose
        self.move_group.go(joint_state, wait=True)
        self.move_group.stop()
        # check if reached home pose
        current_joints = self.move_group.get_current_joint_values()
        return all_close(joint_state, current_joints, 0.01)




if __name__ == '__main__':
    robot_grasp = GraspDemo()
    success = robot_grasp.go_home()
    if success:
        rospy.loginfo("Homing successful!")
    else:
        rospy.loginfo("Homing failed!")
    
    traj_record_pub = rospy.Publisher('/robot_traj_recorder/command', String, queue_size=1)
    while not rospy.is_shutdown():
        # wait for user input
        key = input("Press enter to grasp ; 'q' to quit: ")
        if key == 'q':
            break
        traj_record_pub.publish('start')
        # load grasp pose
        grasp_pose = np.load('/root/data/grasp.npy')
        robot_grasp.grasp(grasp_pose)
        traj_record_pub.publish('stop')
        # print current state
        joint_state, eef_pose = robot_grasp.get_robot_states()
        rospy.loginfo("Joint states: {}".format(joint_state))
        rospy.loginfo("End effector pose: {}".format(eef_pose))