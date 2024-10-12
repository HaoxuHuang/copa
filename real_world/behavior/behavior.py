import pickle
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


def all_close(goal, actual, tolerance=1e-3):
    for i in range(len(goal)):
        if fabs(goal[i] - actual[i]) > tolerance:
            return False
    return True

def pose_msg_to_matrix(pose_msg):
    # convert pose msg to matrix
    pose_matrix = np.zeros((4, 4))
    pose_matrix[0:3, 0:3] = R.from_quat(
        [pose_msg.orientation.x,
         pose_msg.orientation.y,
         pose_msg.orientation.z,
         pose_msg.orientation.w]).as_matrix()
    pose_matrix[0:3, 3] = [pose_msg.position.x,
                           pose_msg.position.y,
                           pose_msg.position.z]
    pose_matrix[3, 3] = 1.0
    return pose_matrix

def matrix_to_pose_msg(pose_matrix):
    pose_msg = geometry_msgs.msg.Pose()
    pose_msg.position.x = pose_matrix[0, 3]
    pose_msg.position.y = pose_matrix[1, 3]
    pose_msg.position.z = pose_matrix[2, 3]
    rot_quat = R.from_matrix(pose_matrix[0:3, 0:3]).as_quat()
    pose_msg.orientation.x = rot_quat[0]
    pose_msg.orientation.y = rot_quat[1]
    pose_msg.orientation.z = rot_quat[2]
    pose_msg.orientation.w = rot_quat[3]
    return pose_msg

class BehaviorDemo(object):
    def __init__(self) -> None:
        super().__init__()
        # First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('behavior_demo', anonymous=True)

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
    
    def act(self, pose_trans, interactive=True):
        joint_state, eef_pose = self.get_robot_states()
        # get target pose
        current_pose = pose_msg_to_matrix(eef_pose)
        print('Current pose: ', current_pose)
        print('Current joint state: ', joint_state)
        target_pose = pose_trans @ current_pose
        print('Target pose: ', target_pose)
        pose_msg = matrix_to_pose_msg(target_pose)
        print('Target pose msg: ', pose_msg)
        # plan and execute
        self.move_group.set_joint_value_target(pose_msg, True)  # use approximate IK
        success, plan, _, _ = self.move_group.plan()
        if not success:
            return False
        if interactive:
            key = input("Press enter to execute plan ; 'q' to quit: ")
            if key == 'q':
                return False
        self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        # check if reached target pose
        current_pose = self.move_group.get_current_pose().pose
        current_pose = pose_msg_to_matrix(current_pose)
        return np.allclose(current_pose, target_pose, atol=0.01)

    def post_act(self, action, delta):
        if action == 'open':
            self.open_gripper()
            return True
        if action == 'close':
            self.close_gripper()
            return True
        if action == 'move_down':
            self.move_group.shift_pose_target(2, -delta, self.eef_link)
        elif action == 'move_up':
            self.move_group.shift_pose_target(2, delta, self.eef_link)
        elif action == 'move_left':
            self.move_group.shift_pose_target(0, delta, self.eef_link)
        elif action == 'move_right':
            self.move_group.shift_pose_target(0, -delta, self.eef_link)
        elif action == 'forward':
            self.move_group.shift_pose_target(1, delta, self.eef_link)
        elif action == 'backward':
            self.move_group.shift_pose_target(1, -delta, self.eef_link)
        elif action == 'scoop':
            # rotate around given axis
            joint_state, eef_pose = self.get_robot_states()
            eef_pose = pose_msg_to_matrix(eef_pose)
            print('Current pose: ', eef_pose)
            mu = np.sin(np.pi/4)
            axis = eef_pose[:3, :3] @ np.array([mu, mu, 0.0]).T
            # get eef pose in rotation center frame
            eef2center = np.eye(4)
            eef2center[:3, 3] = np.array([-0.12, 0.10, 0.0])
            eef2center[:3, :3] = np.array([[-mu, -mu, 0.0], 
                                           [0.0, 0.0, -1.0],
                                           [mu, -mu, 0.0]])
            # get transform matrix
            rot_mat = R.from_euler('x', delta).as_matrix()
            rot_trans = np.eye(4)
            rot_trans[:3, :3] = rot_mat
            # get new eef pose in world frame
            center2eef = np.linalg.inv(eef2center)
            new_eef_pose = eef_pose @ center2eef @ rot_trans @ eef2center
            print('New pose: ', new_eef_pose)
            # set target pose
            pose_msg = matrix_to_pose_msg(new_eef_pose)
            self.move_group.set_pose_target(pose_msg)
            # plan and execute
            success, plan, _, _ = self.move_group.plan()
            if not success:
                return False
            self.move_group.execute(plan, wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            return True
        elif action == 'rotate':
            # rotate around given axis
            joint_state, eef_pose = self.get_robot_states()
            eef_pose = pose_msg_to_matrix(eef_pose)
            print('Current pose: ', eef_pose)
            mu = np.sin(np.pi/4)
            axis = eef_pose[:3, :3] @ np.array([mu, mu, 0.0]).T
            # get eef pose in rotation center frame
            eef2center = np.eye(4)
            eef2center[:3, 3] = np.array([0, 0.10, 0.0])
            eef2center[:3, :3] = np.array([[-mu, -mu, 0.0], 
                                           [0.0, 0.0, -1.0],
                                           [mu, -mu, 0.0]])
            # get transform matrix
            rot_mat = R.from_euler('z', delta).as_matrix()
            rot_trans = np.eye(4)
            rot_trans[:3, :3] = rot_mat
            # get new eef pose in world frame
            center2eef = np.linalg.inv(eef2center)
            new_eef_pose = eef_pose @ center2eef @ rot_trans @ eef2center
            print('New pose: ', new_eef_pose)
            # set target pose
            pose_msg = matrix_to_pose_msg(new_eef_pose)
            self.move_group.set_pose_target(pose_msg)
            # plan and execute
            success, plan, _, _ = self.move_group.plan()
            if not success:
                return False
            self.move_group.execute(plan, wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            return True
        else:
            raise ValueError('Invalid action: {}'.format(action))
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        if plan:
            return True
        return False


    def open_gripper(self):
        self.gripper_publisher.publish(control_msgs.msg.GripperCommandActionGoal(
            goal=control_msgs.msg.GripperCommandGoal(
                command=control_msgs.msg.GripperCommand(
                    position=0.039, max_effort=5.0))))
        rospy.sleep(1.5)
        
    def close_gripper(self):
        self.gripper_publisher.publish(control_msgs.msg.GripperCommandActionGoal(
            goal=control_msgs.msg.GripperCommandGoal(
                command=control_msgs.msg.GripperCommand(
                    position=0.0, max_effort=10.0))))
        rospy.sleep(1.5)

    def rotate(self, angle, degree=True):
        joint_state, eef_pose = self.get_robot_states()
        if degree:
            angle = angle / 180.0 * pi
        joint_state[-1] += angle
        if joint_state[-1] > 2.86 or joint_state[-1] < -2.86:
            rospy.logwarn('Rotate angle out of range!')
            return False
        self.move_group.go(joint_state, wait=True)
        self.move_group.stop()
        # check if reached target pose
        current_joints = self.move_group.get_current_joint_values()
        return all_close(joint_state, current_joints, 0.01)
        
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
    robot_behavior = BehaviorDemo()
    traj_record_pub = rospy.Publisher('/robot_traj_recorder/command', String, queue_size=1)

    while not rospy.is_shutdown():
        # wait for user input
        key = input("Press enter to act ; 'q' to quit: ")
        if key == 'q':
            break
        # load transform pose
        pose_trans = np.load('/root/data/transform.npy')
        print(pose_trans)
        traj_record_pub.publish('start')
        success = robot_behavior.act(pose_trans)
        if success:
            rospy.loginfo("Act successful!")
        else:
            rospy.loginfo("Act failed!")
        # print current state
        joint_state, eef_pose = robot_behavior.get_robot_states()
        rospy.loginfo("Joint states: {}".format(joint_state))
        rospy.loginfo("End effector pose: {}".format(eef_pose))
        # post action
        key = input("Press keys to post act, q to quit;")
        if key == 'q':
            traj_record_pub.publish('stop')
            break
        actions = pickle.load(open('/root/data/action.pkl', 'rb'))
        for action in actions:
            robot_behavior.post_act(action[0], action[1])
        traj_record_pub.publish('stop')