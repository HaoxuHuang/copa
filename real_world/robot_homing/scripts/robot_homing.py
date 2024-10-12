import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi, tau, dist, fabs, sin, cos

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list


def all_close(goal, actual, tolerance=1e-3):
    for i in range(len(goal)):
        if fabs(goal[i] - actual[i]) > tolerance:
            return False
    return True

class RobotHoming(object):
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

        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.eef_link = eef_link
        self.group_names = group_names
        # home joint angles
        self.home_pose = [0.000,-0.785,0.0,-1.90,0.0,1.37,0.785]

    def get_robot_states(self):
        joint_state = self.move_group.get_current_joint_values()
        eef_pose = self.move_group.get_current_pose().pose
        return joint_state, eef_pose
    
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
    robot_homing = RobotHoming()
    success = robot_homing.go_home()
    if success:
        rospy.loginfo("Homing successful!")
    else:
        rospy.loginfo("Homing failed!")
    # print current state
    joint_state, eef_pose = robot_homing.get_robot_states()
    rospy.loginfo("Joint states: {}".format(joint_state))
    rospy.loginfo("End effector pose: {}".format(eef_pose))
    # wait for user input
    input("Press enter to exit ;")