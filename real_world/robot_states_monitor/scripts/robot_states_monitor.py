import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi, tau, dist, fabs, sin, cos

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list



class RobotStatesMonitor(object):
    def __init__(self) -> None:
        super().__init__()
        # First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('robot_states_monitor', anonymous=True)


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

    def get_robot_states(self):
        joint_state = self.move_group.get_current_joint_values()
        eef_pose = self.move_group.get_current_pose().pose
        return joint_state, eef_pose
    
    def monitor(self, duration=0.1):
        while not rospy.is_shutdown():
            joint_state, eef_pose = self.get_robot_states()
            print("Joint states: ", joint_state)
            print("End effector pose: ", eef_pose)
            rospy.sleep(duration)


if __name__ == '__main__':
    try:
        robot_states_monitor = RobotStatesMonitor()
        robot_states_monitor.monitor()
    except rospy.ROSInterruptException:
        pass