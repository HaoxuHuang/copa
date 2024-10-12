import sys
import os
import copy
import time
import rospy
import atexit
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import pickle
import numpy as np
import dataclasses
from scipy.spatial.transform import Rotation as R
from math import pi, tau, dist, fabs, sin, cos

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

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


@dataclasses.dataclass
class RobotStateFrame:
    timestamp: int  # in milliseconds
    joint_state: list
    eef_pose: np.ndarray

    def asdict(self):
        return dataclasses.asdict(self)


class RobotStatesRecorder(object):
    def __init__(self, frame_rate=10, max_frames=36000) -> None:
        super().__init__()
        # First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('robot_traj_recorder', anonymous=True)


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

        # record command subscriber
        rospy.Subscriber('/robot_traj_recorder/command', String, self.command_callback)
        
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
        self.eef_link = eef_link
        self.group_names = group_names
        self.recording = False
        self.recording_frames = []
        self.frame_rate = rospy.Rate(frame_rate)
        self.max_frames = max_frames

    def get_robot_states(self):
        joint_state = self.move_group.get_current_joint_values()
        eef_pose = self.move_group.get_current_pose()
        return joint_state, eef_pose
    
    def command_callback(self, data):
        command = data.data
        if command == 'start':
            self.start_recording()
        elif command == 'stop':
            self.stop_recording()
        elif command == 'clear':
            self.clear_recording()

    def start_recording(self):
        if self.recording:
            rospy.logwarn("Already recording...")
        self.recording = True
        self.recording_frames = []
        rospy.loginfo("Start recording...")

    def stop_recording(self):
        if not self.recording:
            rospy.logwarn("Not recording...")
            return
        self.recording = False
        rospy.loginfo("Stop recording...")
        # save to file
        pickle.dump(self.recording_frames, open('/root/data/trajectory_record_{}.pkl'.format(time.strftime("%H-%M-%S")), 'wb'))
        rospy.loginfo("{} frames have been saved to file...".format(len(self.recording_frames)))

    def clear_recording(self):
        self.recording_frames = []
        rospy.loginfo("Clear recording...")

    def record(self):
        if self.recording:
            if len(self.recording_frames) >= self.max_frames:
                rospy.logwarn("Recording frames exceed max frames, stop recording...")
                self.stop_recording()
                return
            joint_state, eef_pose = self.get_robot_states()
            timestamp = eef_pose.header.stamp.to_nsec() // 1000000
            eef_pose_mat = pose_msg_to_matrix(eef_pose.pose)
            robot_state_frame = RobotStateFrame(timestamp, joint_state, eef_pose_mat)
            self.recording_frames.append(robot_state_frame.asdict())
        self.frame_rate.sleep()



if __name__ == '__main__':
    recorder = RobotStatesRecorder()
    @atexit.register
    def shutdown():
        rospy.loginfo("Shutting down...")
        recorder.stop_recording()
        rospy.loginfo("Shutdown complete.")

    while not rospy.is_shutdown():
        recorder.record()