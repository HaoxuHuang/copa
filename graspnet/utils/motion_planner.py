# Collistion free motion planning for the robot
# powered by pybullet_planning
#
from pybullet_planning.pybullet_tools.utils import get_movable_joints, get_joint_positions, set_joint_positions, \
    plan_joint_motion, link_from_name, INF
from pybullet_planning.pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver
from pybullet_planning.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF

def collistion_free_motion_planning(robot, joints, end_pos, obstacles, self_collisions=False, **kwargs):
    """
    Using pybullet_planning to plan a collision free motion for the robot
    Alogorithm is birrt: Bi-directional Rapidly-exploring Random Tree"""
    tool_link = link_from_name(robot, 'panda_hand')
    # inverse kinematics
    final_conf = next(either_inverse_kinematics(robot, PANDA_INFO, tool_link, end_pos, use_pybullet=False,
                                          max_distance=INF, max_time=10, max_candidates=INF, **kwargs), None)
    print("final_conf: ", final_conf, flush=True)
    if final_conf is None:
        print("No IK solution found!", flush=True)
        return None
    rtn = plan_joint_motion(robot, joints, final_conf, obstacles=obstacles,
                             self_collisions=self_collisions)
    if rtn is None:
        print("No path found!", flush=True)
        return None
    return rtn


# test
if __name__ == '__main__':
    from pybullet_planning.pybullet_tools.utils import connect, load_model, disconnect, wait_if_gui, create_box, set_point, dump_body, \
        HideOutput, LockRenderer, joint_from_name, set_euler, get_euler, get_point, \
        set_joint_position, get_joint_positions, pairwise_collision, stable_z, wait_for_duration, get_link_pose, \
        link_from_name, get_pose, euler_from_quat, multiply, invert, draw_pose, unit_point, unit_quat, \
        remove_debug, get_aabb, draw_aabb, get_subtree_aabb, ROOMBA_URDF, set_all_static, assign_link_colors, \
        set_camera_pose, RGBA, draw_point, Pose, Point, Euler
    from pybullet_planning.pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver
    from pybullet_planning.pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
    from pybullet_planning.pybullet_tools.utils import get_movable_joints, get_joint_positions, set_joint_positions, \
        plan_joint_motion, link_from_name
    import numpy as np

    connect(use_gui=True)
    robot = load_model(FRANKA_URDF)
    joints = get_movable_joints(robot)
    tool_link = link_from_name(robot, 'panda_hand')
    print("joints: ", joints, flush=True)
    # end_pos = get_link_pose(robot, tool_link)
    sample_joint = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]
    # set initial configuration
    set_joint_positions(robot, joints, sample_joint)
    end_pos = Pose(Point(0.5, 0.5, 0.5), Euler(0, 0, 0))
    obstacles = []
    # draw boxes as obstacles
    for i in range(5):
        box = create_box(0.1, 0.1, 0.1, color=(1, 0, 0, 0.5))
        set_point(box, [0.5, 0.2, 0.3 * i])
        set_euler(box, [0, 0, np.pi / 4 * i])
        obstacles.append(box)
    # draw aabb
    for obstacle in obstacles:
        obstacle_aabb = get_subtree_aabb(obstacle)
        draw_aabb(obstacle_aabb)
    # planning
    ik_joints = get_ik_joints(robot, PANDA_INFO, tool_link)
    path = collistion_free_motion_planning(robot, ik_joints, end_pos, obstacles)
    print(path)
    # draw path
    for i in range(len(path) - 1):
        set_joint_positions(robot, ik_joints, path[i])
        wait_for_duration(0.1)
    wait_if_gui()
    disconnect()