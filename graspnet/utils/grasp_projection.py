from graspnetAPI.grasp import GraspGroup, Grasp
import numpy as np
import copy
import cv2

import pdb


def filter_grasp_in_mask(grasp_group, mask, camera, extrinsics=None, camera_s=1000.0, expansion=5):
    '''Project the grasp pose group to the image plane and filter the grasps in the mask.
    Args:
        grasp_group (GraspGroup): the grasp pose group.
        mask (np.ndarray): the mask of the image.
        camera (np.ndarray or str): the intrinsic matrix or 'kinect' or 'realsense'.
        camera_s (float): the camera scale.
    '''
    # Project the grasp pose group to the image plane
    rect_grasp_group = project_to_image_plane(grasp_group, camera, extrinsics, camera_s)
    # Filter the grasps in the mask
    mask = mask.astype(np.uint8)
    filtered_grasp_group = GraspGroup()
    # pdb.set_trace()
    for i in range(rect_grasp_group.shape[0]):
        rect_grasp_pt = rect_grasp_group[i,:]  # (x, y)
        if (0 < rect_grasp_pt[1] < mask.shape[0] and 
            0 < rect_grasp_pt[0] < mask.shape[1] and 
            mask[int(rect_grasp_pt[1]), int(rect_grasp_pt[0])] > 0):
            filtered_grasp_group.add(grasp_group[i])
    if len(filtered_grasp_group) < 5:
        # binary expansion
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        # mask = cv2.dilate(mask, np.ones((50, 50), np.uint8), iterations=1)
        # pdb.set_trace()
        for i in range(rect_grasp_group.shape[0]):
            rect_grasp_pt = rect_grasp_group[i,:]  # (x, y)
            if (0 < rect_grasp_pt[1] < mask.shape[0] and 
                0 < rect_grasp_pt[0] < mask.shape[1] and 
                mask[int(rect_grasp_pt[1]), int(rect_grasp_pt[0])] > 0):
                filtered_grasp_group.add(grasp_group[i])
    return filtered_grasp_group

def project_to_image_plane(grasp_group, camera, extrinsics=None, camera_s=1000.0):
    """Project the 6DoF grasp pose group to the image plane."""
    if extrinsics is not None:
        grasp_group = copy.deepcopy(grasp_group)
        grasp_group = grasp_group.transform(extrinsics)
    if isinstance(camera, str):
        return grasp_group.to_rect_grasp_group(camera).center_points
    elif isinstance(camera, np.ndarray):
        translations = grasp_group.translations
        if len(camera.shape) == 1:
            fx = camera[0]
            fy = camera[4]
            cx = camera[2]
            cy = camera[5]
        else:
            fx, fy = camera[0, 0], camera[1, 1]
            cx, cy = camera[0, 2], camera[1, 2]
        # z = translations[:, 2] * camera_s
        coords_x = translations[:, 0] / translations[:, 2] * fx + cx
        coords_y = translations[:, 1] / translations[:, 2] * fy + cy
        return np.stack([coords_x, coords_y], axis=-1)
    
def look_at(camera_center, point, upward):
    def normalize(v):
        return v / np.linalg.norm(v)
    forward = normalize(point - camera_center)
    right = normalize(np.cross(upward, forward))
    up = np.cross(forward, right)
    rotation = np.array([right, up, forward]).T
    translation = -rotation @ camera_center
    extrinsics = np.zeros((4, 4))
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation
    extrinsics[3, 3] = 1
    return extrinsics
    

if __name__ == '__main__':
    # Create a dummy grasp group
    grasp_group = GraspGroup()
    grasp_group.add(Grasp())
    grasp_group.add(Grasp())
    grasp_group.add(Grasp())

    # Create a dummy mask
    mask = np.zeros((10, 10))
    mask[0:5, 0:5] = 1

    intrinsics = np.array([[1000, 0, 5], [0, 1000, 5], [0, 0, 1]])
    extrinsics = np.eye(4)

    # Call the function under test
    filtered_grasp_group = filter_grasp_in_mask(grasp_group, mask, intrinsics, extrinsics, camera_s=1000.0)

    # Check the filtered grasp group
    print(len(filtered_grasp_group))