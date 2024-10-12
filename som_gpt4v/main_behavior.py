import io
import gradio as gr
import torch
import argparse
from PIL import Image, ImageDraw
import traceback
import cv2
from typing import List
# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import interactive_seem_m2m_auto, inference_seem_pano_remove_robot_arm, inference_seem_interactive, inference_seem_pano

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto_remove_robot_arm, prompt_switch

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto_remove_robot_arm, inference_sam_m2m_auto


from task_adapter.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

from scipy.ndimage import label
import numpy as np
import pickle
import os
import open3d as o3d

from gpt4v import request_gpt4v, request_gpt4v_multi_image


'''
build args
'''
semsam_cfg = "configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg = "configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = "./swinl_only_sam_many2many.pth"
sam_ckpt = "./sam_vit_h_4b8939.pth"
seem_ckpt = "./seem_focall_v1.pt"

opt_semsam = load_opt_from_config_file(semsam_cfg)
opt_seem = load_opt_from_config_file(seem_cfg)
opt_seem = init_distributed_seem(opt_seem)

'''
build model
'''
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()
model_sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).eval().cuda()
# model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()

# with torch.no_grad():
#     with torch.autocast(device_type='cuda', dtype=torch.float16):
#         model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

def inference(image, granularity, robot_arm_mask, interleave_ratio, if_remove_robot_arm):
    # choose model according to segmentation granularity
    if granularity < 1.5:
        model_name = 'seem'
    elif granularity > 2.5:
        model_name = 'sam'
    else:
        model_name = 'semantic-sam'
        if granularity < 1.5 + 0.14:                
            level = [1]
        elif granularity < 1.5 + 0.28:
            level = [2]
        elif granularity < 1.5 + 0.42:
            level = [3]
        elif granularity < 1.5 + 0.56:
            level = [4]
        elif granularity < 1.5 + 0.70:
            level = [5]
        elif granularity < 1.5 + 0.84:
            level = [6]
        else:
            level = [6, 1, 2, 3, 4, 5]

    # if if_remove_robot_arm:
    #     text_size, hole_scale, island_scale=robot_arm_mask.shape[0],100,100
    # else:
    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh = '','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False

        if model_name == 'semantic-sam':
            model = model_semsam
            output, mask = inference_semsam_m2m_auto_remove_robot_arm(model, image, robot_arm_mask, interleave_ratio, if_remove_robot_arm, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark'])
            output_with_robot, mask_with_robot = inference_semsam_m2m_auto_remove_robot_arm(model, image, robot_arm_mask, interleave_ratio, False, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark'])
            Image.fromarray(output_with_robot).save('output_with_robot.png')

        elif model_name == 'sam':
            model = model_sam
            output, mask = inference_sam_m2m_auto_remove_robot_arm(model, image, robot_arm_mask, interleave_ratio, if_remove_robot_arm, text_size, label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark'])
            output_with_robot, mask_with_robot = inference_sam_m2m_auto_remove_robot_arm(model, image, robot_arm_mask, interleave_ratio, if_remove_robot_arm=False, text_size=text_size, label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark'])
            Image.fromarray(output_with_robot).save('output_with_robot.png')

        elif model_name == 'seem':
            model = model_seem
            output, mask = inference_seem_pano_remove_robot_arm(model, image, robot_arm_mask, interleave_ratio, if_remove_robot_arm, text_size, label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark'])

    return output, mask

def gpt4v_response(instruction, image, prompt_path):
    for i in range(3):
    # while True:
        try:
            prompts, visions = [], []
            for j in range(4):
                with open(f'{prompt_path}/prompt{j + 1}.txt', 'r') as f:
                    prompts.append(f.read())
                if j < 3:
                    visions.append(f'{prompt_path}/prompt{j + 1}.png')
            prompts[-1] += instruction
            visions.append(image)
            res = request_gpt4v_multi_image(prompts, visions)
            return res
        except Exception as e:
            traceback.print_exc()
            continue

def extract_label(res) -> List[str]:
    '''Extract the label in the respond of GPT-4V'''
    import re
    res = re.findall(r'\[(.*?)\]', res)
    res = res[-1].split(',')
    res = [int(i.strip()) for i in res]
    return res

def crop_mask(image, output, mask):
    output_img = Image.fromarray(output)
    mask_bbox1 = (np.array(mask['bbox']) * (image.width / output_img.width)).astype(np.int32)
    # print(mask_bbox1)
    margin = 0.2
    wh = (image.height, image.width)
    bbox = (int(mask_bbox1[0]-margin*mask_bbox1[2]), int(mask_bbox1[1]-margin*mask_bbox1[3]), 
            int(mask_bbox1[0]+(1+margin)*mask_bbox1[2]), int(mask_bbox1[1]+(1+margin)*mask_bbox1[3]))
    bbox = (max(0, bbox[0]), max(0, bbox[1]), min(wh[1], bbox[2]), min(wh[0], bbox[3]))
    image1 = image.crop(bbox)
    return image1, bbox

def combine_mask(image, mask, bbox):
    mask_to_save = np.zeros((image.height, image.width), dtype=np.uint8)
    img_mask = Image.fromarray(mask['segmentation'])
    resized_mask = img_mask.resize((bbox[2]-bbox[0], bbox[3]-bbox[1]))
    mask_to_save[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np.array(resized_mask)
    return mask_to_save

def draw_pic(image, segmentation_list, label_mode='1', alpha=0.1, anno_mode=['Mask', 'Mark']):
    image_ori = np.asarray(image)
    visual = Visualizer(image_ori, metadata=metadata)
    for i, ann in enumerate(segmentation_list):
        demo = visual.draw_binary_mask_with_number(ann, text=str(i + 1), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode)
    im = demo.get_image()
    return im

def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def fit_line_to_points(points):
    from scipy.linalg import svd
    mean_point = np.mean(points, axis=0)
    position_vectors = points - mean_point
    covariance_matrix = np.cov(position_vectors.T)
    U, S, Vt = svd(covariance_matrix)
    direction_vector = Vt[0]
    return mean_point, direction_vector

def get_box(segments):
    image = np.uint8(segments) * 255
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def fit_plane_normal(points):
    from scipy.linalg import eigh
    mean_point = np.mean(points, axis=0)
    position_vectors = points - mean_point
    covariance_matrix = np.cov(position_vectors.T)
    eigenvalues, eigenvectors = eigh(covariance_matrix)
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
    return mean_point, normal_vector

def fit_plane_normal_ransac(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                            ransac_n=3,
                                            num_iterations=1000)
    # find the plane normal
    [a, b, c, d] = plane_model
    normal_vector = np.array([a, b, c])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    # find plane center
    plane_center = np.mean(points[inliers], axis=0)
    return plane_center, normal_vector

def convert_point_cloud(point_cloud, extrinsic):
    height, width = point_cloud.shape[:2]
    color_cloud = point_cloud.reshape(-1, 6)[:, 3:]
    point_cloud = point_cloud.reshape(-1, 6)[:, :3]
    point_cloud = np.concatenate([point_cloud, np.ones((point_cloud.shape[0], 1))], axis=1)
    point_cloud = np.matmul(point_cloud, extrinsic.T)
    point_cloud = point_cloud[:, :3]
    point_cloud = np.concatenate([point_cloud, color_cloud], axis=1)
    point_cloud = point_cloud.reshape(height, width, 6)
    return point_cloud

def get_aspect_ratio(segmentation):
    # import pdb; pdb.set_trace()
    img_uint8 = np.uint8(segmentation * 255)
    # close operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_uint8 = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)
    # find contours
    contours, _ = cv2.findContours(img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img_uint8 = np.ascontiguousarray(img_uint8).copy()
    img_uint8 = cv2.drawContours(img_uint8, [box], 0, (255, 0, 0), 2)
    cv2.imwrite('output_draw_box.png', img_uint8)
    aspect_ratio = max(distance(box[0], box[1]), distance(box[1], box[2])) / min(distance(box[0], box[1]), distance(box[1], box[2]))
    return aspect_ratio

def draw_surface(center_point, norm_vec, intrinsics, extrinsics, length=0.1):
    """
    Draw a surface on the image
    :param center_point: 3D center point of the surface
    :param norm_vec: 3D normal vector of the surface
    :param length: length of the surface
    """
    end_point = center_point + length * norm_vec
    center_point = project_point(center_point, intrinsics, extrinsics)
    end_point = project_point(end_point, intrinsics, extrinsics)

    return center_point, end_point

def project_point(point, intrinsics, extrinsics):
    """
    Project a point to the image plane
    :param point: point to project
    :param intrinsics: intrinsic matrix
    :param extrinsics: extrinsic matrix
    :return: projected point
    """
    # Convert the point to homogeneous coordinates
    point = np.array([point[0], point[1], point[2], 1])
    uv = np.dot(intrinsics, np.dot(extrinsics[:3,:], point))
    uv = uv / uv[2]
    # uv = uv[1::-1]
    return uv[:2].astype(int)

def find_true_point_on_line(binary_matrix, x, y, slope, intercept, direction):
    height, width = binary_matrix.shape
    while 0 <= x < width and 0 <= y < height:
        if binary_matrix[int(round(y)), int(round(x))]:
            return int(round(x)), int(round(y))
        x += direction
        y = slope * x + intercept
    return None

def fit_line_to_region(binary_matrix):
    # get coordinates of non-zero pixels
    from sklearn.linear_model import LinearRegression
    y_coords, x_coords = np.where(binary_matrix)

    # Linear regression
    model = LinearRegression().fit(x_coords.reshape(-1, 1), y_coords)
    slope = model.coef_[0]
    intercept = model.intercept_

    # leftmost and rightmost x coordinates
    x_left, x_right = x_coords.min(), x_coords.max()

    # search for intersection points
    left_point = find_true_point_on_line(binary_matrix, x_left, slope * x_left + intercept, slope, intercept, 1)
    right_point = find_true_point_on_line(binary_matrix, x_right, slope * x_right + intercept, slope, intercept, -1)

    return left_point, right_point

def get_robot_center_point(robot_arm_mask, image):
    resized_robot_arm_mask = cv2.resize(robot_arm_mask.astype(np.uint8), (image.width, image.height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
    y_coords, x_coords = np.where(resized_robot_arm_mask)
    robot_center_point = (int(x_coords.mean()), int(y_coords.mean()))
    return robot_center_point


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_base_dir', type=str, default='/home/pjlab/moveit_ws/env_pics/', help='path to image')
    parser.add_argument('--task', type=str, default='test', help='task to perform')
    parser.add_argument('--if_remove_robot_arm', action='store_false', default=True, help='if remove robot arm')
    parser.add_argument('--robot_arm_mask_name', type=str, default='robot_arm_mask.npy', help='path to robot arm mask')
    parser.add_argument('--extrinsic_path', type=str, default='data/camera_extrinsic2.npy', help='path to camera extrinsic')
    parser.add_argument('--interleave_ratio', type=float, default=0.2, help='remove the mask with interleave ratio larger than this value')
    parser.add_argument('--granularity_1', type=float, default=1.9, help='segmentation granularity for 1st stage')
    parser.add_argument('--granularity_2', type=float, default=2.0, help='segmentation granularity for 2nd stage')
    parser.add_argument('--line_aspect_ratio', type=float, default=3, help='segmentation granularity for 2nd stage')
    parser.add_argument('--base_output_dir', type=str, default='results', help='output dir')
    args = parser.parse_args()

    output_dir = os.path.join(args.base_output_dir, args.task)
    os.makedirs(output_dir, exist_ok=True)
    robot_arm_mask = np.load(os.path.join(args.image_base_dir, args.task, args.robot_arm_mask_name))
    image = Image.open(os.path.join(args.image_base_dir, args.task, args.task + '.png'))

    with open(os.path.join(args.image_base_dir, args.task, 'instruction.txt'), 'r') as f:
        instruction = f.read()
        print(instruction)
    output, masks = inference(image, args.granularity_1, robot_arm_mask, args.interleave_ratio, args.if_remove_robot_arm)
    output_img = Image.fromarray(output)
    output_img.save(os.path.join(output_dir, 'output.png'))

    respond = gpt4v_response(instruction, output_img, 'first_prompt')
    # # respond = gpt4v_response(instruction, '/home/pjlab/grasp-pipeline/som_gpt4v/outputs/som.png', 'first_prompt')

    print()
    print(respond)
    labels = extract_label(respond)
    
    object_masks = [masks[label - 1] for label in labels]
    first_aspect_ratio = [get_aspect_ratio(mask['segmentation']) for mask in object_masks]
    # print('First ratio:', first_aspect_ratio)
    
    segmentation_list, aspect_ratio_list = [], []
    for i, object_mask in enumerate(object_masks):
        object_segment = cv2.resize(object_mask['segmentation'].astype(np.uint8), (image.width, image.height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
        if first_aspect_ratio[i] > args.line_aspect_ratio:
            segmentation_list.append(object_segment)
            aspect_ratio_list.append(first_aspect_ratio[i])
            continue
        crop_image, bbox = crop_mask(image, output, object_mask)
        output_bbox = (np.array(bbox) * (output_img.width / image.width)).astype(np.int32)
        object_robot_arm_mask = robot_arm_mask[output_bbox[1]:output_bbox[3], output_bbox[0]:output_bbox[2]]
        height, width = 640, 640 * crop_image.width // crop_image.height
        resized_object_robot_arm_mask = cv2.resize(object_robot_arm_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        output2, masks2 = inference(crop_image, args.granularity_2, resized_object_robot_arm_mask, args.interleave_ratio, args.if_remove_robot_arm)
        # paint object_robot_arm_mask blask on output2
        aa = np.zeros((resized_object_robot_arm_mask.shape[0], resized_object_robot_arm_mask.shape[1], 3), dtype=np.uint8)
        aa[:, :, 0] = resized_object_robot_arm_mask
        aa[:, :, 1] = resized_object_robot_arm_mask
        aa[:, :, 2] = resized_object_robot_arm_mask
        output2 = output2 * (1 - aa)
        Image.fromarray(output2).save(os.path.join(output_dir, f'seg_output_{i}.png'))

        for j, single_mask2 in enumerate(masks2):
            if single_mask2['segmentation'][:5, :5].sum() > 0 or single_mask2['segmentation'][:5, -5:].sum() > 0 or single_mask2['segmentation'][-5:, :5].sum() > 0 or single_mask2['segmentation'][-5:, -5:].sum() > 0:
                continue
            single_mask2['segmentation'] = single_mask2['segmentation'].astype(np.uint8)
            resized_single_mask = cv2.resize(single_mask2['segmentation'], (crop_image.width, crop_image.height), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
            single_segmentation = np.zeros((image.height, image.width), dtype=np.bool_)
            single_segmentation[bbox[1]:bbox[3], bbox[0]:bbox[2]] = resized_single_mask
            
            # filter out the mask with too little area
            if single_segmentation.sum() < 200:
                continue
            # get intersection of single_segmentation and object_segment
            intersection = single_segmentation & object_segment
            if intersection.sum() / single_segmentation.sum() < 0.5:
                continue

            segmentation_list.append(single_segmentation)

            aspect_ratio = get_aspect_ratio(single_segmentation)
            aspect_ratio_list.append(aspect_ratio)
    output_mix_image = Image.fromarray(draw_pic(image, segmentation_list))
    output_mix_image.save(os.path.join(output_dir, 'output_mix.png'))
    [Image.fromarray(draw_pic(image, [segmentation_list[i]])).save(os.path.join(output_dir, f'output_draw_{i}.png')) for i in range(len(segmentation_list))]

    respond = gpt4v_response(instruction, output_mix_image, 'second_prompt')
    print()
    print(respond)
    labels = extract_label(respond)
    # labels = [4, 5, 6]

    final_masks = [segmentation_list[label - 1] for label in labels]
    final_aspect_ratio = [aspect_ratio_list[label - 1] for label in labels]
    
    Image.fromarray(draw_pic(image, final_masks)).save(os.path.join(output_dir, 'final_output.png'))

    point_cloud = np.load(os.path.join(args.image_base_dir, args.task, args.task + '.npy'))
    extrinsic = np.load(args.extrinsic_path)
    intrinsic = np.array([[607.3267822265625,  0, 332.8100891113281], [ 0, 606.141845703125, 245.5809326171875], [ 0,  0,  1]])
    point_cloud = convert_point_cloud(point_cloud, extrinsic)
    robot_center_point = get_robot_center_point(robot_arm_mask, image)

    target_point_pos, mask_only_surface, spatial_data = [], [], []
    for single_seg, aspect_ratio in zip(final_masks, final_aspect_ratio):
        if aspect_ratio > args.line_aspect_ratio:
            # line_point_cloud = point_cloud[single_seg][:, :3]
            # line_point_cloud = line_point_cloud[line_point_cloud[:, 2] < 0.45]
            # mean_point, direction_vector = fit_line_to_points(line_point_cloud)
            left_point, right_point = fit_line_to_region(single_seg)
            left_point_pos = (left_point[0] * 0.95 + right_point[0] * 0.05, left_point[1] * 0.95 + right_point[1] * 0.05)
            right_point_pos = (left_point[0] * 0.05 + right_point[0] * 0.95, left_point[1] * 0.05 + right_point[1] * 0.95)
            left_point_pos, right_point_pos = np.array(left_point_pos).astype(int), np.array(right_point_pos).astype(int)
            if distance(left_point_pos, robot_center_point) < distance(right_point_pos, robot_center_point):
                target_point_pos.append((aspect_ratio, left_point_pos, right_point_pos))
                spatial_data.append((aspect_ratio, point_cloud[left_point_pos[1], left_point_pos[0], :3], point_cloud[right_point_pos[1], right_point_pos[0], :3]))
            else:
                target_point_pos.append((aspect_ratio, right_point_pos, left_point_pos))
                spatial_data.append((aspect_ratio, point_cloud[right_point_pos[1], right_point_pos[0], :3], point_cloud[left_point_pos[1], left_point_pos[0], :3]))
        else:
            mask_only_surface.append(single_seg)
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(single_seg)
            boundaries = single_seg ^ eroded
            surface_point_cloud = point_cloud[boundaries][:, :3]
            surface_point_cloud = surface_point_cloud[surface_point_cloud[:, 2] < 0.45]
            # mean_point, normal_vector = fit_plane_normal(surface_point_cloud)
            mean_point, normal_vector = fit_plane_normal_ransac(surface_point_cloud)
            normal_vector = np.sign(np.dot(extrinsic[:3, 3] - mean_point, normal_vector)) * normal_vector

            center_point, end_point = draw_surface(mean_point, normal_vector, intrinsic, np.linalg.inv(extrinsic))
            target_point_pos.append((aspect_ratio, center_point, end_point))
            spatial_data.append((aspect_ratio, mean_point, normal_vector))

    image_add_mask = draw_pic(image, mask_only_surface, alpha=0.2, anno_mode=['Mask'])

    for i, (aspect_ratio, start_point, end_point) in enumerate(target_point_pos):
        image_add_mask = cv2.circle(image_add_mask, start_point, 5, (0, 0, 255), -1)
        image_add_mask = cv2.line(image_add_mask, start_point, end_point, (0, 0, 255), 2)
        mark_point = start_point if aspect_ratio < args.line_aspect_ratio else end_point
        another_point = start_point if aspect_ratio >= args.line_aspect_ratio else end_point

        ratio = 0.17
        # if i == 0:
        #     marker_pos = (mark_point[0] * (1 + ratio) + another_point[0] * -ratio + 30, mark_point[1] * (1 + ratio) + another_point[1] * -ratio - 25)
        # else:
        marker_pos = (mark_point[0] * (1 + ratio) + another_point[0] * -ratio, mark_point[1] * (1 + ratio) + another_point[1] * -ratio - 5)
        marker_pos = np.array(marker_pos).astype(int)

        visual = Visualizer(Image.fromarray(image_add_mask), metadata=metadata)
        visual.draw_text(str(i + 1), (marker_pos[0], marker_pos[1]), color=[1,1,1])
        image_add_mask = visual.output.get_image()

    Image.fromarray(image_add_mask).save(os.path.join(output_dir, 'final_output_add_mask.png'))
    pickle.dump(spatial_data, open(os.path.join(output_dir, 'spatial_data.pkl'), 'wb'))
    pickle.dump(final_masks, open(os.path.join(output_dir, 'segmentation_list.pkl'), 'wb'))
    # print(spatial_data)