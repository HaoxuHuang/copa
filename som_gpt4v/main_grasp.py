import io
import pdb
import torch
import argparse
from PIL import Image
import cv2
from typing import List
from pathlib import Path
import traceback
import shutil
# seem
from seem.modeling.BaseModel import BaseModel as BaseModel_Seem
from seem.utils.distributed import init_distributed as init_distributed_seem
from seem.modeling import build_model as build_model_seem
from task_adapter.seem.tasks import interactive_seem_m2m_auto, inference_seem_pano, inference_seem_interactive

# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch, inference_semsam_m2m_auto_filter

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto, inference_sam_m2m_auto_filter
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive


from detectron2.data import MetadataCatalog
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

from scipy.ndimage import label
import numpy as np
from omegaconf import OmegaConf

from gpt4v_azure import request_gpt4v, clear_history
from mask_filters import get_mask_filter


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
model_seem = BaseModel_Seem(opt_seem, build_model_seem(opt_seem)).from_pretrained(seem_ckpt).eval().cuda()


def inference(image, granularity, *args, **kwargs):
    # choose model according to segmentation granularity
    if granularity < 1.5:
        model_name = 'seem'
    elif granularity > 2.5:
        model_name = 'sam'
    else:
        model_name = 'semantic-sam'
        if granularity < 1.5 + 0.14:                # 1.64
            level = [1]
        elif granularity < 1.5 + 0.28:            # 1.78
            level = [2]
        elif granularity < 1.5 + 0.42:          # 1.92
            level = [3]
        elif granularity < 1.5 + 0.56:          # 2.06
            level = [4]
        elif granularity < 1.5 + 0.70:          # 2.20
            level = [5]
        elif granularity < 1.5 + 0.84:          # 2.34
            level = [6]
        else:
            level = [6, 1, 2, 3, 4, 5]


    text_size, hole_scale, island_scale=500,100,100
    text, text_part, text_thresh = '','','0.0'
    anno_mode = ['Mask', 'Mark']
    # anno_mode = ['Mark']
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False

        if model_name == 'semantic-sam':
            model = model_semsam
            # output, mask = inference_semsam_m2m_auto(model, image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, alpha=0.1, anno_mode=anno_mode, *args, **kwargs)
            output, mask = inference_semsam_m2m_auto_filter(model, image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, alpha=0.1, anno_mode=anno_mode, *args, **kwargs)

        elif model_name == 'sam':
            model = model_sam
            # output, mask = inference_sam_m2m_auto(model, image, text_size, anno_mode=anno_mode, alpha=0.1)
            output, mask = inference_sam_m2m_auto_filter(model, image, text_size, anno_mode=anno_mode, alpha=0.1, *args, **kwargs)

        elif model_name == 'seem':
            model = model_seem
            output, mask = inference_seem_pano(model, image, text_size, anno_mode=anno_mode, alpha=0.1)

    return output, mask


def gpt4v_response(message, vision, stage):
    for i in range(3):
        try:
            if stage == 1:
                # locate the object in the image
                prompt1 = f"""
Answer the question as if you are a robot with a parallel jaw gripper having access to a segmented photo. Your task is to find the suitable grasp pose for the given task. Follow the exact format.
The first line should describe the basic action needed to de the task.
The second line should be the object needed to do the task, which is the target object.
The third line should be adjectives describing the properties of the target object, such as hot, full, smooth.
The following lines are the steps to accomplish the task.
The last line is the label of the target object in the photo.

Example:
Instruction: grasp a mug with hot water.
Task: pour
Object: mug
Object Property: hot
step1: find the position of mug with hot water in the provided figure
step2: find suitable areas to grasp in the sub-figure of mug with hot water
step3: grasp the suitable area to pick up the mug
Object Label: [2]

Instruction: {message}
"""
                res = request_gpt4v(prompt1, vision)
                return res
            elif stage == 2:
                # locate the object in the image
                prompt2 = f"""
Now look at the center cropped image of the target object. Which labeled component should you grasp? 
First describe the object part each label corresponds to. Then output the label of the target object part.
ATTENTION: you act as the robot, not the person in the scene. You should avoid fragile parts, and hold on protective covers if there exists. The output label must be part of the target object.
EXAMPLE:
Label 1: the handle of the mug
Label 2: the body of the mug
Label 3: the water in the mug
Target Object Part: [1]

Instruction: {message}
"""
                res = request_gpt4v(prompt2, vision)
                return res
        except Exception as e:
            traceback.print_exc()
            continue    # try again

    
def extract_label(respond) -> List[str]:
    '''Extract the label in the respond of GPT-4V'''
    # find the seperate numbers in sentence res
    respond = respond.split(' ')
    respond = [r.replace('.','').replace(',','').replace(')','').replace('"','') for r in respond]
    # find all numbers in '[]'
    respond = [r for r in respond if '[' in r]
    respond = [r.split('[')[1] for r in respond]
    respond = [r.split(']')[0] for r in respond]
    respond = [r for r in respond if r.isdigit()]
    respond = list(set(respond))
    return respond


def get_mask(masks, labels):
    # get the mask index
    mask_index = int(labels[0])
    # get the mask
    mask = masks[mask_index-1]
    return mask


def crop_mask(image, output, mask):
    output_img = Image.fromarray(output)
     # center crop the mask
    mask_bbox1 = (np.array(mask['bbox']) * (image.width / output_img.width)).astype(int)
    print(mask_bbox1)
    margin = 0.1
    wh = (image.height, image.width)
    bbox = (int(np.clip(mask_bbox1[0]-margin*mask_bbox1[2], 0, wh[1])), 
            int(np.clip(mask_bbox1[1]-margin*mask_bbox1[3], 0, wh[0])), 
            int(np.clip(mask_bbox1[0]+(1+margin)*mask_bbox1[2], 0, wh[1])), 
            int(np.clip(mask_bbox1[1]+(1+margin)*mask_bbox1[3], 0, wh[0])))
    image1 = image.crop(bbox)
    mask_in_bbox = Image.fromarray(mask['segmentation']).resize(image.size).crop(bbox)
    mask_in_bbox.save('outputs/mask_in_bbox.png')
    return image1, bbox, np.array(mask_in_bbox)

def combine_mask(image, mask, bbox):
    mask_to_save = np.zeros((image.height, image.width), dtype=np.uint8)
    img_mask = Image.fromarray(mask['segmentation'])
    resized_mask = img_mask.resize((bbox[2]-bbox[0], bbox[3]-bbox[1]))
    mask_to_save[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np.array(resized_mask)
    return mask_to_save


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file name')
    cli_args = parser.parse_args()

    while True:
        input("Press enter to inference ;")
        args = OmegaConf.load(f'data/grasp_cfg_{cli_args.config}.yaml')
        image = Image.open(args.image)
        shutil.copyfile(args.image, f'data/{cli_args.config}/image.png')
        output, masks = inference(image, args.granularity1)
        Image.fromarray(output).save(f'data/{cli_args.config}/som.png')
        respond = gpt4v_response(args.task, Image.fromarray(output), 1)
        print(respond)
        if respond is None:
            clear_history()
            continue
        labels = extract_label(respond)
        mask = get_mask(masks, labels)
        image1, bbox, mask_in_bbox = crop_mask(image, output, mask)

        # second stage
        mask_filter = get_mask_filter(args.filter.corner, args.filter.area, args.filter.intersection, 
                                      args.filter.area_thresh, mask2=mask_in_bbox, intersection_thresh=args.filter.intersection_thresh)
        output2, masks2 = inference(image1, args.granularity2, mask_filter=mask_filter)
        Image.fromarray(output2).save(f'data/{cli_args.config}/som2.png')
        respond2 = gpt4v_response(args.task, Image.fromarray(output2), 2)
        if respond2 is None:
            clear_history()
            continue
        print(respond2)
        labels2 = extract_label(respond2)
        mask2 = get_mask(masks2, labels2)
        clear_history()

        # combine the mask
        mask_to_save = combine_mask(image, mask2, bbox)
        np.save(args.output, mask_to_save)

        # save response
        with open(args.response, 'w') as f:
            f.write(respond)
            f.write('\n\n\n')
            f.write(respond2)

        # visualize
        masked_image = np.array(image)
        masked_image[mask_to_save==0] = 0
        # pdb.set_trace()
        Image.fromarray(masked_image).show()
