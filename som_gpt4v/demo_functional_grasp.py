# --------------------------------------------------------
# Set-of-Mark (SoM) Prompting for Visual Grounding in GPT-4V
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by:
#   Jianwei Yang (jianwyan@microsoft.com)
#   Xueyan Zou (xueyan@cs.wisc.edu)
#   Hao Zhang (hzhangcx@connect.ust.hk)
# --------------------------------------------------------
import io
import gradio as gr
import torch
import argparse
from PIL import Image
import traceback
import cv2
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
from task_adapter.semantic_sam.tasks import inference_semsam_m2m_auto, prompt_switch

# sam
from segment_anything import sam_model_registry
from task_adapter.sam.tasks.inference_sam_m2m_auto import inference_sam_m2m_auto
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive


from task_adapter.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

from scipy.ndimage import label
import numpy as np

from gpt4v import request_gpt4v
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

import matplotlib.colors as mcolors
css4_colors = mcolors.CSS4_COLORS
color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]

bbox = None
client = OpenAI()

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

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        model_seem.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

# history variables
history_images = []
history_masks = []
history_texts = []
stage = 1   # 1: coarse, 2: fine
wh = None	# width and height of input image
@torch.no_grad()
def inference(image, slider, mode, alpha, label_mode, anno_mode, *args, **kwargs):
    global history_images; history_images = []
    global history_masks; history_masks = []    
    if slider < 1.5:
        model_name = 'seem'
    elif slider > 2.5:
        model_name = 'sam'
    else:
        if mode == 'Automatic':
            model_name = 'semantic-sam'
            if slider < 1.5 + 0.14:                
                level = [1]
            elif slider < 1.5 + 0.28:
                level = [2]
            elif slider < 1.5 + 0.42:
                level = [3]
            elif slider < 1.5 + 0.56:
                level = [4]
            elif slider < 1.5 + 0.70:
                level = [5]
            elif slider < 1.5 + 0.84:
                level = [6]
            else:
                level = [6, 1, 2, 3, 4, 5]
        else:
            model_name = 'sam'


    if label_mode == 'Alphabet':
        label_mode = 'a'
    else:
        label_mode = '1'

    text_size, hole_scale, island_scale=640,100,100
    text, text_part, text_thresh = '','','0.0'
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False

        if mode == "Interactive":
            labeled_array, num_features = label(np.asarray(image['mask'].convert('L')))
            spatial_masks = torch.stack([torch.from_numpy(labeled_array == i+1) for i in range(num_features)])

        if model_name == 'semantic-sam':
            model = model_semsam
            output, mask = inference_semsam_m2m_auto(model, image['image'], level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, *args, **kwargs)

        elif model_name == 'sam':
            model = model_sam
            if mode == "Automatic":
                output, mask = inference_sam_m2m_auto(model, image['image'], text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_sam_m2m_interactive(model, image['image'], spatial_masks, text_size, label_mode, alpha, anno_mode)

        elif model_name == 'seem':
            model = model_seem
            if mode == "Automatic":
                output, mask = inference_seem_pano(model, image['image'], text_size, label_mode, alpha, anno_mode)
            elif mode == "Interactive":
                output, mask = inference_seem_interactive(model, image['image'], spatial_masks, text_size, label_mode, alpha, anno_mode)

        # convert output to PIL image
        history_masks.append(mask)
        history_images.append(Image.fromarray(output))
        return (output, [])


def gpt4v_response(message, history):
    global history_images
    global history_texts; history_texts = []    
    global stage
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
            res = request_gpt4v(prompt1, history_images[0])
            history_texts.append(res)
            stage = 2
        elif stage == 2:
            # locate the object part for grasp
            prompt2 = """
Now look at the center cropped image of the target object. Which labeled component should you grasp? The first line of your output is the label only. The second line explains your choice. ATTENTION: you act as the robot, not the person in the scene.
"""
            res = request_gpt4v(prompt2, history_images[-1])
            history_texts.append(res)
            stage = 1
        return res
    except Exception as e:
        traceback.print_exc()
        return None


def highlight(image, mode, alpha, label_mode, anno_mode, *args, **kwargs):
    res = history_texts[0]
    global bbox
    global wh
    # find the seperate numbers in sentence res
    res = res.split(' ')
    res = [r.replace('.','').replace(',','').replace(')','').replace('"','') for r in res]
    # find all numbers in '[]'
    res = [r for r in res if '[' in r]
    res = [r.split('[')[1] for r in res]
    res = [r.split(']')[0] for r in res]
    res = [r for r in res if r.isdigit()]
    res = list(set(res))
    sections = []
    for i, r in enumerate(res):
        mask_i = history_masks[0][int(r)-1]['segmentation']
        sections.append((mask_i, r))
    # get the mask index
    mask_index1 = int(res[0])
    # get the mask
    mask1 = history_masks[-1][mask_index1-1]
    # save output mask
    if bbox is not None:
        mask_to_save = np.zeros(wh)
        img_mask = Image.fromarray(mask1['segmentation'])
        resized_mask = img_mask.resize((bbox[2]-bbox[0], bbox[3]-bbox[1]))
        print('mask shape', mask1['segmentation'].shape)
        print('resized mask', resized_mask.size)
        print('bbox', bbox)
        print('mask to save shape ', mask_to_save.shape)
        mask_to_save[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np.array(resized_mask)
        np.save('result_mask.npy', mask_to_save)
        bbox = None
    # center crop the mask
    mask_bbox1 = (np.array(mask1['bbox']) * (image['image'].width / history_images[0].width)).astype(np.int)
    print(mask_bbox1)
    margin = 0.1
    wh = (image['image'].height, image['image'].width)
    bbox = (int(mask_bbox1[0]-margin*mask_bbox1[2]), int(mask_bbox1[1]-margin*mask_bbox1[3]), 
            int(mask_bbox1[0]+(1+margin)*mask_bbox1[2]), int(mask_bbox1[1]+(1+margin)*mask_bbox1[3]))
    image1 = image['image'].crop(bbox)
    return (history_images[0], sections), image1

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)

'''
launch app
'''

demo = gr.Blocks()
image = ImageMask(label="Input", type="pil", brush_radius=20.0, brush_color="#FFFFFF", height=512)
# image = gr.Image(label="Input", type="pil", height=512)
slider = gr.Slider(1, 3, value=1.8, label="Granularity") # info="Choose in [1, 1.5), [1.5, 2.5), [2.5, 3] for [seem, semantic-sam (multi-level), sam]"
mode = gr.Radio(['Automatic', 'Interactive', ], value='Automatic', label="Segmentation Mode")
anno_mode = gr.CheckboxGroup(choices=["Mark", "Mask", "Box"], value=['Mark'], label="Annotation Mode")
image_out = gr.AnnotatedImage(label="SoM Visual Prompt",type="pil", height=512)
runBtn = gr.Button("Run")
highlightBtn = gr.Button("Highlight")
bot = gr.Chatbot(label="GPT-4V + SoM", height=256)
slider_alpha = gr.Slider(0, 1, value=0.05, label="Mask Alpha") #info="Choose in [0, 1]"
label_mode = gr.Radio(['Number', 'Alphabet'], value='Number', label="Mark Mode")
image_cropped = gr.ImageMask(label="GPT4V referred",type="pil", height=512)
runCroppedBtn = gr.Button("Run on Cropped Image")
image_cropped_out = gr.AnnotatedImage(label="Second SoM Visual Prompt",type="pil", height=512)
croppedHighlightBtn = gr.Button("Highlight on Cropped Image")

title = "Set-of-Mark (SoM) Visual Prompting for GPT-4V Assisted Functional Grasp"
description = "This is a demo for SoM Prompting to unleash extraordinary visual grounding in GPT-4V. Please upload an image and them click the 'Run' button to get the image with marks. Then giving the task instruction below!"

with demo:
    gr.Markdown("<h1 style='text-align: center'><img src='https://som-gpt4v.github.io/website/img/som_logo.png' style='height:50px;display:inline-block'/>  Set-of-Mark (SoM) Prompting Unleashes Extraordinary Visual Grounding in GPT-4V</h1>")
    # gr.Markdown("<h2 style='text-align: center; margin-bottom: 1rem'>Project: <a href='https://som-gpt4v.github.io/'>link</a>     arXiv: <a href='https://arxiv.org/abs/2310.11441'>link</a>     Code: <a href='https://github.com/microsoft/SoM'>link</a></h2>")
    with gr.Row():
        with gr.Column():
            image.render()
            slider.render()
            with gr.Accordion("Detailed prompt settings (e.g., mark type)", open=False):
                with gr.Row():
                    mode.render()
                    anno_mode.render()
                with gr.Row():
                    slider_alpha.render()
                    label_mode.render()
        with gr.Column():
            image_out.render()
            runBtn.render()
            highlightBtn.render()
    with gr.Row():    
        gr.ChatInterface(chatbot=bot, fn=gpt4v_response)
    with gr.Row():
        with gr.Column():
            image_cropped.render()
        with gr.Column():
            image_cropped_out.render()
            runCroppedBtn.render()
            croppedHighlightBtn.render()

    runBtn.click(inference, inputs=[image, slider, mode, slider_alpha, label_mode, anno_mode],
              outputs = image_out)
    highlightBtn.click(highlight, inputs=[image, mode, slider_alpha, label_mode, anno_mode],
              outputs = [image_out, image_cropped])
    runCroppedBtn.click(inference, inputs=[image_cropped, slider, mode, slider_alpha, label_mode, anno_mode],
                outputs = image_cropped_out)
    croppedHighlightBtn.click(highlight, inputs=[image_cropped, mode, slider_alpha, label_mode, anno_mode],
                outputs = [image_cropped_out, image_cropped])
    
    
                        

demo.queue().launch(share=True,server_port=6092)

