from gpt4v import request_gpt4v_multi_image_behavior
import traceback
import os
import argparse

def gpt4v_response(instruction, image, prompt_path):
    for i in range(3):
        try:
            prompts, visions = [], []
            for j in range(4):
                with open(f'{prompt_path}/prompt{j + 1}.txt', 'r') as f:
                    prompts.append(f.read())
                if j < 3:
                    visions.append(f'{prompt_path}/prompt{j + 1}.png')
            prompts[-1] += instruction
            visions.append(image)
            res = request_gpt4v_multi_image_behavior(prompts, visions)
            return res
        except Exception as e:
            traceback.print_exc()
            continue

parser = argparse.ArgumentParser(description='Process some instructions and images.')
parser.add_argument('--instruction_base_dir', type=str, required=True, help='Base directory for instructions')
parser.add_argument('--results_image_base_dir', type=str, required=True, help='Base directory for result images')
parser.add_argument('--task', type=str, required=True, help='Task name')

args = parser.parse_args()

instruction_base_dir = args.instruction_base_dir
results_image_base_dir = args.results_image_base_dir
task = args.task

with open(os.path.join(instruction_base_dir, task, 'instruction.txt'), 'r') as f:
    instruction = f.read()
response = gpt4v_response(instruction, os.path.join(results_image_base_dir, task, 'final_output_add_mask.png'), 'behavior')

print(response)
with open(os.path.join(results_image_base_dir, task, 'constraint_response.txt'), 'w') as f:
    f.write(response)