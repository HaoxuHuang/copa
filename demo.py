import subprocess
import argparse

parser = argparse.ArgumentParser(description='Quick demo of CoPa.')
parser.add_argument('task', type=str, help='Task to run (e.g. "flower", "button")')
args = parser.parse_args()
task = args.task

# Task-oriented Grasping
## Ground grasp part
subprocess.run(['python', 'som_gpt4v/main_grasp.py', task])

## Generate and filter grasp candidates
subprocess.run(['python', 'graspnet/demo_filter_grasp.py',
                '--candidates_path', f'data/{task}/grasp.npy',
                '--pointcloud_path', f'data/{task}/pointcloud1.npy',
                '--mask_path', f'data/{task}/mask.npy',
                '--output_path', f'data/{task}/grasp.npy',])
## Uncomment and replace above if you want to generate grasp candidates
# subprocess.run(['python', 'graspnet/main_to_ros.py',
#                 '--checkpoint_path', 'graspnet/checkpoint-rs.tar',
#                 '--pointcloud_path', f'data/{task}/pointcloud1.npy',
#                 '--mask_path', f'data/{task}/mask.npy',
#                 '--output_path', f'data/{task}/grasp.npy',])

# Task-aware Motion Planning
## Extract geometric elements
subprocess.run(['python', 'som_gpt4v/main_behavior.py',
                '--image_base_dir', 'data',
                '--task', task,
                '--base_output_dir', 'data',])
## Generate spatial constraints
subprocess.run(['python', 'som_gpt4v/main_constraint.py',
                '--instruction_base_dir', 'data',
                '--results_image_base_dir', 'data',
                '--task', task,])
## Generate target pose
subprocess.run(['python', 'constraint_solver/main_infer_trans.py',
                '--spatial_data', f'data/{task}/spatial_data.pkl',
                '--constraints', f'data/{task}/constraint_response.txt',
                '--output', f'data/{task}/transform.npy',])