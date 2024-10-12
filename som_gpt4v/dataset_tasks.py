from pathlib import Path
from PIL import Image
import numpy as np

# Path to the dataset
dataset_path = Path('dataset')

class DatasetTasks:
    def __init__(self, task_name) -> None:
        self.task_name = task_name
        self.dataset_path = dataset_path / task_name
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset {task_name} not found at {self.dataset_path}")
        self.data_list = list((self.dataset_path / 'pointcloud').glob('*.npy'))
        
    def __getitem__(self, idx):
        data_name = self.data_list[idx].stem
        color1 = Image.open(self.dataset_path / 'color' / f'{data_name}_color1.png')
        color2 = Image.open(self.dataset_path / 'color' / f'{data_name}_color2.png')
        depth1 = np.load(self.dataset_path / 'depth' / f'{data_name}_depth1.npy')
        depth2 = np.load(self.dataset_path / 'depth' / f'{data_name}_depth2.npy')
        pointcloud = np.load(self.dataset_path / 'pointcloud' / f'{data_name}.npy')
        return color1, color2, depth1, depth2, pointcloud
    
    def __len__(self):
        return len(self.data_list)
    

# Example usage
if __name__ == '__main__':
    dataset = DatasetTasks('dataset_button')
    print(len(dataset))
    for i in range(len(dataset)):
        color1, color2, depth1, depth2, pointcloud = dataset[i]
        print(color1.size, color2.size, depth1.shape, depth2.shape, pointcloud.shape)
        break