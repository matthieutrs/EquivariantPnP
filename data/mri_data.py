import json
from pathlib import Path

import numpy as np

import torch
from deepinv.utils.demo import MRIData
from torch.utils.data import DataLoader
from torchvision import transforms

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

with open('config/config.json') as json_file:
    config = json.load(json_file)

PATH_MRI_DATA = config['PATH_MRI_DATA']


def create_vertical_lines_mask(image_shape=(320, 320), acceleration_factor=4, seed=0):
    np.random.seed(seed)
    if acceleration_factor == 4:
        central_lines_percent = 0.08
        num_lines_center = int(central_lines_percent * image_shape[-1])
        side_lines_percent = 0.25 - central_lines_percent
        num_lines_side = int(side_lines_percent * image_shape[-1])
    if acceleration_factor == 8:
        central_lines_percent = 0.04
        num_lines_center = int(central_lines_percent * image_shape[-1])
        side_lines_percent = 0.125 - central_lines_percent
        num_lines_side = int(side_lines_percent * image_shape[-1])
    mask = np.zeros(image_shape)
    center_line_indices = np.linspace(image_shape[0] // 2 - num_lines_center // 2,
                                       image_shape[0] // 2 + num_lines_center // 2 + 1, dtype=np.int32)
    mask[:, center_line_indices] = 1
    random_line_indices = np.random.choice(image_shape[0], size=(num_lines_side // 2,), replace=False)
    mask[:, random_line_indices] = 1
    return torch.from_numpy(mask).type(Tensor)




def get_data(img_size=128):

    # Set the global random seed from pytorch to ensure reproducibility of the example.
    torch.manual_seed(0)

    PTH_DATASET = Path(PATH_MRI_DATA)
    transform = transforms.Compose([transforms.Resize(img_size)])

    test_dataset = MRIData(PTH_DATASET, transform=transform, train=False)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1)

    return test_dataloader
