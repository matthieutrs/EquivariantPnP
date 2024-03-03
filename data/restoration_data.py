import numpy as np
import scipy

import json

from torch.utils.data import DataLoader
import torch
from pathlib import Path
from torchvision import transforms
from deepinv.utils.demo import load_dataset

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

with open('config/config.json') as json_file:
    config = json.load(json_file)

ROOT_DATASET = Path(config['ROOT_DATASET'])

def get_blur_kernel(id=1):
    pth_blur = ROOT_DATASET / str('blur_models/blur_'+str(id)+'.mat')
    h = scipy.io.loadmat(pth_blur)
    h = torch.from_numpy(np.array(h['blur'])).unsqueeze(0).unsqueeze(0)
    return h


def get_mask_sr(x, sr_fact=2):
    mask = torch.zeros_like(x)
    mask[..., ::sr_fact, ::sr_fact] = 1
    mask = mask[0]
    return mask


def init_sr_solution(y, sr_fact=2):
    x_sr = y.clone()
    for _ in range(sr_fact-1):
        x_sr[..., _+1::sr_fact, :] = y[..., ::sr_fact, :]
        x_sr[..., _+1::sr_fact] = x_sr[..., ::sr_fact]
    return x_sr


def get_data(dataset_name='set3c', batch_size=1, img_size=None, n_channels=None):

    # Set the global random seed from pytorch to ensure reproducibility of the example.
    torch.manual_seed(0)

    test_transform = []

    if img_size is not None:
        # test_transform.append(transforms.CenterCrop(img_size))
        test_transform.append(transforms.Resize(img_size))
    if n_channels is not None and n_channels == 1:
        test_transform.append(transforms.Grayscale(num_output_channels=1))
    test_transform.append(transforms.ToTensor())
    test_transform = transforms.Compose(test_transform)

    test_dataset = load_dataset(dataset_name, ROOT_DATASET, test_transform, download=False)

    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True
    )

    return test_dataloader
