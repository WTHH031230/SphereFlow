import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)

class BraTS2023Dataset(Dataset):
    def __init__(self, json_path, phase='train', transform=None, modality_index=None):
        """
        Dataset for BraTS 2023 2D slices.
        modality_index: int (0-3) or None. If 0-3, return only that modality repeated 3 times.
        """
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)[phase]
        
        self.phase = phase
        self.transform = transform
        self.modality_index = modality_index

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 1. Load data from .npy file
        # The structure is assumed to be {'image': (4, H, W), 'label': (H, W), ...}
        item_path = self.data_list[idx]['path']
        data_dict = np.load(item_path, allow_pickle=True).item()
        
        # 2. Prepare dictionary for MONAI transforms
        # MONAI expects keys to apply transforms.
        # Ensure image is (C, H, W). Label is (1, H, W) or (H, W).
        image = data_dict['image'] # (4, H, W)
        
        if self.modality_index is not None:
            # Select specific modality and repeat to 3 channels
            image = image[self.modality_index:self.modality_index+1] # (1, H, W)
            image = np.repeat(image, 3, axis=0) # (3, H, W)

        sample = {
            'image': image, 
            'label': data_dict['label'][None] # Add channel dim to label: (1, H, W)
        }

        # 3. Apply Transforms
        if self.transform:
            sample = self.transform(sample)
            
        return sample['image'], 0 # Return image and a dummy label to match ImageFolder format


class BraTS2023AllDataset(Dataset):
    """Returns all modalities (each as 3-ch) from the same sample.

    Output: (image_0, image_1, ...), each (3, H, W).
    """

    def __init__(self, json_path, phase='train', transform=None, num_modalities=4):
        with open(json_path, 'r') as f:
            self.data_list = json.load(f)[phase]
        self.phase = phase
        self.transform = transform
        self.num_modalities = num_modalities

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item_path = self.data_list[idx]['path']
        data_dict = np.load(item_path, allow_pickle=True).item()
        image = data_dict['image']  # (N, H, W)
        sample = {
            'image': image,  # (N, H, W)
            'label': data_dict['label'][None], # (1, H, W)
        }
        if self.transform:
            sample = self.transform(sample)
        image = sample['image']
        images_out = []
        for m in range(self.num_modalities):
            img_m = image[m:m + 1]  # (1, H, W)
            if hasattr(img_m, 'repeat') and not isinstance(img_m, np.ndarray):
                img_m = img_m.repeat(3, 1, 1)
            else:
                img_m = np.repeat(img_m, 3, axis=0)
            images_out.append(img_m)
        return tuple(images_out)


def get_transforms(phase='train'):
    """
    Returns MONAI transforms based on MICCAI 2025 settings.
    """
    if phase == 'train':
        return Compose([
            ToTensord(keys=['image', 'label']),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
            RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=3),
        ])
    else:
        # Validation/Test: No augmentation, just tensor conversion
        return Compose([
            ToTensord(keys=['image', 'label']),
        ])


def get_loader(json_path, phase='train', batch_size=16, num_workers=4, drop_last=False):
    """DataLoader over BraTS2023Dataset with all modalities; shuffled only for 'train'."""
    dataset = BraTS2023Dataset(json_path=json_path, phase=phase, transform=get_transforms(phase))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
