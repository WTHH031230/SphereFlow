"""BraTS 2023 3D dataset.

Each `.npy` file is a 2D slice with structure:
    {
        'image':     np.ndarray, shape (4, 256, 256), float32 in [0, 1]
                     channels = [T1, T1ce, T2, FLAIR]
        'label':     np.ndarray, shape (3, 256, 256), uint8 in {0, 1}
                     channels = [WT, TC, ET]
        'case_name': str,
        'slice_idx': int,
    }

Each case has exactly 80 consecutive slices, which we stack along the depth
axis to form a (4, 80, 256, 256) volume for the image and a
(3, 80, 256, 256) volume for the label.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler


def _group_by_case(items):
    """Group slice records by case name and sort by slice index."""
    grouped = defaultdict(list)
    for it in items:
        grouped[it['case']].append(it)
    cases = []
    for case_name, recs in grouped.items():
        recs_sorted = sorted(recs, key=lambda x: x['slice_idx'])
        cases.append({'case': case_name, 'records': recs_sorted})
    cases.sort(key=lambda x: x['case'])
    return cases


class BraTS3DDataset(Dataset):
    """3D BraTS dataset that returns (4, D, H, W) image and (3, D, H, W) label."""

    def __init__(
        self,
        json_path: str,
        phase: str = 'train',
        transform: Optional[Callable] = None,
        expected_depth: int = 80,
    ):
        super().__init__()
        with open(json_path, 'r') as f:
            split = json.load(f)
        if phase not in split:
            raise ValueError(f"phase '{phase}' not found in {json_path}")
        self.phase = phase
        self.transform = transform
        self.expected_depth = expected_depth
        self.cases = _group_by_case(split[phase])
        for c in self.cases:
            if len(c['records']) != expected_depth:
                raise RuntimeError(
                    f"Case {c['case']} has {len(c['records'])} slices, "
                    f"expected {expected_depth}."
                )

    def __len__(self) -> int:
        return len(self.cases)

    def _load_volume(self, records):
        imgs, labels = [], []
        for rec in records:
            data = np.load(rec['path'], allow_pickle=True).item()
            imgs.append(data['image'])    # (4, H, W)
            labels.append(data['label'])  # (3, H, W)
        image = np.stack(imgs, axis=1).astype(np.float32)   # (4, D, H, W)
        label = np.stack(labels, axis=1).astype(np.float32)  # (3, D, H, W)
        return image, label

    def __getitem__(self, idx):
        case = self.cases[idx]
        image, label = self._load_volume(case['records'])
        sample = {
            'image': image,
            'label': label,
            'case_name': case['case'],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class NoPadDistributedSampler(Sampler[int]):
    """Distributed sampler that shards without padding/duplicating samples.

    PyTorch's default ``DistributedSampler`` pads eval datasets so each rank
    sees the same number of batches. That makes the aggregate sample count drift
    from the real dataset size (e.g. 251 -> 252 on 4 GPUs). For validation and
    test we want exact case counts instead, so we shard by striding indices.
    """

    def __init__(
        self,
        dataset_len: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        if num_replicas is None:
            if not (dist.is_available() and dist.is_initialized()):
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not (dist.is_available() and dist.is_initialized()):
                rank = 0
            else:
                rank = dist.get_rank()
        if num_replicas <= 0:
            raise ValueError("num_replicas must be > 0")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.dataset_len = int(dataset_len)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_len, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_len))
        return iter(indices[self.rank::self.num_replicas])

    def __len__(self):
        if self.dataset_len <= self.rank:
            return 0
        return int(math.ceil((self.dataset_len - self.rank) / self.num_replicas))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


def get_dataloader(
    json_path: str,
    phase: str = 'train',
    batch_size: int = 1,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    distributed: bool = False,
    pin_memory: bool = True,
    expected_depth: int = 80,
):
    dataset = BraTS3DDataset(
        json_path=json_path,
        phase=phase,
        transform=transform,
        expected_depth=expected_depth,
    )

    sampler = None
    shuffle = (phase == 'train')
    if distributed:
        if phase == 'train':
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            sampler = NoPadDistributedSampler(
                dataset_len=len(dataset),
                shuffle=False,
            )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(phase == 'train'),
        persistent_workers=(num_workers > 0),
    )
    return loader, sampler, dataset


if __name__ == '__main__':
    json_path = os.environ.get('BRATS_JSON', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'brats2023_distribution.json'))
    ds = BraTS3DDataset(json_path=json_path, phase='val')
    print(f'Number of cases: {len(ds)}')
    sample = ds[0]
    print(f"Case: {sample['case_name']}")
    print(f"Image: shape={sample['image'].shape}, dtype={sample['image'].dtype}, "
          f"min={sample['image'].min():.4f}, max={sample['image'].max():.4f}")
    print(f"Label: shape={sample['label'].shape}, dtype={sample['label'].dtype}, "
          f"unique={np.unique(sample['label']).tolist()}")
