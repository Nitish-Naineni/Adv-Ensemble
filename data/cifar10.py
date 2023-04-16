import torch
import torchvision
import numpy as np
import os
import lmdb
from PIL import Image
from .transformations import get_transfom

DATA_DESC = {
    'data': 'cifar10',
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465], 
    'std': [0.2023, 0.1994, 0.2010],
}

class LMDBDataset(torch.utils.data.Dataset):
    """
    A dataset with auxiliary pseudo-labeled data stored in a LMDB
    """
    def __init__(self, root, transform=None, **kwargs):
        self.env = lmdb.open(root, readonly=True, readahead=False, meminit=False, max_readers=2048, map_size=206333317120)
        self.length = self.env.stat()['entries']
        self.keys = [f"{i:08}".encode('ascii') for i in range(self.length)]
        self.transform = transform
        self.dataset_size = 50000
        self.sup_indices = list(range(self.dataset_size))
        self.unsup_indices = list(range(self.dataset_size, self.length))

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False, buffers=False) as txn:
            value = txn.get(self.keys[index])
        img_bytes = value[:3072]
        target_bin = value[3072:]
        img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(32, 32, 3)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        target = int(target_bin)
        return img, target

    def __len__(self):
        return self.length


def load_cifar10(data_dir, augmentation='base'):
    """
    Arguments:
        data_dir (str): path to data directory.
        augmentation: use different augmentations for training set.
    Returns:
        train dataset, test dataset, validation dataset. 
    """
    train_transform = get_transfom(augmentation=augmentation)
    test_transform = get_transfom(augmentation='none')

    train_dataset = LMDBDataset(root=data_dir, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir,train=False, download=True, transform=test_transform)
    val_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform)
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, 1024))  # split from training set
    return train_dataset, test_dataset, val_dataset
 