import torch
import os
import numpy as np

from .cifar10 import load_cifar10

DATASETS = ['cifar10']

_LOAD_DATASET_FN = {
    'cifar10': load_cifar10
}

class SemiSupervisedSampler(torch.utils.data.Sampler):
    """
    Reference : https://github.com/wzekai99/DM-Improves-AT/blob/main/core/data/semisup.py
    Balanced sampling from the labeled and unlabeled data.
    """
    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5, num_batches=None):
        if unsup_fraction is None or unsup_fraction < 0:
            self.sup_inds = sup_inds + unsup_inds
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

        self.batch_size = batch_size
        unsup_batch_size = int(batch_size * unsup_fraction)
        self.sup_batch_size = batch_size - unsup_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(np.ceil(len(self.sup_inds) / self.sup_batch_size))
        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            sup_inds_shuffled = [self.sup_inds[i] for i in torch.randperm(len(self.sup_inds))]
            for sup_k in range(0, len(self.sup_inds), self.sup_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = sup_inds_shuffled[sup_k:(sup_k + self.sup_batch_size)]
                if self.sup_batch_size < self.batch_size:
                    batch.extend([self.unsup_inds[i] for i in torch.randint(
                        high=len(self.unsup_inds), 
                        size=(self.batch_size - len(batch),), 
                        dtype=torch.int64)
                    ])
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches

def load_data(data_dir="./data/database/",dataset="cifar10", batch_size=256, batch_size_test=256, num_workers=4, augmentation='base', 
                shuffle=True, unsup_fraction=None):
    """
    Returns dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        batch_size (int): batch size for training.
        batch_size_test (int): batch size for validation.
        num_workers (int): number of workers for loading the data.
        augmentation (base/none): whether to use augmentations for training set.
        shuffle_train (bool): whether to shuffle training set.
        aux_data_filename (str): path to unlabelled data.
        unsup_fraction (float): fraction of unlabelled data per batch.
    """
    load_dataset_fn = _LOAD_DATASET_FN[dataset]

    train_dataset, test_dataset, val_dataset = load_dataset_fn(
        data_dir=data_dir, 
        augmentation=augmentation
    )

    dataset_size = train_dataset.dataset_size
    train_batch_sampler = SemiSupervisedSampler(
        sup_inds=train_dataset.sup_indices, 
        unsup_inds=train_dataset.unsup_indices, 
        batch_size=batch_size, 
        unsup_fraction=unsup_fraction, 
        num_batches=int(np.ceil(dataset_size/batch_size))
    )
    epoch_size = len(train_batch_sampler) * batch_size

    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, **kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, **kwargs)
   
    return train_dataloader, test_dataloader, val_dataloader

