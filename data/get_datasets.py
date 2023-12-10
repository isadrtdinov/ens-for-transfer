import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from data import datasets
from utils.constants import TRAIN_TEST_SPLIT_RANDOM_STATE
from typing import Tuple
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision.datasets import ImageFolder, ImageNet
from torchvision import transforms


DS_TO_METRIC_AND_N_CLASSES = {
    'aircraft': ('mean_per_class_acc', 100),
    'caltech101': ('mean_per_class_acc', 101),
    'cars': ('top1_acc', 196),
    'chestx': ('top1_acc', 7),
    'cifar10': ('top1_acc', 10),
    'cifar100': ('top1_acc', 100),
    'cinic10': ('top1_acc', 10),
    'clipart': ('top1_acc', 345),
    'dtd': ('top1_acc', 47),
    'eurosat': ('top1_acc', 10),
    'flowers': ('mean_per_class_acc', 102),
    'food': ('top1_acc', 101),
    'imagenet': ('top1_acc', 1000),
    'isic': ('top1_acc', 7),
    'pets': ('mean_per_class_acc', 37),
    'sun': ('top1_acc', 397),
    'quickdrawnew': ('top1_acc', 345)
}


DS_TO_SIZES = {
    'aircraft': (3334, 3333, 3333),
    'caltech101': (2525, 505, 5647),
    'cars': (6494, 1650, 8041),
    'chestx': (15593, 3898, 6357),
    'cifar10': (45000, 5000, 10000),
    'cifar100': (45000, 5000, 10000),
    'cinic10': (90000, 90000, 90000),
    'clipart': (26820, 6705, 14604),
    'dtd': (1880, 1880, 1880),
    'eurosat': (10800, 2700, 13500),
    'flowers': (1020, 1020, 6149),
    'food': (68175, 7575, 25250),
    'isic': (7011, 1502, 1502),
    'pets': (2940, 740, 3669),
    'sun': (15880, 3970, 19850),
    'quickdrawnew': (45000, 5000, 51750),
}

# copied from https://github.com/microsoft/robust-models-transfer/blob/master/src/utils/transfer_datasets.py
class TransformedDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.transform = transform
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample, label = self.ds[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, label


def split_set_by_sizes(dataset, targets, sizes: tuple):
    assert len(dataset) == sizes[0] + sizes[1]
    assert len(dataset) == len(targets)

    train_ind, test_ind = train_test_split(
        np.arange(len(dataset)), train_size=sizes[0],
        stratify=targets, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)

    train_set = Subset(dataset, train_ind)
    test_set = Subset(dataset, test_ind)

    return train_set, test_set


def split_targets_by_class(targets, train_img_per_class):
    train_indices = []
    test_indices = []
    labels = np.unique(targets)
    for label in labels:
        ind = np.where(targets == label)[0]
        cur_tr_ind, cur_te_ind = train_test_split(
            ind, train_size=train_img_per_class,
            random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)

        assert type(train_indices) == list
        train_indices += list(cur_tr_ind)
        test_indices += list(cur_te_ind)

    return train_indices, test_indices


def get_aircraft(root, use_test, cache_ram):
    if use_test:
        train_set = datasets.FGVCAircraft(root, split='trainval', download=True, cache_ram=cache_ram)
        test_set = datasets.FGVCAircraft(root, split='test', download=True, cache_ram=cache_ram)
    else:
        train_set = datasets.FGVCAircraft(root, split='train', download=True, cache_ram=cache_ram)
        test_set = datasets.FGVCAircraft(root, split='val', download=True, cache_ram=cache_ram)

    return train_set, test_set


def get_caltech101(root, use_test, cache_ram):
    NUM_TRAINING_SAMPLES_PER_CLASS = 30
    full_set = datasets.Caltech101(root, download=True, cache_ram=cache_ram)

    train_ind, test_ind = split_targets_by_class(full_set.y, NUM_TRAINING_SAMPLES_PER_CLASS)

    if not use_test:
        NUM_VALIDATION_SAMPLES_PER_CLASS = 5

        train_ind, test_ind = split_targets_by_class(
            np.array(full_set.y)[train_ind],
            NUM_TRAINING_SAMPLES_PER_CLASS - NUM_VALIDATION_SAMPLES_PER_CLASS)

    train_set = Subset(full_set, train_ind)
    test_set = Subset(full_set, test_ind)

    return train_set, test_set


def get_cars(root, use_test, cache_ram):

    full_train_set = datasets.StanfordCars(root, split='train', download=True, cache_ram=cache_ram)
    targets = [full_train_set._samples[i][1] for i in range(len(full_train_set._samples))]
    return (
        [full_train_set, datasets.StanfordCars(root, split='test', download=True, cache_ram=cache_ram)] if use_test
        else split_set_by_sizes(full_train_set, targets, sizes=DS_TO_SIZES['cars'][:2]))


def get_cifar10(root, use_test, *unused):
    full_train_set = datasets.CIFAR10(root, train=True, download=True)
    targets = full_train_set.targets
    return (
        [full_train_set, datasets.CIFAR10(root, train=False, download=True)] if use_test
        else split_set_by_sizes(full_train_set, targets, sizes=DS_TO_SIZES['cifar10'][:2]))


def get_cifar100(root, use_test, *unused):
    full_train_set = datasets.CIFAR100(root, train=True, download=True)
    targets = full_train_set.targets
    return (
        [full_train_set, datasets.CIFAR100(root, train=False, download=True)] if use_test
        else split_set_by_sizes(full_train_set, targets, sizes=DS_TO_SIZES['cifar100'][:2]))


def get_cifar10_1(root, use_test, *unused):
    assert use_test
    test_set = datasets.CIFAR10_1(root)
    train_set = [(torch.rand(3, 32, 32), -1)]  # a placeholder for an absent train set
    return [train_set, test_set]


def get_cifar100c(root, use_test, corruption, severity, *unused):
    assert use_test
    test_test = datasets.CIFAR100C(root, corruption, severity)
    train_set = [(torch.rand(3, 32, 32), -1)]  # a placeholder for an absent train set
    return [train_set, test_test]


def get_cinic10(root, use_test, *unused):
    """
    CINIC-10 dataset: https://github.com/BayesWatch/cinic-10
    """
    train_set = ImageFolder(os.path.join(root, 'cinic10', 'train'), transform=transforms.ToTensor())
    val_set = ImageFolder(os.path.join(root, 'cinic10', 'valid'), transform=transforms.ToTensor())
    test_set = ImageFolder(os.path.join(root, 'cinic10', 'test'), transform=transforms.ToTensor())

    return (
        [ConcatDataset([train_set, val_set]), test_set] if use_test
        else [train_set, val_set])

def get_chestx(root, use_test, cache_ram):
    train_set = datasets.ChestX(root, 'train', cache_ram)
    if not use_test:
        train_set, val_set = split_set_by_sizes(
            train_set, train_set._targets,
            (DS_TO_SIZES['chestx'][0], DS_TO_SIZES['chestx'][1])
        )
        return train_set, val_set

    test_set = datasets.ChestX(root, 'test', cache_ram)

    return train_set, test_set

def get_domainnet(root, dataset_name, use_test, cache_ram):
    dataset_dir = dataset_name if dataset_name != 'quickdrawnew' else 'quickdraw'
    train_set = datasets.DomainNet(root, 'train', dataset_dir, cache_ram)
    if dataset_name == 'quickdrawnew':
        train_size = DS_TO_SIZES[dataset_name][0] + DS_TO_SIZES[dataset_name][1]
        train_ind, _ = train_test_split(
            np.arange(len(train_set)), train_size=train_size,
            stratify=train_set._targets, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE
        )
        new_train_set = Subset(train_set, train_ind)
        new_targets = [train_set._targets[x] for x in new_train_set.indices]
        setattr(new_train_set, '_targets', new_targets)
        train_set = new_train_set

    if not use_test:
        train_set, val_set = split_set_by_sizes(
            train_set, train_set._targets,
            (DS_TO_SIZES[dataset_name][0], DS_TO_SIZES[dataset_name][1])
        )
        return train_set, val_set

    test_set = datasets.DomainNet(root, 'test', dataset_dir, cache_ram)

    return train_set, test_set

def get_clipart(root, use_test, cache_ram):
    return get_domainnet(root, 'clipart', use_test, cache_ram)
def get_quickdrawnew(root, use_test, cache_ram):
    return get_domainnet(root, 'quickdrawnew', use_test, cache_ram)

def get_dtd(root, use_test, cache_ram):
    train_set = datasets.DTD(root, split='train', download=True, cache_ram=cache_ram)
    val_set = datasets.DTD(root, split='val', download=True, cache_ram=cache_ram)
    test_set = datasets.DTD(root, split='test', download=True, cache_ram=cache_ram)

    return (
        [ConcatDataset([train_set, val_set]), test_set] if use_test
        else [train_set, val_set])


def get_eurosat(root, use_test, cache_ram):
    full_dataset = datasets.EuroSat(root, cache_ram=cache_ram)
    train_set, test_set = split_set_by_sizes(
        full_dataset, full_dataset._targets,
        (DS_TO_SIZES['eurosat'][0] + DS_TO_SIZES['eurosat'][1], DS_TO_SIZES['eurosat'][2])
    )

    if use_test:
        return train_set, test_set
    else:
        new_targets = [full_dataset._targets[x] for x in train_set.indices]
        train_set, val_set = split_set_by_sizes(
            train_set, new_targets,
            (DS_TO_SIZES['eurosat'][0], DS_TO_SIZES['eurosat'][1])
        )

        return train_set, val_set

def get_flowers(root, use_test, cache_ram):
    train_set = datasets.Flowers102(root, split='train', download=True, cache_ram=cache_ram)
    val_set = datasets.Flowers102(root, split='val', download=True, cache_ram=cache_ram)
    test_set = datasets.Flowers102(root, split='test', download=True, cache_ram=cache_ram)

    return (
        [ConcatDataset([train_set, val_set]), test_set] if use_test
        else [train_set, val_set])


def get_food(root, use_test, *unused):
    # too long caching
    full_train_set = datasets.Food101(root, split='train', download=True, cache_ram=False)
    targets = full_train_set._labels
    return (
        [full_train_set, datasets.Food101(root, split='test', download=True, cache_ram=False)] if use_test
        else split_set_by_sizes(full_train_set, targets, sizes=DS_TO_SIZES['food'][:2]))

def get_imagenet(root, use_test, *unused):
    image_dir = str(os.path.join(root, 'imagenet'))

    train_dir = os.path.join(image_dir, 'train')
    train_set = ImageFolder(train_dir, transform=transforms.ToTensor())

    test_dir = os.path.join(image_dir, 'val')
    test_set = ImageFolder(test_dir, transform=transforms.ToTensor())

    return train_set, test_set

def get_isic(root, use_test, cache_ram):
    full_dataset = datasets.ISIC2018(root, cache_ram=cache_ram)
    train_set, test_set = split_set_by_sizes(
        full_dataset, full_dataset._targets,
        (DS_TO_SIZES['isic'][0] + DS_TO_SIZES['isic'][1], DS_TO_SIZES['isic'][2])
    )

    if use_test:
        return train_set, test_set
    else:
        new_targets = [full_dataset._targets[x] for x in train_set.indices]
        train_set, val_set = split_set_by_sizes(
            train_set, new_targets,
            (DS_TO_SIZES['isic'][0], DS_TO_SIZES['isic'][1])
        )

        return train_set, val_set

def get_pets(root, use_test, cache_ram):
    full_train_set = datasets.OxfordIIITPet(root, split='trainval', download=True, cache_ram=cache_ram)

    if use_test:
        return full_train_set, datasets.OxfordIIITPet(root, split='test', download=True, cache_ram=cache_ram)

    NUM_VALIDATION_SAMPLES_PER_CLASS = 20
    # substract valid_set
    test_ind, train_ind = split_targets_by_class(full_train_set._labels, NUM_VALIDATION_SAMPLES_PER_CLASS)

    train_set = Subset(full_train_set, train_ind)
    test_set = Subset(full_train_set, test_ind)

    return train_set, test_set


def get_sun(root, use_test, cache_ram):
    full_train_set = datasets.SUN397(root, partition=1, train=True, cache_ram=cache_ram)

    if use_test:
        return full_train_set, datasets.SUN397(root, partition=1, train=False, cache_ram=cache_ram)

    NUM_VALIDATION_SAMPLES_PER_CLASS = 10
    # substract valid_set
    test_ind, train_ind = split_targets_by_class(full_train_set._labels, NUM_VALIDATION_SAMPLES_PER_CLASS)

    train_set = Subset(full_train_set, train_ind)
    test_set = Subset(full_train_set, test_ind)

    return train_set, test_set


def get_datasets(root, name, use_test, transforms: Tuple, cache_ram=True):
    if 'cifar100c' in name:
        split_name = name.split('-')
        assert len(split_name) == 3
        corruption = split_name[1]
        severity = int(split_name[2])
        train_set, test_set = get_cifar100c(root, use_test, corruption, severity, cache_ram)
    else:
        train_set, test_set = globals()[f'get_{name}'](root, use_test, cache_ram)

    return TransformedDataset(train_set, transforms[0]), TransformedDataset(test_set, transforms[1])
