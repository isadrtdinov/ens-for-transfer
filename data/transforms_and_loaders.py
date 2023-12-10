from copy import deepcopy
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.get_datasets import get_datasets
from .swin_transforms import build_transform


def _pil_interp(method):
    if method == 'bicubic':
        return transforms.InterpolationMode.BICUBIC
    elif method == 'lanczos':
        return transforms.InterpolationMode.LANCZOS
    elif method == 'hamming':
        return transforms.InterpolationMode.HAMMING
    else:
        return transforms.InterpolationMode.BILINEAR


def get_train_and_test_transforms(config):
    transform_name = config.transform_name.lower()
    if transform_name in ['imagenet', 'cifar']:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if transform_name == 'cifar':
            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                normalize,
            ])
            test = transforms.Compose([
                normalize,
            ])
        else:
            interpolation_mode = _pil_interp(config.interpolation)
            train = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=interpolation_mode),
                transforms.RandomHorizontalFlip(),
                normalize,
            ])
            test = transforms.Compose([
                transforms.Resize(256, interpolation=interpolation_mode),
                transforms.CenterCrop(224),
                normalize,
            ])
    elif transform_name == 'swin_imagenet':
        train = build_transform(is_train=True, config=config)
        test = build_transform(is_train=False, config=config)
    else:
        raise NotImplementedError(f"Unknown transform name {transform_name}")

    return train, test


def get_dataloaders(args, use_test=False, unshuffled_train=False, augment_train=True, cache_ram=True):
    name = args.dataset.lower()
    train_transform, test_transform = get_train_and_test_transforms(config=args)
    transforms = (train_transform, test_transform)

    train_set, test_set = get_datasets(
        args.data_path, name, use_test=use_test, transforms=transforms,
        cache_ram=cache_ram
    )

    dataloaders = {
        'train': DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=getattr(args, 'drop_last', False),
        ),
        'test': DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        ),
    }

    if unshuffled_train:
        if not augment_train:
            unshuffled_train = deepcopy(train_set)
            unshuffled_train.transform = test_transform
        else:
            unshuffled_train = train_set

        dataloaders['unshuffled_train'] = DataLoader(
            unshuffled_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    return dataloaders
