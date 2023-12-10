# modified from https://github.com/microsoft/Swin-Transformer/blob/main/data/build.py

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def get_mixup_fn(config):

    mixup_fn = None
    mixup_active = config.aug.mixup > 0 or config.aug.cutmix > 0. or config.aug.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.aug.mixup, cutmix_alpha=config.aug.cutmix, cutmix_minmax=config.aug.cutmix_minmax,
            prob=config.aug.mixup_prob, switch_prob=config.aug.mixup_switch_prob, mode=config.aug.mixup_mode,
            label_smoothing=config.label_smoothing, num_classes=config.num_classes)

    return mixup_fn


def build_transform(is_train, config):
    resize_im = config.img_size > 32

    # convert ds to pil image
    to_pil_im = transforms.ToPILImage()

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.img_size,
            is_training=True,
            color_jitter=config.aug.color_jitter if config.aug.color_jitter > 0 else None,
            auto_augment=config.aug.auto_augment if config.aug.auto_augment != 'none' else None,
            re_prob=config.aug.reprob,
            re_mode=config.aug.remode,
            re_count=config.aug.recount,
            interpolation=config.interpolation,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.img_size, padding=4)
        return transforms.Compose([to_pil_im, transform])

    t = []
    if resize_im:
        if config.test_crop:
            size = int((256 / 224) * config.img_size)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.interpolation)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.img_size))
        else:
            t.append(
                transforms.Resize((config.img_size, config.img_size),
                                  interpolation=_pil_interp(config.interpolation))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose([to_pil_im] + t)
