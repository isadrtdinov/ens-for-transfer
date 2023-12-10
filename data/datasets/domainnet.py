import os
import torch
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as T
from typing import Any, Callable, Optional, Union, Tuple

from torch.utils.data import Dataset


class DomainNet(Dataset):

    def __init__(self, root, split, dataset_name, cache_ram):

        self.root = str(pathlib.Path(root).expanduser())
        self.img_path = os.path.join(self.root, dataset_name)
        file_name = os.path.join(self.img_path, f'{dataset_name}_{split}.txt')

        self._images = []
        self._targets = []
        with open(file_name) as f:
            for line in f:
                cur_line = line.strip().split(' ')
                assert len(cur_line) == 2
                self._images.append(str(os.path.join(self.root, cur_line[0])))
                self._targets.append(int(cur_line[1]))

        self.data_len = len(self._targets)
        self._cache_ram = cache_ram
        self._cached_images = []
        self._to_tensor = T.ToTensor()
        if self._cache_ram:
            for image_path in self._images:
                self._cached_images.append(
                    self._to_tensor(Image.open(image_path).convert("RGB"))
                )

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if self._cache_ram:
            image = self._cached_images[idx]
        else:
            image = self._to_tensor(Image.open(self._images[idx]).convert("RGB"))

        target = self._targets[idx]

        return image, target

    def __len__(self):
        return self.data_len
