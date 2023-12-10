import os
import torch
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as T
from typing import Any, Callable, Optional, Union, Tuple

from torch.utils.data import Dataset


class ISIC2018(Dataset):
    def __init__(self, root, cache_ram):

        self.root = str(pathlib.Path(root).expanduser())
        self.img_path = os.path.join(self.root, "ISIC2018_Task3_Training_Input/")
        self.csv_path = os.path.join(self.root, "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")

        # Transforms
        self._to_tensor = T.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name = np.asarray(self.data_info.iloc[:, 0])
        self._images = [os.path.join(self.img_path, name + ".jpg") for name in self.image_name]

        self.labels = np.asarray(self.data_info.iloc[:, 1:])
        self._targets = (self.labels != 0).argmax(axis=1)
        # Calculate len
        self.data_len = len(self.data_info.index)

        self._cache_ram = cache_ram
        self._cached_images = []
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
