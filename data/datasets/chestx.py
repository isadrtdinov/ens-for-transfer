import os
import torch
import pathlib
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as T
from typing import Any, Callable, Optional, Union, Tuple

from torch.utils.data import Dataset


class ChestX(Dataset):
    def __init__(self, root, split, cache_ram):
        """
        Args:
            root (string): path to dataset
        """
        self.root = str(os.path.join(root, "chestx"))
        self.img_path = self.root + "/images_256/"
        self.csv_path = self.root + "/Data_Entry_2017.csv"

        txt_file_name = 'train_val_list.txt' if split == 'train' else 'test_list.txt'
        self.txt_path = self.root + "/" + txt_file_name

        split_images = []
        with open(self.txt_path) as f:
            for line in f:
                split_images.append(line.strip())
        split_images_idx = 0

        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}

        labels_set = []

        # Transforms
        self._to_tensor = T.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self._images = []
        self._targets = []

        for name, label in zip(self.image_name_all, self.labels_all):
            label = label.split("|")

            if split_images_idx == len(split_images):
                break
            if name != split_images[split_images_idx]:
                continue
            else:
                split_images_idx += 1

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self._targets.append(self.labels_maps[label[0]])
                self._images.append(os.path.join(self.img_path, name))

        self.data_len = len(self._targets)

        self._cache_ram = cache_ram
        self._cached_images = []
        self._to_tensor = T.ToTensor()
        if self._cache_ram:
            for image_path in self._images:
                self._cached_images.append(
                    self._to_tensor(Image.open(image_path).resize((256, 256)).convert("RGB"))
                )

    def __getitem__(self, idx):
        # Get image name from the pandas df
        if self._cache_ram:
            image = self._cached_images[idx]
        else:
            image = self._to_tensor(Image.open(self._images[idx]).resize((256, 256)).convert("RGB"))

        target = self._targets[idx]

        return image, target

    def __len__(self):
        return self.data_len
