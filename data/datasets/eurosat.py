import os
import os.path
import pathlib
from typing import Any, Callable, Optional, Union, Tuple
from typing import Sequence

from PIL import Image

import torchvision.transforms
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

class EuroSat(VisionDataset):

    def __init__(
        self,
        root: str,
        cache_ram: bool = False,
    ):

        self.root = pathlib.Path(root).expanduser() / "2750"
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self._images = []
        self._targets = []

        for froot, _, fnames in sorted(os.walk(self.root, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(froot, fname)
                self._images.append(path)
                target = self.class_to_idx[pathlib.Path(path).parts[-2]]
                self._targets.append(target)

        self._cache_ram = cache_ram
        self._cached_images = []
        self._to_tensor = torchvision.transforms.ToTensor()
        if self._cache_ram:
            for image_path in self._images:
                self._cached_images.append(
                    self._to_tensor(Image.open(image_path).convert("RGB"))
                )

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if self._cache_ram:
            image = self._cached_images[idx]
        else:
            image = self._to_tensor(Image.open(self._images[idx]).convert("RGB"))

        target = self._targets[idx]

        return image, target