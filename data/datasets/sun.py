import PIL.Image
from pathlib import Path
from typing import Any, Tuple, Callable, Optional
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms


class SUN397(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.
    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.
    Args:
        root (string): Root directory of the dataset.
        partition (int): Train-test dataset partition (must be in range 1-10),
            leave None for full split.
        train (bool): Whether to use training part, works only with partition set.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        cache_ram (bool, optional): If True, images are preloaded to RAM during initialization.
    """
    def __init__(
        self,
        root: str,
        partition: int = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_ram: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._cache_ram = cache_ram
        self._data_dir = Path(self.root) / "sun397"

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        if partition is not None:
            split = "Training" if train else "Testing"
            partition_file = self._data_dir / f"Partitions/{split}_{partition:02d}.txt"
            self._image_files = [
                self._data_dir / path[1:] for path in open(partition_file).read().splitlines()
            ]
        else:
            self._image_files = list(self._data_dir.rglob("sun_*.jpg"))

        self._labels = [
            self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in self._image_files
        ]

        self._to_tensor = transforms.ToTensor()
        self._cached_images = []
        if self._cache_ram:
            for image_file in self._image_files:
                self._cached_images.append(
                    self._to_tensor(PIL.Image.open(image_file).convert("RGB"))
                )

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        if self._cache_ram:
            image = self._cached_images[idx]
        else:
            image = self._to_tensor(PIL.Image.open(image_file).convert("RGB"))

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
