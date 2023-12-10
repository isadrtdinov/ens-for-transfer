import math
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed


data_root = "./datasets"
target_pixels = 120000
num_workers = 8

data_dir = Path(data_root) / "sun397"
image_files = list(data_dir.rglob("sun_*.jpg"))


def process_image(path):
    image = Image.open(path).convert("RGB")
    height, width = image.size
    actual_pixels = height * width

    if target_pixels and actual_pixels > target_pixels:
        factor = math.sqrt(target_pixels / actual_pixels)
        new_size = (int(height * factor), int(width * factor))
        image = image.resize(new_size, resample=Image.Resampling.BILINEAR)
        image.save(path, 'jpeg', quality=72)


files = list(data_dir.rglob("sun_*.jpg"))
Parallel(n_jobs=num_workers)(delayed(process_image)(path) for path in tqdm(files))

