import os
import glob

import pickle
import imageio
import tqdm
from torch.utils.data import Dataset

from .utils import dl2ld


class ImageDataset(Dataset):
    """
    Args:
        data_dirs: Dictionary of data directories.
            Ex) {'hr': HR_IMAGE_DIR, 'lr': LR_IMAGE_DIR}
    """
    def __init__(self, data_dirs, transform=None, is_binary=True, train=True):
        super().__init__()
        self.transform = transform
        self.is_binary = is_binary
        self.train = train
        self.files = self._get_files(data_dirs)

    def _scan_dir(self, data_path, bin_path=None):
        imgs = sorted(glob.glob(os.path.join(data_path, '*.png')))

        print(f"Scanning data directory {data_path}")
        print(f"Find {len(imgs)} images.")

        if self.is_binary:
            if bin_path is None:
                base = os.path.basename(data_path)
            else:
                base = bin_path

            bin_path = data_path.replace(base, os.path.join(base, 'bin'))
            bin_ext = 'pt'
            os.makedirs(bin_path, exist_ok=True)

            for img in tqdm.tqdm(imgs, ncols=80):
                name = os.path.basename(img)
                bin_name = name.replace('png', bin_ext)
                bin_name = os.path.join(bin_path, bin_name)
                if not os.path.isfile(bin_name):
                    x = imageio.imread(img)
                    with open(bin_name, 'wb') as f:
                        pickle.dump(x, f)
            imgs = sorted(glob.glob(os.path.join(bin_path, '*.' + bin_ext)))

        return imgs

    def _get_files(self, data_dirs):
        files = {}
        for k, data_dir in data_dirs.items():
            files[k] = self._scan_dir(data_dir)
        # Dictionary of lists to list of dictionaries
        files = dl2ld(files)
        return files

    def _load_data(self, filename):
        if self.is_binary:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            return imageio.imread(filename)

    def __getitem__(self, index):
        paths = self.files[index]
        data = {}
        for k, path in paths.items():
            data[k] = self._load_data(path)
        if self.transform:
            data = self.transform(data)
        data['index'] = index
        return data

    def __len__(self):
        return len(self.files)
