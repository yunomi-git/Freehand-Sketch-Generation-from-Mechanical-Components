import os
import numpy as np
import random
from PIL import Image
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

try:
    import pickle5 as pickle
except ModuleNotFoundError:
    import pickle

from utils.sketch_utils import *
from utils.my_utils import *

from paths import DirectoryPathManager

class SketchDataset(Dataset):
    def __init__(self, args, sketch_data_file, data_root, transform, contour_root, augment=False, test_only=False, num_test=100):
        self.paths_dict = {}
        self.augment = augment
        self.transform = transform
        self.contour_root = contour_root
        self.label_data_source = args.label_data_source
        self.test_only = test_only

        if test_only:
            self.test_imgs = []
            path_manager = DirectoryPathManager(base_path=data_root + "/", base_unit_is_file=True)
            for file in path_manager.file_paths[:num_test]:
                if file.extension == ".png" or file.extension == ".jpg":
                    self.test_imgs.append(file.as_absolute())
            return

        if self.label_data_source == 'init_clipasso':
            paths_dict_ = get_path_dict(args, data_root)
        else:
            sketch_dir = sketch_data_file
            # sketch_dir = os.path.join(sketch_data_file, f"path.pkl")
            with open(sketch_dir, "rb") as f:
                paths_dict_ = pickle.load(f)

        for key, val in paths_dict_.items():        # key: [idx]_[seed]
            # discard if it does not contain information about both the initial stroke and L intermediate strokes.
            if len(val['iterations']) != 9:
                continue

            # change
            data_idx, seed = key.split("_")
            data_idx = int(data_idx)
            seed = int(seed)
            if data_idx in self.paths_dict:
                self.paths_dict[data_idx][seed] = val
            else:
                self.paths_dict[data_idx] = {seed: val}

    def __getitem__(self, index):
        if self.test_only:
            img_path = self.test_imgs[index]
            image = Image.open(img_path).convert('RGB')
            res_img = self.transform(image)
            return res_img, img_path

        # NOTE: seed set to be 0 by default
        path = self.paths_dict[index][0]['iterations']
        pos_list = []
        for idx in sorted(map(int, path.keys())):
            pos = torch.tensor(path[f"{idx}"]["pos"])
            pos_list.append(pos)
        res_pos = torch.stack(pos_list, dim=0)

        # NOTE: seed set to be 0 by default
        img_path = self.contour_root + self.paths_dict[index][0]['img_path']
        image = Image.open(img_path).convert('RGB')
        res_img = self.transform(image)

        return res_img, res_pos, img_path

    def __len__(self):
        if self.test_only:
            return len(self.test_imgs)
        return len(self.paths_dict)


def get_dataset(args, test_only=False):
    data_root = args.data_root
    # sketch_root = os.path.join(data_root, 'logs', args.dataset)
    sketch_data_file = os.path.join(data_root, args.sketch_data_file)
    contour_root = os.path.join(data_root, args.contour_dir)
    # data root is only for testing
    test_root = os.path.join(args.data_root, 'test_imgs/')

    image_shape = (3, args.custom_transforms_size, args.custom_transforms_size)

    transform = transforms.Compose([
        transforms.Resize((image_shape[1], image_shape[1])),
        transforms.ToTensor(),
    ])

    test_dataset = SketchDataset(
        args=args,
        sketch_data_file=None,
        data_root=test_root,
        transform=transform,
        test_only=True,
        contour_root=contour_root
    )

    if test_only:
        train_dataset = SketchDataset(
            args=args,
            sketch_data_file=None,
            data_root=contour_root + "0001/",
            transform=transform,
            test_only=True,
            contour_root=contour_root
        )
        return train_dataset, test_dataset, image_shape

    train_dataset = SketchDataset(
        args=args,
        sketch_data_file=sketch_data_file,
        data_root=data_root,
        transform=transform,
        contour_root=contour_root
    )

    if args.train_val_split_ratio > 0:
        train_dataset, val_dataset = split_dataset(train_dataset, args.train_val_split_ratio, seed=args.seed, use_stratify=False)
    else:
        val_dataset = train_dataset

    return train_dataset, val_dataset, test_dataset, image_shape

