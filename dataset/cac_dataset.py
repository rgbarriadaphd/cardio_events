"""
# Author: ruben 
# Date: 22/9/22
# Project: CardioEvents
# File: cac_dataset.py

Description: Class that implements CAC Learning dataset
"""
from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms
from PIL import Image, ImageStat
import numpy as np
from constants.path_constants import *
from constants.train_constants import *


def get_custom_normalization():
    """
    Get normalization according to input train dataset
    :return: ((list) mean, (list) std) Normalization values mean and std
    """
    target = os.path.join(CAC_DATASET_FOLDER, TRAIN)

    means = []
    stds = []

    for root, dirs, files in os.walk(target):
        for file in files:
            image_path = os.path.join(root, file)
            assert os.path.exists(image_path)
            with open(image_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                stat = ImageStat.Stat(img)
                local_mean = [stat.mean[0] / 255., stat.mean[1] / 255., stat.mean[2] / 255.]
                local_std = [stat.stddev[0] / 255., stat.stddev[1] / 255., stat.stddev[2] / 255.]
                means.append(local_mean)
                stds.append(local_std)

    return list(np.array(means).mean(axis=0)), list(np.array(stds).mean(axis=0))


class CustomImageFolder(datasets.ImageFolder):
    """
    Custom ImageFolder class. Workaround to swap class index assignment.
    """

    def __init__(self, dataset, dataset_name, transform=None, class_values=None):
        """
        :param dataset: (str) Dataset path
        :param dataset_name: (str) identifier name of the dataset
        :param transform: (torch.transforms) Set of transforms to be applied to input data
        :param class_values: (dict) definition of classes and numeric value used by the model
        """
        self.class_values = class_values
        super(CustomImageFolder, self).__init__(dataset, transform=transform)
        self.dataset_name = dataset_name

    def find_classes(self, root):
        if self.class_values:
            self.class_to_idx = self.class_values

        return self.class_to_idx.keys(), self.class_to_idx

    def __getitem__(self, index):
        """
        Extended method of ImageFolder
        :param index: (int) image index
        :return: Image, label and data info
        """
        sample, label = super(datasets.ImageFolder, self).__getitem__(index)
        return sample, label, index, self.dataset_name


def load_and_transform_data(stage, shuffle=False, mean=None, std=None):
    """
    Loads a dataset and applies the corresponding transformations
    :param stage: (str) Dataset to be loaded based on stage: train, test, validation (if any)
    :param shuffle: (list of int) shuffle samples order within datasets
    :param mean: (list) Normalized mean
    :param std: (list) Normalized std
    """
    assert stage in ['train', 'test']

    if mean is None and std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        print(f'Applying custom normalization: mean={mean}, std={std}')

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset_loaders = []
    for dataset_name, data in DATASETS.items():
        dataset_path = os.path.join(os.path.abspath(DATASETS[dataset_name]['path']), stage)
        custom_class_values = DATASETS[dataset_name]['class_values'] if stage=='train' else {'CACSmas400': 1}
        dataset = CustomImageFolder(dataset_path,
                                    dataset_name,
                                    class_values=custom_class_values,
                                    transform=data_transforms)

        bs = DATASETS[dataset_name]['batch_size'] if stage == 'train' else 1
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=bs,
                                                  shuffle=shuffle,
                                                  num_workers=4)
        print(f'Loaded {len(dataset)} images under {dataset_path}: Classes: {dataset.class_to_idx}')
        dataset_loaders.append(data_loader)

    return dataset_loaders


