import cv2
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.transforms import ToFloat

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler

import copy
import matplotlib.pyplot as plt


class CSVDataset(Dataset):

    def __init__(self,csv_file,transform=None):
        self.csv_file = csv_file
        self.transform = transform
        self._read_csv()

    def _read_csv(self):
        df = pd.read_csv(self.csv_file)
        self.filepaths = df['FILEPATHS'].values
        self.labels = df['LABELS'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        img_path = self.filepaths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        if self.transform:
            img = self.transform(image=img)['image']
        return img,label


def weighted_sampler(labels):
    """
    Create a sampler to ensure that each batch sees a proportional number of all classes"""
    counts = np.unique(labels, return_counts=True)[1]
    weights = 1.0/torch.tensor(counts, dtype=torch.float)
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler


def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def load_data_from_csv(train_csv:list,val_csv:list,input_size=(224,224)):
    tsfm = {
        'train': A.Compose([
            A.Resize(*input_size),
            ToFloat(),
            ToTensorV2()
        ]),
        'val': A.Compose([
            A.Resize(*input_size),
            ToFloat(),
            ToTensorV2()
        ])
    }
    
    train_set = [CSVDataset(csv,transform=tsfm['train']) for csv in train_csv]
    val_set = [CSVDataset(csv,transform=tsfm['val']) for csv in val_csv]

    train_dataset = ConcatDataset(train_set)
    val_dataset = ConcatDataset(val_set)

    train_labels = []
    for _,label in train_dataset:
        train_labels.append(label)
    train_sampler = weighted_sampler(train_labels)

    return train_dataset, val_dataset, train_sampler


def _count_class_samples(samples,labels):
    """Count samples of each class labels"""
    return {labels[i]:j for i, j in zip(*np.unique(samples, return_counts=True))}