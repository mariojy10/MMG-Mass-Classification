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


def load_data_from_csv(train_csv,val_csv,input_size,transform_config):
    tsfm = create_transform(input_size,transform_config)

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


def _collater(batch):
    imgs,labels = [], []
    for img,label in batch:
        imgs.append(img)
        labels.append(label)

    # imgs = [item[0] for item in batch]
    imgs = torch.from_numpy(np.stack(imgs,axis=0))
    imgs = imgs.permute(0,3,1,2)

    # labels = [item[1] for item in batch]
    labels = torch.LongTensor(labels)
    return [imgs,labels]
    

def create_transform(input_size,config_file=None):
    """Compose albu transformations of the given transform config file"""
    from importlib import import_module
    from pathlib import Path
    import sys

    if not config_file.endswith('.py'):
        raise Exception('Unsupported config file format.Supported format are {}'.format('.py'))

    config_file_path = Path(config_file).resolve() 
    if not config_file_path.is_file():
        raise FileNotFoundError('Config file not exists')

    #preparation to import the config file as module
    config_file_directory = config_file_path.parent
    sys.path.insert(0,str(config_file_directory))
    config_file_name = config_file_path.stem

    mod = import_module(config_file_name)    
    mod_dict = {k:v for k,v in mod.__dict__.items() if not k.startswith('__')}

    #clean sys
    sys.path.pop(0)
    del sys.modules[config_file_name]

    #grab transforms
    tsfm = dict()
    for v in mod_dict.values():
        if callable(v):
            tsfm = v()
    
    #make sure to resize the image correctly 
    for k,v in tsfm.items():
        if any(isinstance(t, A.Resize) for t in v):
            for t in v:
                if isinstance(t, A.Resize):
                    t.height, t.width = input_size

    #compose transform
    for k,v in tsfm.items():
        if isinstance(v,(list,tuple)):
            tsfm[k] = A.Compose(v)

    return tsfm
