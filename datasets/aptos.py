import glob
import os
import cv2

import torch
from torch.utils.data import Dataset
import numpy as np
from .transforms import compose
import pandas as pd
from PIL import Image
import yaml

data_folder =  '/home/ibrahimalmakky/Documents/data/DiabeticRetina/DR/aptos'
csv_file = '/home/ibrahimalmakky/Documents/data/DiabeticRetina/DR/aptos/aptos_dataset_splits.csv'

def get_transforms(mode="train"):
    # preprocessing
    return compose(
        yaml.load(
            open('/home/santoshsanjeev/MedMerge/params/transforms/rsna_pneumonia.yaml', encoding="UTF-8"), Loader=yaml.FullLoader
        )[mode]
    )

class AptosDataset(Dataset):
    def __init__(self, csv_file, data_folder, split, task, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = data_folder
        self.transform = transform
        self.split = split
        self.data = self.data.loc[self.data["split"] == self.split]
        self.task = task
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.data.iloc[idx, 1]), str(self.data.iloc[idx, 0]))
        image = Image.open(img_name)
        label = self.data.iloc[idx, 1]
        if self.task == 'Regression':
            label = np.expand_dims(label, -1)
        if self.transform:
            image = self.transform(image)
        
        return image, label


def prepare_train_loaders(config):
    train_dataset = config['wrapper'](
        csv_file= csv_file, 
        data_folder=data_folder, 
        split="train",
        task="multi-class",
        transform=get_transforms(mode="train"),
    )

    train_loader = {'full': torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle_train'],
        num_workers=config['num_workers'],
    )}

    return train_loader


def prepare_val_loaders(config):
    val_dataset = config['wrapper'](
        csv_file=csv_file, 
        data_folder=data_folder, 
        split="val",
        task="multi-class",
        transform=get_transforms(mode="test"),
    )

    
    val_loader = {'full':torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle_test'],
        num_workers=config['num_workers'],
    )}


    return val_loader





def prepare_test_loaders(config):
    test_dataset = config['wrapper'](
        csv_file= csv_file, 
        data_folder=data_folder, 
        split="test",
        task="multi-class",
        transform=get_transforms(mode="test"),
    )

    test_loader = {'full':torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle_test'],
        num_workers=config['num_workers'],
    )}

    test_loader['class_names'] = ['','','','','']
    return test_loader